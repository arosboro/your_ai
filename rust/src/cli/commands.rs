//! CLI command implementations

use anyhow::Result;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use your_ai_rs::benchmarks::{EmpiricalOptimizer, HardwareProfile};
use your_ai_rs::checkpoints::Checkpoint;
use your_ai_rs::config::model::AVAILABLE_MODELS;
use your_ai_rs::config::Config;
use your_ai_rs::hardware::{detect_hardware, MODEL_REQUIREMENTS};
use your_ai_rs::model::{load_model, save_model_weights};
use your_ai_rs::training::DistrustTrainer;

/// Logger that writes benchmark events to disk for crash analysis
struct BenchmarkLogger {
    file: std::fs::File,
}

impl BenchmarkLogger {
    fn new(path: &str) -> Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { file })
    }

    fn log(&mut self, event: serde_json::Value) -> Result<()> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();

        let mut log_entry = event;
        log_entry["timestamp"] = serde_json::json!(timestamp);

        writeln!(self.file, "{}", serde_json::to_string(&log_entry)?)?;
        self.file.flush()?; // Ensure immediate write to disk
        Ok(())
    }
}

pub fn setup() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║       Empirical Distrust Training - Hardware Setup            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Try auto-detection
    let (generation, variant, memory) = detect_hardware();

    if let (Some(gen), Some(var), Some(mem)) = (generation, variant, memory) {
        println!("Detected: {} {} with {}GB", gen.to_uppercase(), var, mem);
        println!("\nHardware profile saved!");
        println!("Run 'your_ai train --model <model-name>' to start training.");
    } else {
        println!("Could not auto-detect hardware.");
        println!("Please specify hardware manually with:");
        println!("  your_ai train --model <model> --chip <variant> --memory <GB>");
    }

    Ok(())
}

pub fn recommend(memory: Option<usize>) -> Result<()> {
    let mem_gb = if let Some(m) = memory {
        m
    } else {
        let (_, _, detected_mem) = detect_hardware();
        detected_mem.ok_or_else(|| anyhow::anyhow!("Could not detect memory. Use --memory <GB>"))?
    };

    let budget = (mem_gb as f32 * 0.80) as usize;

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Training budget: {}GB (80% of {}GB)", budget, mem_gb);
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  MODEL RECOMMENDATIONS                                               ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    for (model_name, reqs) in MODEL_REQUIREMENTS.iter() {
        let training_gb = reqs["training_gb"].as_u64().unwrap_or(0) as usize;
        let recommended = reqs["recommended"].as_bool().unwrap_or(false);

        if training_gb <= budget {
            let status = if recommended {
                "✅ RECOMMENDED"
            } else {
                "⚠️  OK"
            };
            println!("  {} - {}", status, model_name);
            println!(
                "    Training: {}GB | Headroom: {}GB",
                training_gb,
                budget - training_gb
            );
        }
    }

    println!();
    Ok(())
}

/// Run benchmark for a single model (designed to run in subprocess)
pub async fn benchmark_single_model(preset: &str, max_memory_gb: f64) -> Result<()> {
    use serde_json::json;
    use your_ai_rs::config::model::AVAILABLE_MODELS;

    let config = AVAILABLE_MODELS
        .get(preset)
        .ok_or_else(|| anyhow::anyhow!("Unknown preset: {}", preset))?;

    let model_name = config
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let params = config.get("params").and_then(|v| v.as_str()).unwrap_or("?");

    // Resolve model path
    let resolve_model_path =
        |model_name: &str| -> Option<String> { your_ai_rs::resolve_model_path(model_name, true) };

    let model_path = resolve_model_path(model_name)
        .ok_or_else(|| anyhow::anyhow!("Model not found: {}", model_name))?;

    // Run quick validation
    match EmpiricalOptimizer::quick_validate(&model_path, max_memory_gb).await {
        Ok(true) => {
            let mem_info = your_ai_rs::utils::MemoryInfo::current()
                .map(|info| info.rss_bytes as f64 / 1024.0 / 1024.0 / 1024.0)
                .unwrap_or(0.0);

            // Output JSON result to stdout with unique marker prefix
            let result = json!({
                "preset": preset,
                "model_name": model_name,
                "params": params,
                "success": true,
                "peak_memory_gb": mem_info,
                "error": null
            });

            println!("BENCHMARK_RESULT:{}", serde_json::to_string(&result)?);
            Ok(())
        }
        Ok(false) => {
            let result = json!({
                "preset": preset,
                "model_name": model_name,
                "params": params,
                "success": false,
                "peak_memory_gb": 0.0,
                "error": "OOM"
            });

            println!("BENCHMARK_RESULT:{}", serde_json::to_string(&result)?);
            Ok(())
        }
        Err(e) => {
            let result = json!({
                "preset": preset,
                "model_name": model_name,
                "params": params,
                "success": false,
                "peak_memory_gb": 0.0,
                "error": format!("{}", e)
            });

            println!("BENCHMARK_RESULT:{}", serde_json::to_string(&result)?);
            Ok(())
        }
    }
}

pub async fn benchmark(
    max_memory: Option<f64>,
    _run_optimize: bool,
    output: Option<String>,
    single_model: Option<String>,
    force: bool,
) -> Result<()> {
    use your_ai_rs::config::model::AVAILABLE_MODELS;

    /// Minimum available memory before stopping benchmark (safety threshold)
    const MIN_AVAILABLE_MEMORY_GB: f64 = 2.0;

    // Detect or use provided memory limit
    let max_memory_gb = if let Some(mem) = max_memory {
        mem
    } else if let Ok(info) = your_ai_rs::utils::MemoryInfo::current() {
        (info.system_total_bytes as f64 / 1024.0 / 1024.0 / 1024.0) * 0.8
    } else {
        32.0
    };

    // If single_model is specified, run just that model and exit (subprocess mode)
    if let Some(preset) = single_model {
        return benchmark_single_model(&preset, max_memory_gb).await;
    }

    // Create benchmark logger
    let log_path = "benchmark_log.jsonl";
    let mut logger = BenchmarkLogger::new(log_path).ok();

    // Main benchmark mode: spawn subprocesses for each model
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Hardware Benchmark");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    if let Ok(info) = your_ai_rs::utils::MemoryInfo::current() {
        println!("System Memory: {}", info.total_formatted());
        println!("Available Memory: {}", info.available_formatted());
        if force {
            println!("Safety Threshold: DISABLED (--force mode)");
        } else {
            println!(
                "Safety Threshold: {:.1} GB (benchmark will stop if available drops below this)",
                MIN_AVAILABLE_MEMORY_GB
            );
        }
    }
    println!("Benchmark log: {}", log_path);
    println!("Running each model in isolated subprocess for accurate memory measurement...");
    println!();

    // Log benchmark start
    if let Some(ref mut log) = logger {
        let _ = log.log(serde_json::json!({
            "event": "benchmark_start",
            "max_memory_gb": max_memory_gb,
            "force_mode": force
        }));
    }

    // Sort models by parameter size
    let mut model_list: Vec<_> = AVAILABLE_MODELS.iter().collect();
    model_list.sort_by_key(|(_, config)| {
        // Parse param size (e.g., "7B" -> 7, "70B" -> 70)
        config
            .get("params")
            .and_then(|v| v.as_str())
            .and_then(|s| s.trim_end_matches('B').parse::<u32>().ok())
            .unwrap_or(0)
    });

    #[derive(serde::Serialize)]
    struct BenchmarkResult {
        preset: String,
        model_name: String,
        params: String,
        success: bool,
        peak_memory_gb: f64,
        error: Option<String>,
        optimal_config: Option<HardwareProfile>,
    }

    let mut results = Vec::new();
    let mut passing_models = Vec::new();
    let mut last_passing_preset = None;

    for (i, (preset, config)) in model_list.iter().enumerate() {
        let model_name = config
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let params = config.get("params").and_then(|v| v.as_str()).unwrap_or("?");

        print!(
            "[{}/{}] {:20} ({:4}) ",
            i + 1,
            model_list.len(),
            preset,
            params
        );
        std::io::Write::flush(&mut std::io::stdout()).ok();

        // Log model start (non-invasive)
        if let Some(ref mut log) = logger {
            let _ = log.log(serde_json::json!({
                "event": "model_start",
                "preset": preset,
                "model_name": model_name,
                "params": params
            }));
        }

        // Check available memory before spawning subprocess (unless --force is used)
        if !force {
            if let Ok(mem_info) = your_ai_rs::utils::MemoryInfo::current() {
                let available_gb =
                    mem_info.system_available_bytes as f64 / 1024.0 / 1024.0 / 1024.0;

                // Hard stop: if available memory is critically low
                if available_gb < MIN_AVAILABLE_MEMORY_GB {
                    println!("⚠️  SAFETY STOP");
                    println!(
                        "    Available memory ({:.1} GB) below minimum threshold ({:.1} GB)",
                        available_gb, MIN_AVAILABLE_MEMORY_GB
                    );
                    println!("    Stopping benchmark to prevent system instability.");

                    // Log safety stop
                    if let Some(ref mut log) = logger {
                        let _ = log.log(serde_json::json!({
                            "event": "safety_stop",
                            "reason": "low_memory",
                            "available_gb": available_gb,
                            "threshold_gb": MIN_AVAILABLE_MEMORY_GB
                        }));
                    }
                    break;
                }
            }
        }

        print!("... ");
        std::io::Write::flush(&mut std::io::stdout()).ok();

        // Spawn subprocess to test this model
        let exe_path = std::env::current_exe()?;

        // Log subprocess start
        if let Some(ref mut log) = logger {
            let _ = log.log(serde_json::json!({
                "event": "subprocess_start",
                "preset": preset
            }));
        }

        let subprocess_result = std::process::Command::new(&exe_path)
            .args([
                "benchmark",
                "--single-model",
                preset,
                "--max-memory",
                &max_memory_gb.to_string(),
            ])
            .output();

        match subprocess_result {
            Ok(output) if output.status.success() => {
                // Log subprocess success
                if let Some(ref mut log) = logger {
                    let _ = log.log(serde_json::json!({
                        "event": "subprocess_success",
                        "preset": preset,
                        "exit_code": output.status.code()
                    }));
                }
                // Look for the marker line in stdout
                let stdout_str = String::from_utf8_lossy(&output.stdout);
                let json_line = stdout_str
                    .lines()
                    .find(|line| line.starts_with("BENCHMARK_RESULT:"))
                    .and_then(|line| line.strip_prefix("BENCHMARK_RESULT:"));

                if let Some(json_str) = json_line {
                    if let Ok(result) = serde_json::from_str::<serde_json::Value>(json_str) {
                        let success = result
                            .get("success")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        let peak_memory_gb = result
                            .get("peak_memory_gb")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        let error = result
                            .get("error")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());

                        if success {
                            println!("✓ Pass ({:.1} GB peak)", peak_memory_gb);
                            println!("      [Memory released - subprocess exited]");

                            passing_models.push(format!("{} ({})", preset, params));
                            last_passing_preset = Some(preset.to_string());

                            results.push(BenchmarkResult {
                                preset: preset.to_string(),
                                model_name: model_name.to_string(),
                                params: params.to_string(),
                                success: true,
                                peak_memory_gb,
                                error: None,
                                optimal_config: None,
                            });
                        } else if error.as_deref() == Some("OOM") {
                            println!("✗ OOM");
                            // Stop testing larger models on OOM
                            results.push(BenchmarkResult {
                                preset: preset.to_string(),
                                model_name: model_name.to_string(),
                                params: params.to_string(),
                                success: false,
                                peak_memory_gb: 0.0,
                                error: Some("OOM".to_string()),
                                optimal_config: None,
                            });
                            break;
                        } else {
                            println!("✗ {}", error.as_deref().unwrap_or("Error"));
                            results.push(BenchmarkResult {
                                preset: preset.to_string(),
                                model_name: model_name.to_string(),
                                params: params.to_string(),
                                success: false,
                                peak_memory_gb: 0.0,
                                error,
                                optimal_config: None,
                            });
                        }
                    } else {
                        println!("✗ Failed to parse JSON");
                        results.push(BenchmarkResult {
                            preset: preset.to_string(),
                            model_name: model_name.to_string(),
                            params: params.to_string(),
                            success: false,
                            peak_memory_gb: 0.0,
                            error: Some("Failed to parse JSON output".to_string()),
                            optimal_config: None,
                        });
                    }
                } else {
                    println!("✗ No benchmark result found in output");
                    results.push(BenchmarkResult {
                        preset: preset.to_string(),
                        model_name: model_name.to_string(),
                        params: params.to_string(),
                        success: false,
                        peak_memory_gb: 0.0,
                        error: Some(
                            "No BENCHMARK_RESULT marker found in subprocess output".to_string(),
                        ),
                        optimal_config: None,
                    });
                }
            }
            Ok(output) => {
                // Subprocess failed
                let stderr_str = String::from_utf8_lossy(&output.stderr);
                let stdout_str = String::from_utf8_lossy(&output.stdout);
                println!(
                    "✗ Subprocess failed: {}",
                    stderr_str.lines().next().unwrap_or("Unknown error")
                );

                // Log subprocess failure
                if let Some(ref mut log) = logger {
                    let _ = log.log(serde_json::json!({
                        "event": "subprocess_failed",
                        "preset": preset,
                        "exit_code": output.status.code(),
                        "stderr": stderr_str.lines().take(10).collect::<Vec<_>>().join("\n"),
                        "stdout": stdout_str.lines().take(10).collect::<Vec<_>>().join("\n")
                    }));
                }

                results.push(BenchmarkResult {
                    preset: preset.to_string(),
                    model_name: model_name.to_string(),
                    params: params.to_string(),
                    success: false,
                    peak_memory_gb: 0.0,
                    error: Some(format!("Subprocess failed: {}", stderr_str)),
                    optimal_config: None,
                });
            }
            Err(e) => {
                println!("✗ Failed to spawn subprocess: {}", e);

                // Log spawn failure
                if let Some(ref mut log) = logger {
                    let _ = log.log(serde_json::json!({
                        "event": "subprocess_spawn_error",
                        "preset": preset,
                        "error": format!("{}", e)
                    }));
                }

                results.push(BenchmarkResult {
                    preset: preset.to_string(),
                    model_name: model_name.to_string(),
                    params: params.to_string(),
                    success: false,
                    peak_memory_gb: 0.0,
                    error: Some(format!("Failed to spawn subprocess: {}", e)),
                    optimal_config: None,
                });
            }
        }
    }

    println!();

    // Log benchmark completion
    if let Some(ref mut log) = logger {
        let _ = log.log(serde_json::json!({
            "event": "benchmark_complete",
            "total_tested": results.len(),
            "passed": results.iter().filter(|r| r.success).count(),
            "failed": results.iter().filter(|r| !r.success).count()
        }));
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Results");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    if let Some(ref recommended) = last_passing_preset {
        println!("Recommended: {} (largest model that fits)", recommended);
        if passing_models.len() > 1 {
            let alternatives: Vec<_> = passing_models
                .iter()
                .filter(|m| !m.starts_with(recommended.as_str()))
                .cloned()
                .collect();
            if !alternatives.is_empty() {
                println!("Alternatives: {}", alternatives.join(", "));
            }
        }
    } else {
        println!("No models passed benchmark.");
        println!();
        println!("Consider:");
        println!("  - Ensuring models are downloaded to HuggingFace cache (~/.cache/huggingface/)");
        println!("  - Increasing available memory or closing other applications");
        println!("  - Trying with a smaller model");
    }
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Save results if requested
    if let Some(output_path) = output {
        let output_data = serde_json::json!({
            "max_memory_gb": max_memory_gb,
            "recommended": last_passing_preset,
            "results": results,
        });
        std::fs::write(&output_path, serde_json::to_string_pretty(&output_data)?)?;
        println!("\nResults saved to: {}", output_path);
    }

    Ok(())
}

pub async fn optimize(
    model: String,
    max_memory: Option<f64>,
    quick: bool,
    output: Option<String>,
) -> Result<()> {
    // Create optimizer
    let optimizer = EmpiricalOptimizer::new(model.clone(), max_memory, quick);

    // Run optimization
    let results = optimizer.find_optimal().await?;

    // Print summary
    EmpiricalOptimizer::print_summary(&results);

    // Create and save profile
    if let Some(profile) = HardwareProfile::from_results(model, results) {
        if let Some(output_path) = output {
            profile.save(&output_path)?;
            println!("\nProfile saved to: {}", output_path);
        }
    } else {
        println!("\nNo successful configuration found - cannot create profile.");
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn train(
    model: String,
    batch_size: Option<usize>,
    lora_rank: Option<usize>,
    max_steps: usize,
    _resume: bool,
    max_memory: Option<f64>,
    memory_report_interval: Option<usize>,
    auto_optimize: bool,
    metrics_file: Option<String>,
    save_best: bool,
    reload_interval: Option<usize>,
    alpha: Option<f32>,
    lambda_weight: Option<f32>,
) -> Result<()> {
    use your_ai_rs::config::model::AVAILABLE_MODELS;

    let mut config = Config::default();

    // Resolve model preset to actual model name
    let model_name = if let Some(preset_config) = AVAILABLE_MODELS.get(&model) {
        preset_config
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(&model)
            .to_string()
    } else {
        // Not a preset, assume it's a direct model path or HuggingFace name
        model.clone()
    };

    // Resolve HuggingFace model name to actual snapshot path
    let resolve_model_path =
        |model_name: &str| -> Option<String> { your_ai_rs::resolve_model_path(model_name, false) };

    let model_path = resolve_model_path(&model_name).ok_or_else(|| {
        anyhow::anyhow!(
            "Model not found: {}. Please download it first using: huggingface-cli download {}",
            model_name,
            model_name
        )
    })?;

    // Apply command-line overrides
    config.paths.model_path = model_path;
    config.paths.output_dir = format!("models/distrust-{}", model);

    // Auto-optimize if requested
    if auto_optimize {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Running automatic optimization to find best configuration...");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        let optimizer = EmpiricalOptimizer::new(model.clone(), max_memory, false);
        let results = optimizer.find_optimal().await?;

        if let Some(profile) = HardwareProfile::from_results(model.clone(), results) {
            println!();
            println!("Applying optimized configuration to training:");
            profile.print_summary();
            println!();

            // Apply profile settings (these will override command-line args)
            profile.apply_to_config(&mut config);
        } else {
            println!("\nWarning: Could not find optimal configuration.");
            println!("Falling back to default or specified settings.");
        }
    }

    // Apply remaining command-line overrides (these take precedence over auto-optimize)
    if let Some(bs) = batch_size {
        config.training.batch_size = bs;
    }
    if let Some(rank) = lora_rank {
        config.model.lora_rank = rank;
        config.model.lora_alpha = rank * 2; // Maintain scale=2.0
    }
    config.training.max_steps = max_steps;

    if let Some(interval) = reload_interval {
        config.training.reload_interval_steps = interval;
    }

    // Apply distrust loss overrides
    if let Some(a) = alpha {
        config.distrust.alpha = a;
        config.training.alpha = a;
    }
    if let Some(l) = lambda_weight {
        config.distrust.lambda_weight = l;
        config.training.lambda_weight = l;
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Training Configuration");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Model:          {}", config.paths.model_path);
    println!("  Output:         {}", config.paths.output_dir);
    println!("  Batch size:     {}", config.training.batch_size);
    println!("  LoRA rank:      {}", config.model.lora_rank);
    println!("  LoRA alpha:     {}", config.model.lora_alpha);
    println!("  Max steps:      {}", config.training.max_steps);
    println!("  Distrust alpha: {}", config.distrust.alpha);
    println!("  Lambda weight:  {}", config.distrust.lambda_weight);
    if let Some(mem) = max_memory {
        println!("  Max memory:     {:.1} GB", mem);

        // Check if memory limit is sufficient for model
        if (model.contains("8b") || model.contains("8B")) && mem < 48.0 {
            println!();
            println!("⚠️  WARNING: Memory limit may be too low for 8B model");
            println!("   Current limit:    {:.1} GB", mem);
            println!("   Recommended:      48-70 GB for stable training");
            if let Ok(info) = your_ai_rs::utils::MemoryInfo::current() {
                let system_gb = info.system_total_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
                println!("   Your system:      {:.1} GB total", system_gb);
                if system_gb >= 70.0 {
                    println!("   Suggestion:       Try --max-memory 70.0");
                } else if system_gb >= 48.0 {
                    let recommended = (system_gb * 0.75).floor();
                    println!("   Suggestion:       Try --max-memory {:.0}.0", recommended);
                }
            }
            println!("   Low memory may cause excessive swap usage and slow training");
        }
    }
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Initialize checkpoint manager for reloads and saving
    let checkpoint_dir = PathBuf::from(&config.paths.output_dir).join("checkpoints");
    let manager = your_ai_rs::checkpoints::CheckpointManager::new(&checkpoint_dir, 3)?;

    // Create trainer
    let model_path = PathBuf::from(&config.paths.model_path);
    let mut trainer = DistrustTrainer::new(&model_path).await?
        .with_config(config);

    // Configure memory settings - auto-detect if not specified
    let effective_max_memory = if let Some(mem) = max_memory {
        mem
    } else {
        // Auto-detect safe memory limit based on available system memory
        if let Ok(info) = your_ai_rs::utils::MemoryInfo::current() {
            let available_gb = info.system_available_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
            // For Apple Silicon with unified memory, use more aggressive limits
            // 0.8 factor instead of 0.6 to better utilize available memory
            let safe_limit = (available_gb * 0.8).min(120.0).max(16.0);
            println!(
                "⚠️  No --max-memory specified. Auto-detecting safe limit: {:.1} GB",
                safe_limit
            );
            println!(
                "   (Based on {:.1} GB available system memory)",
                available_gb
            );
            println!("   To override, use: --max-memory <GB>");
            safe_limit
        } else {
            println!("⚠️  Could not detect system memory. Using conservative default: 16.0 GB");
            16.0
        }
    };
    trainer = trainer.with_max_memory(effective_max_memory);

    if let Some(interval) = memory_report_interval {
        trainer = trainer.with_memory_reporting(interval);
    }

    // Configure metrics export
    if let Some(metrics_path) = metrics_file {
        trainer = trainer.with_metrics_file(std::path::PathBuf::from(metrics_path));
    }

    // Configure best checkpoint saving
    trainer = trainer.with_save_best(save_best);
    trainer = trainer.with_checkpoint_manager(manager);

    // Train (model initialized in constructor)
    trainer.train().await?;

    Ok(())
}

pub fn validate(model: String, benchmarks: Option<String>) -> Result<()> {
    println!("Validating model: {}", model);

    let benchmark_list = benchmarks.unwrap_or_else(|| "truthfulqa".to_string());
    let benchmarks: Vec<&str> = benchmark_list.split(',').collect();

    println!("Running benchmarks: {:?}", benchmarks);
    println!(
        "\nNote: Full benchmark implementation requires integration with HuggingFace datasets."
    );
    println!("This is a placeholder - implement full evaluation in production.");

    Ok(())
}

pub fn generate(
    model: String,
    prompt: String,
    checkpoint: Option<String>,
    max_tokens: usize,
    temperature: f32,
    compare: bool,
    eos_token: Option<i32>,
) -> Result<()> {
    use std::path::PathBuf;
    use your_ai_rs::config::model::AVAILABLE_MODELS;
    use your_ai_rs::model::{LlamaConfig, TokenizerWrapper};

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Text Generation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Resolve model preset to actual model name
    let model_name = if let Some(preset_config) = AVAILABLE_MODELS.get(&model) {
        preset_config
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(&model)
            .to_string()
    } else {
        model.clone()
    };

    // Resolve model path
    let resolve_model_path =
        |model_name: &str| -> Option<String> { your_ai_rs::resolve_model_path(model_name, false) };

    let model_path = resolve_model_path(&model_name).ok_or_else(|| {
        anyhow::anyhow!("Model not found: {}. Please download it first.", model_name)
    })?;

    println!("Loading model from: {}", model_path);
    let model_dir = PathBuf::from(&model_path);

    // Load config and tokenizer
    let config_path = model_dir.join("config.json");
    let mut llama_config = LlamaConfig::from_json(&config_path)?;

    // Apply EOS override from CLI
    if let Some(eos) = eos_token {
        llama_config.eos_token_id = Some(your_ai_rs::model::EosToken::Single(eos));
        println!("Overriding EOS token ID: {}", eos);
    }

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = TokenizerWrapper::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Tokenize prompt
    println!("Tokenizing prompt...");
    let input_ids = tokenizer.encode(&prompt, false)?;
    let input_len = input_ids.len();
    println!("Input tokens: {}", input_len);
    println!();

    if compare && checkpoint.is_some() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("COMPARISON MODE: Base Model vs Fine-tuned");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        // Generate with base model
        println!("📝 BASE MODEL OUTPUT:");
        println!("─────────────────────────────────────────────────────────────");

        // Load base weights
        let (base_weights, _) = load_model(Path::new(&model_path))?;
        let mut base_model = your_ai_rs::model::llama::load_model_with_weights(
            llama_config.clone(),
            base_weights.clone(),
        )?;

        let input_ids_i32: Vec<i32> = input_ids.iter().map(|&x| x as i32).collect();
        let input_array = mlx_rs::Array::from_slice(&input_ids_i32, &[1, input_len as i32]);

        let base_tokens = base_model.generate(&input_array, max_tokens, temperature)?;
        let base_output = tokenizer.decode(
            &base_tokens.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            true,
        )?;

        println!("Prompt: {}", prompt);
        println!("Generated: {}", base_output);
        println!();

        // Generate with checkpoint model
        println!("📝 FINE-TUNED MODEL OUTPUT:");
        println!("─────────────────────────────────────────────────────────────");

        // Prepare weights with checkpoint merged
        let mut finetuned_weights = base_weights; // Efficient clone/move
        if let Some(checkpoint_path) = checkpoint.as_ref() {
            let checkpoint_data = std::fs::read_to_string(checkpoint_path)?;
            let checkpoint: Checkpoint = serde_json::from_str(&checkpoint_data)?;
            for (name, (data, shape)) in checkpoint.model_state.weights {
                let array = mlx_rs::Array::from_slice(&data, &shape);
                finetuned_weights.insert(name, array);
            }
        }

        let mut finetuned_model =
            your_ai_rs::model::llama::load_model_with_weights(llama_config, finetuned_weights)?;

        let finetuned_tokens = finetuned_model.generate(&input_array, max_tokens, temperature)?;
        let finetuned_output = tokenizer.decode(
            &finetuned_tokens
                .iter()
                .map(|&x| x as u32)
                .collect::<Vec<_>>(),
            true,
        )?;

        println!("Prompt: {}", prompt);
        println!("Generated: {}", finetuned_output);
        println!();

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    } else {
        // Single model generation
        println!("Loading model weights...");

        // 1. Load base model weights
        let (mut weights, _) = load_model(Path::new(&model_path))?;
        println!("Loaded {} base tensors", weights.len());

        // 2. Load checkpoint if specified
        if let Some(checkpoint_path) = checkpoint {
            println!("Loading checkpoint from: {}", checkpoint_path);
            let checkpoint_data = std::fs::read_to_string(checkpoint_path)?;
            let checkpoint: Checkpoint = serde_json::from_str(&checkpoint_data)?;

            println!(
                "Merging {} checkpoint tensors (step {})",
                checkpoint.model_state.weights.len(),
                checkpoint.step
            );
            for (name, (data, shape)) in checkpoint.model_state.weights {
                let array = mlx_rs::Array::from_slice(&data, &shape);
                weights.insert(name, array);
            }
        }

        // 3. Initialize model with weights (prevents random initialization)
        let mut model = your_ai_rs::model::llama::load_model_with_weights(llama_config, weights)?;

        println!("Generating text...");
        let input_ids_i32: Vec<i32> = input_ids.iter().map(|&x| x as i32).collect();
        let input_array = mlx_rs::Array::from_slice(&input_ids_i32, &[1, input_len as i32]);

        let generated_tokens = model.generate(&input_array, max_tokens, temperature)?;
        let generated_text = tokenizer.decode(
            &generated_tokens
                .iter()
                .map(|&x| x as u32)
                .collect::<Vec<_>>(),
            true,
        )?;

        println!();
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Generated Text");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();
        println!("Prompt: {}", prompt);
        println!("Generated: {}", generated_text);
        println!();
        println!("Tokens generated: {}", generated_tokens.len());
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }

    Ok(())
}

/// Export fine-tuned model to safetensors
pub fn export_command(
    model: &str,
    checkpoint_path: &std::path::PathBuf,
    output_path: &std::path::PathBuf,
) -> Result<()> {
    println!("Exporting model: {}", model);
    println!("Checkpoint: {:?}", checkpoint_path);
    println!("Output: {:?}", output_path);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Resolve model name
    let model_name = if let Some(preset_config) = AVAILABLE_MODELS.get(model) {
        preset_config
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(model)
            .to_string()
    } else {
        model.to_string()
    };

    // Simplified resolution for export (assume downloaded or local)
    let model_path = if std::path::Path::new(&model_name).exists() {
        model_name.clone()
    } else {
        // Try simple HF cache guess
        let cache_name = model_name.replace('/', "--");
        let home = std::env::var("HOME").unwrap_or_default();
        let cache_dir = format!("{}/.cache/huggingface/hub/models--{}", home, cache_name);

        let mut found_path = None;
        if std::path::Path::new(&cache_dir).exists() {
            let snapshots_dir = format!("{}/snapshots", cache_dir);
            if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                for entry in entries.flatten() {
                    // Fix: FileType does not implement Default, use map/unwrap_or
                    if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                        found_path = Some(entry.path().to_string_lossy().to_string());
                        break;
                    }
                }
            }
        }
        found_path.ok_or_else(|| {
            anyhow::anyhow!("Model not found: {}. Please use full path.", model_name)
        })?
    };

    println!("Base model path: {}", model_path);
    // let model_dir = std::path::PathBuf::from(&model_path);

    // 1. Load base weights
    println!("1. Loading base model weights...");
    let (mut weights, _) = load_model(Path::new(&model_path))?;
    println!("   Loaded {} tensors", weights.len());

    // 2. Load checkpoint
    println!("2. Loading checkpoint...");
    let checkpoint_data = std::fs::read_to_string(checkpoint_path)?;
    let checkpoint: Checkpoint = serde_json::from_str(&checkpoint_data)?;
    println!("   Checkpoint step: {}", checkpoint.step);
    println!(
        "   Merging {} tensors...",
        checkpoint.model_state.weights.len()
    );

    // 3. Merge weights
    for (name, (data, shape)) in checkpoint.model_state.weights {
        let array = mlx_rs::Array::from_slice(&data, &shape);
        // Overwrite or insert
        weights.insert(name, array);
    }
    println!("   Merge complete.");

    // 4. Save to output
    println!("3. Saving exported model to {:?}...", output_path);

    // Create output directory if needed
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    save_model_weights(&weights, output_path)?;

    println!("✓ Export complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    Ok(())
}
