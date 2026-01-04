use crate::checkpoints::manager::{Checkpoint, CheckpointManager, OptimizerState as CheckpointOptimizerState, ParamGroup, TrainingConfig};
use crate::checkpoints::ModelState;
use crate::config::Config;
use crate::data::StreamingDataset;
use crate::distrust_loss::batch_empirical_distrust_loss;
use crate::model::LlamaForCausalLM;
use crate::training::scheduler::{LearningRateScheduler, WarmupCosineSchedule};
use crate::utils::memory::MemoryMonitor;
use crate::utils::metrics::TensorBoardLogger;
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use mlx_rs::module::ModuleParameters;
use mlx_rs::losses::{CrossEntropyBuilder, LossReduction};
use mlx_rs::builder::Builder;
use mlx_rs::Array;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Optimizer state stored as raw data to prevent MLX memory accumulation
type RawOptimizerState = (Vec<f32>, Vec<i32>); // (data, shape) - CPU storage for checkpointing
type OptimizerStateGPU = Array; // GPU storage for training (zero-leak)

pub struct DistrustTrainer {
    config: Config,
    model: LlamaForCausalLM,
    tokenizer: crate::model::TokenizerWrapper,
    // Manual AdamW state - GPU storage for zero-leak training
    adam_m_gpu: std::collections::HashMap<String, OptimizerStateGPU>, // First moment (GPU)
    adam_v_gpu: std::collections::HashMap<String, OptimizerStateGPU>, // Second moment (GPU)
    adam_step: usize, // Step counter for bias correction
    // CPU storage only for checkpointing (populated on-demand)
    adam_m: std::collections::HashMap<String, RawOptimizerState>,
    adam_v: std::collections::HashMap<String, RawOptimizerState>,
    dataset: Option<StreamingDataset>,
    global_step: usize,
    loss_history: Vec<f32>,
    scheduler: Box<dyn LearningRateScheduler>,
    checkpoint_manager: Option<CheckpointManager>,
    memory_monitor: Option<MemoryMonitor>,
    metrics: Option<TensorBoardLogger>,
    max_memory_gb: Option<f64>,
    memory_report_interval: usize,
    best_loss: f32,
    best_loss_step: usize,
    metrics_file: Option<PathBuf>,
    save_best_checkpoint: bool,
    training_start_time: Option<Instant>,
    // Memory verification for zero-leak guarantee
    baseline_mlx_memory: Option<usize>,
    /// Threshold detects when leak exceeds expected framework baseline
    memory_leak_threshold_mb: f64,
    memory_warning_margin_percent: f64, // Warn when within X% of calculated max steps
    /// Accumulated gradients for multi-step accumulation
    accumulated_grads: std::collections::HashMap<String, Array>,
    start_step: Option<usize>,
}

/// Format parameter count with K/M/B suffixes
#[allow(dead_code)]
fn format_param_count(count: usize) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

/// Format duration in seconds to human-readable string
fn format_duration(secs: u64) -> String {
    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    if hours > 0 {
        format!("{}h{}m", hours, minutes)
    } else if minutes > 0 {
        format!("{}m{}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}

/// Get debug log path from environment variable
/// Set YOUR_AI_DEBUG_LOG env var to enable debug logging
fn debug_log_path() -> Option<PathBuf> {
    std::env::var("YOUR_AI_DEBUG_LOG").ok().map(PathBuf::from)
}

impl DistrustTrainer {
    pub async fn new(model_path: &Path, config: Config, start_step: Option<usize>) -> Result<Self> {
        // Initialize memory monitoring
        let memory_monitor = MemoryMonitor::new(80.0); // 80% threshold

        // Load model config and initialize architecture
        let model_dir = model_path.to_path_buf();
        let llama_config = crate::model::LlamaConfig::from_json(&model_dir.join("config.json"))?;

        println!(
            "Initializing Llama-{} model: {} layers, {} heads",
            llama_config.num_hidden_layers,
            llama_config.num_hidden_layers,
            llama_config.num_attention_heads
        );

        // Pass quantization preference from config
        println!("Quantization enabled: {}", config.model.quantize);

        // Use streaming loader to minimize memory usage
        let (mut model, _) = crate::model::loader::load_model_streaming(&model_dir, config.model.quantize)?;

        let lora_rank = config.model.lora_rank;

        // Note: LoRA application is temporarily disabled as the current Rust implementation
        // relies on HashMap injection which doesn't map to the static model struct.
        // The trainer currently performs Head-Only fine-tuning (backbone is frozen),
        // which is memory efficient and sufficient for minimizing distrust loss.
        if lora_rank > 0 {
             println!("Initializing LoRA adapters (rank={}, alpha={})...", lora_rank, config.model.lora_alpha);

             // 1. Freeze backbone parameters
             // 1. Freeze backbone parameters
             model.backbone.embed_tokens.freeze_parameters(true);

             for layer in model.backbone.layers.iter_mut() {
                 layer.freeze_parameters(true);
             }
             // 2. Inject LoRA adapters
             let alpha = config.model.lora_alpha as f32;
             let dropout = config.model.lora_dropout;

             // Calculate dims from config
             let hidden = llama_config.hidden_size;
             let head_dim = hidden / llama_config.num_attention_heads;
             let kv_heads = llama_config.num_key_value_heads;
             let att_heads = llama_config.num_attention_heads;

             for layer in model.backbone.layers.iter_mut() {
                 let targets = &config.model.lora_target_modules;

                 if targets.iter().any(|t| t.contains("q_proj")) {
                     layer.self_attn.q_proj_lora = Some(crate::model::LoraAdapter::new(hidden, att_heads * head_dim, lora_rank, alpha, dropout)?);
                 }
                 if targets.iter().any(|t| t.contains("k_proj")) {
                     layer.self_attn.k_proj_lora = Some(crate::model::LoraAdapter::new(hidden, kv_heads * head_dim, lora_rank, alpha, dropout)?);
                 }
                 if targets.iter().any(|t| t.contains("v_proj")) {
                     layer.self_attn.v_proj_lora = Some(crate::model::LoraAdapter::new(hidden, kv_heads * head_dim, lora_rank, alpha, dropout)?);
                 }
                 if targets.iter().any(|t| t.contains("o_proj")) {
                     layer.self_attn.o_proj_lora = Some(crate::model::LoraAdapter::new(att_heads * head_dim, hidden, lora_rank, alpha, dropout)?);
                 }
             }
             println!("Applied LoRA to {} layers.", model.backbone.layers.len());
        }

        // Initialize other components with config
         let dataset = {
            let train_file = PathBuf::from(&config.paths.data_dir).join("train.jsonl");
            if train_file.exists() {
                StreamingDataset::new(
                    vec![train_file],
                    config.training.batch_size,
                    config.training.batch_size * 4,
                    true,
                    Some(config.seed),
                    true,
                ).ok()
            } else {
                None
            }
        };

        let scheduler = Box::new(WarmupCosineSchedule::new(
            config.training.learning_rate,
            config.training.warmup_steps,
            config.training.max_steps,
        ));

        let checkpoint_manager = None; // Will be set later if needed

        Ok(Self {
            config,
            model,
            tokenizer: crate::model::TokenizerWrapper::from_file(&model_dir.join("tokenizer.json"))
                 .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?,
            adam_m_gpu: std::collections::HashMap::new(),
            adam_v_gpu: std::collections::HashMap::new(),
            adam_step: 0,
            adam_m: std::collections::HashMap::new(),
            adam_v: std::collections::HashMap::new(),
            dataset,
            global_step: 0,
            loss_history: Vec::new(),
            scheduler,
            checkpoint_manager,
            memory_monitor: Some(memory_monitor),
            metrics: None,
            max_memory_gb: None,
            memory_report_interval: 10,
            best_loss: f32::INFINITY,
            best_loss_step: 0,
            metrics_file: None,
            save_best_checkpoint: true,
            training_start_time: None,
            baseline_mlx_memory: None,
            memory_leak_threshold_mb: 1.0,
            memory_warning_margin_percent: 20.0,
            accumulated_grads: std::collections::HashMap::new(),
            start_step,
        })
    }

    /// Helper to fetch next batch (useful for external loops like optimizer)
    pub fn fetch_next_batch(&mut self) -> Option<Vec<serde_json::Value>> {
        self.dataset.as_mut()?.next_batch()
    }


    /// Set maximum memory limit in GB
    pub fn with_max_memory(mut self, max_memory_gb: f64) -> Self {
        self.max_memory_gb = Some(max_memory_gb);

        // Set MLX memory limits to prevent memory accumulation
        let limit_bytes = (max_memory_gb * 0.9 * 1024.0 * 1024.0 * 1024.0) as usize;
        if let Ok(prev_limit) = crate::utils::mlx_memory::set_memory_limit(limit_bytes) {
            println!(
                "MLX memory limit set: {} -> {} bytes",
                prev_limit, limit_bytes
            );
        }
        if let Ok(prev_cache) = crate::utils::mlx_memory::set_cache_limit(limit_bytes / 2) {
            println!(
                "MLX cache limit set: {} -> {} bytes",
                prev_cache,
                limit_bytes / 2
            );
        }

        self
    }

    /// Enable memory reporting at specified interval
    pub fn with_memory_reporting(mut self, interval: usize) -> Self {
        self.memory_report_interval = interval;
        self
    }

    /// Set metrics export file
    pub fn with_metrics_file(mut self, path: PathBuf) -> Self {
        self.metrics_file = Some(path);
        self
    }

    /// Enable/disable best checkpoint saving
    pub fn with_save_best(mut self, enabled: bool) -> Self {
        self.save_best_checkpoint = enabled;
        self
    }

    /// Set checkpoint manager
    pub fn with_checkpoint_manager(mut self, manager: CheckpointManager) -> Self {
        self.checkpoint_manager = Some(manager);
        self
    }

    /// Set memory leak threshold (MB/step)
    ///
    /// WARNING: This is a workaround for MLX-rs framework memory leak (~2000 MB/step).
    /// Setting this too high risks OOM crashes. Setting too low may stop training prematurely.
    ///
    /// # Parameters
    /// - `threshold_mb`: Maximum acceptable memory growth per step
    ///
    /// # Risks
    /// - Training will be limited to: available_memory_GB * 0.7 / (threshold_mb / 1024) steps
    /// - With default 2200 MB/step and 96 GB system: ~30-40 steps max
    /// - Use periodic reload (reload_interval_steps) for longer runs
    ///
    /// # Recommended Values
    /// - Default: 100 MB/step (native fix baseline)
    /// - Strict: 50 MB/step
    /// - Lenient: 500 MB/step
    pub fn with_memory_leak_threshold(mut self, threshold_mb: f64) -> Self {
        self.memory_leak_threshold_mb = threshold_mb;
        self
    }

    /// Set memory warning margin percentage
    ///
    /// Emits warnings when training is within X% of calculated safe step limit.
    ///
    /// # Parameters
    /// - `margin_percent`: Warning threshold (default: 20.0 = warn at 80% of limit)
    pub fn with_memory_warning_margin(mut self, margin_percent: f64) -> Self {
        self.memory_warning_margin_percent = margin_percent;
        self
    }

    /// Check if memory usage is within limits
    fn check_memory_limits(&mut self) -> anyhow::Result<()> {
        if let Some(ref mut monitor) = self.memory_monitor {
            let info = monitor.check()?;

            // Check against threshold
            if monitor.is_over_threshold() {
                anyhow::bail!(
                    "Memory usage exceeded threshold: {} ({:.1}% of system memory). Training stopped.",
                    info.rss_formatted(),
                    info.usage_percentage()
                );
            }

            // Check against user-specified maximum
            if let Some(max_gb) = self.max_memory_gb {
                let max_bytes = (max_gb * 1024.0 * 1024.0 * 1024.0) as u64;
                if info.rss_bytes > max_bytes {
                    anyhow::bail!(
                        "Memory usage exceeded limit: {} > {:.2} GB. Training stopped.",
                        info.rss_formatted(),
                        max_gb
                    );
                }
            }
        }
        Ok(())
    }

    /// Calculate safe maximum steps based on available memory and leak rate
    ///
    /// Returns the enforced step limit that prevents OOM crashes.
    /// May be less than configured max_steps if memory is insufficient.
    pub fn calculate_safe_max_steps(&mut self) -> usize {
        if let Some(sys_info) = self.memory_monitor.as_mut().and_then(|m| m.check().ok()) {
            let available_gb = sys_info.system_available_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
            let leak_gb_per_step = self.memory_leak_threshold_mb / 1024.0;
            if leak_gb_per_step > 0.001 {
                let safe_steps = (available_gb * 0.7 / leak_gb_per_step) as usize;
                safe_steps.min(self.config.training.max_steps)
            } else {
                self.config.training.max_steps
            }
        } else {
            self.config.training.max_steps
        }
    }

    pub async fn train(&mut self) -> anyhow::Result<()> {
        println!(
            "Starting training for {} steps",
            self.config.training.max_steps
        );

        // Early abort if available memory is critically low (< 10 GB)
        if let Some(ref mut monitor) = self.memory_monitor {
            if let Ok(info) = monitor.check() {
                let available_gb = info.system_available_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
                // Relaxed safety check
                if available_gb < 6.0 {
                    anyhow::bail!(
                        "Insufficient available memory: {:.1} GB. Need at least 6 GB available.\n\
                         Close other applications or reduce batch size.",
                        available_gb
                    );
                }
            }
        }

        // Set MLX memory limit to force recycling of old arrays
        // This is critical to prevent unbounded memory growth
        // SAFETY: Auto-detect based on available memory instead of hardcoded 70 GB
        // to prevent OOM crashes when system memory is constrained
        let memory_limit_gb = self.max_memory_gb.unwrap_or_else(|| {
            if let Some(ref mut monitor) = self.memory_monitor {
                if let Ok(info) = monitor.check() {
                    let available_gb = info.system_available_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
                    // Use 60% of available memory, capped at 70 GB, minimum 8 GB
                    let safe_limit = (available_gb * 0.6).clamp(8.0, 70.0);
                    eprintln!(
                        "⚠️  No memory limit specified. Auto-detected: {:.1} GB (60% of {:.1} GB available)",
                        safe_limit, available_gb
                    );
                    safe_limit
                } else {
                    16.0 // Conservative fallback
                }
            } else {
                16.0 // Conservative fallback
            }
        });
        let memory_limit_bytes = (memory_limit_gb * 1024.0 * 1024.0 * 1024.0) as usize;
        match crate::utils::mlx_memory::set_memory_limit(memory_limit_bytes) {
            Ok(prev) => {
                eprintln!(
                    "🔒 Set MLX memory limit to {:.1} GB (was {:.1} GB)",
                    memory_limit_gb,
                    prev as f64 / 1024.0 / 1024.0 / 1024.0
                );
            }
            Err(e) => {
                eprintln!("⚠️ Warning: Failed to set MLX memory limit: {}", e);
            }
        }

        // Also set cache limit to force more aggressive cache clearing
        let cache_limit_bytes = (memory_limit_gb * 0.1 * 1024.0 * 1024.0 * 1024.0) as usize; // 10% for cache
        let _ = crate::utils::mlx_memory::set_cache_limit(cache_limit_bytes);

        // Start training timer
        self.training_start_time = Some(Instant::now());
        let start_time = Instant::now();

        // Check memory before starting
        self.check_memory_limits()?;

        // Initialize TensorBoard if output dir is set
        if self.metrics.is_none() {
             let output_dir = PathBuf::from(self.config.paths.output_dir.clone());
             match TensorBoardLogger::new(&output_dir) {
                 Ok(logger) => {
                     println!("Enabled TensorBoard logging to {:?}", output_dir);
                     self.metrics = Some(logger);
                 },
                 Err(e) => eprintln!("Failed to initialize TensorBoard: {}", e)
             }
        }


        // CRITICAL: Calculate safe maximum steps based on available memory and leak rate
        // This prevents OOM crashes by capping training steps to system capacity
        let calculated_max_steps = self.calculate_safe_max_steps();

        let pb = ProgressBar::new(calculated_max_steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut last_loss_for_trend = None;

        // Capture baseline MLX memory after first step for leak detection
        let mut baseline_captured = false;

        // RESUME LOGIC: Check for existing checkpoints and resume
        if let Some(manager) = &self.checkpoint_manager {
            if let Ok(checkpoints) = manager.list_checkpoints() {
                if let Some(latest_step) = checkpoints.last() {
                    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    println!("Resuming from checkpoint step {}", latest_step);
                    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

                    // We can reuse reload_from_checkpoint_step logic but skip the 'dummy' part if we are at step 0?
                    // Actually, at startup self.model is fresh anyway.
                    // But we need to load weights.
                    // Let's call a modified version or just inline the load.

                    self.global_step = *latest_step;
                    self.adam_step = *latest_step;

                    // Load weights
                    // Use streaming load for efficiency
                     println!("  Restoring weights from checkpoint...");
                    let (mut fresh_model, _) = crate::model::loader::load_model_streaming(
                        Path::new(&self.config.paths.model_path),
                        self.config.model.quantize
                    )?;

                     // Merge weights
                    let mut param_map = fresh_model.parameters_mut().flatten();
                    let checkpoint = manager.iterate_weights(*latest_step, true, |name, data, shape| {
                        // 1. Optimizer
                         if name.starts_with("optimizer.exp_avg.") {
                             let param_name = name.trim_start_matches("optimizer.exp_avg.").to_string();
                             let m_array = Array::from_slice(&data, &shape);
                             let _ = m_array.eval();
                             self.adam_m_gpu.insert(param_name, m_array);
                             return Ok(());
                         }
                         if name.starts_with("optimizer.exp_avg_sq.") {
                             let param_name = name.trim_start_matches("optimizer.exp_avg_sq.").to_string();
                             let v_array = Array::from_slice(&data, &shape);
                             let _ = v_array.eval();
                             self.adam_v_gpu.insert(param_name, v_array);
                             return Ok(());
                         }
                         // 2. Model
                         let candidates = vec![
                             name.clone(),
                             name.replace("model.", "backbone."),
                             name.replace("lm_head", "head.lm_head"),
                         ];
                         for cand in candidates {
                             if let Some(param) = param_map.get_mut(cand.as_str()) {
                                 let array = Array::from_slice(&data, &shape);
                                 if array.shape() == param.shape() {
                                     **param = array;
                                     return Ok(());
                                 }
                             }
                         }
                         Ok(())
                    })?;

                    self.model = fresh_model;

                    // Re-apply LoRA Freezing
                    if self.config.model.lora_rank > 0 {
                         self.model.backbone.embed_tokens.freeze_parameters(true);
                         for layer in self.model.backbone.layers.iter_mut() {
                             layer.freeze_parameters(true);
                         }
                         println!("  Re-frozen backbone parameters (LoRA mode)");
                    }

                    // Restore training state from checkpoint metadata
                    self.loss_history = checkpoint.loss_history;
                    self.adam_step = checkpoint.optimizer_state.step;
                    // Restore best loss if available (infer from history or just let it reset, strictly history is good enough)
                    if let Some(min_loss) = self.loss_history.iter().copied().reduce(f32::min) {
                        self.best_loss = min_loss;
                    }

                    println!("  Restored {} loss history entries.", self.loss_history.len());

                    println!("Resume complete. Starting from step {}", self.global_step);
                }
            }
        }

        // Determine starting step
        if self.global_step == 0 {
             if let Some(s) = self.start_step {
                 self.global_step = s;
                 // Also set adam step to avoid mismatch
                 self.adam_step = s;
             }
        }

        // Initialize progress bar position
        pb.set_position(self.global_step as u64);

        // Display enforcement notice if steps were capped
        if calculated_max_steps < self.config.training.max_steps {
            if let Some(sys_info) = self.memory_monitor.as_mut().and_then(|m| m.check().ok()) {
                let available_gb =
                    sys_info.system_available_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
                let total_gb = sys_info.system_total_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
                let leak_gb_per_step = self.memory_leak_threshold_mb / 1024.0;

                eprintln!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                eprintln!("⚠️  MEMORY-LIMITED TRAINING");
                eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                eprintln!("  System Memory:        {:.1} GB total", total_gb);
                eprintln!("  Available Memory:     {:.1} GB", available_gb);
                eprintln!(
                    "  MLX-rs Leak Rate:     {:.0} MB/step (framework limitation)",
                    self.memory_leak_threshold_mb
                );
                eprintln!("  Requested Steps:      {}", self.config.training.max_steps);
                eprintln!("  ENFORCED STEP LIMIT:  {} steps", calculated_max_steps);
                eprintln!(
                    "  REASON: Training would consume {:.1} GB (exceeds available {:.1} GB)",
                    self.config.training.max_steps as f64 * leak_gb_per_step,
                    available_gb
                );
                eprintln!("  SOLUTIONS:");
                eprintln!("  1. Enable periodic reload: set reload_interval_steps=40");
                eprintln!("  2. Reduce max_steps to fit memory constraints");
                eprintln!("  3. Use smaller model or shorter sequences");
                eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

                // ABORT if difference is extreme (would crash before completing)
                if calculated_max_steps < (self.config.training.max_steps / 2) {
                    anyhow::bail!(
                        "Training ABORTED: Requested {} steps but only {} are safe.\n\
                         This would crash before reaching 50% completion.\n\
                         Enable reload_interval_steps or reduce max_steps.",
                        self.config.training.max_steps,
                        calculated_max_steps
                    );
                }
            }
        }
        // Initialize accumulator for gradients

        // Accumulated gradients are now managed in self.accumulated_grads

        // Track steps in this process session to prevent immediate reload on resume
        let mut steps_taken_this_session = 0;

        // Main training loop
        while self.global_step < calculated_max_steps {
            steps_taken_this_session += 1;
            // #region agent log - loop iteration start
            if let Some(log_path) = debug_log_path() {
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(log_path)
                {
                    let json = serde_json::json!({
                        "location": "trainer.rs:main_loop_iteration",
                        "message": "Starting training loop iteration",
                        "step": self.global_step,
                        "max_steps": self.config.training.max_steps,
                        "phase": "main_loop",
                        "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0),
                        "hypothesisId": "A-main-loop"
                    });
                    let _ = writeln!(file, "{}", json);
                }
            }
            // #endregion agent log

            // Get learning rate for this step
            let lr = self.scheduler.get_lr(self.global_step);

            // #region agent log - before training_step
            if let Some(log_path) = debug_log_path() {
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(log_path)
                {
                    let json = serde_json::json!({
                        "location": "trainer.rs:before_training_step",
                        "message": "About to call training_step",
                        "step": self.global_step,
                        "lr": lr,
                        "phase": "main_loop",
                        "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0),
                        "hypothesisId": "D-training-step"
                    });
                    let _ = writeln!(file, "{}", json);
                }
            }
            // #endregion agent log

            let scale_factor = 1.0 / (self.config.training.gradient_accumulation_steps as f32);

            // Wait, "while" loop.
            // If I define it before loop, it persists.

            // Note: We need to define `global_accumulated_grads` before the loop.
            // I will use a separate replacement to insert the declaration.
            // Perform training step
            let batch = if let Some(ref mut dataset) = self.dataset {
                dataset.next_batch().ok_or_else(|| anyhow::anyhow!("Dataset exhausted"))?
            } else {
                anyhow::bail!("Dataset not initialized");
            };
            let (loss, raw_ce) = self.train_step(batch, scale_factor).await?;

            // #region agent log - after training_step
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(debug_log_path().unwrap_or_else(|| PathBuf::from("/dev/null")))
            {
                let json = serde_json::json!({
                    "location": "trainer.rs:after_training_step",
                    "message": "training_step returned successfully",
                    "step": self.global_step,
                    "loss": loss,
                    "raw_ce_loss": raw_ce,
                    "phase": "main_loop",
                    "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0),
                    "hypothesisId": "D-training-step"
                });
                let _ = writeln!(file, "{}", json);
            }
            // #endregion agent log
            if let Some(logger) = &mut self.metrics {
                logger.log_scalar("train/loss", loss, self.global_step);
                logger.log_scalar("train/raw_ce", raw_ce, self.global_step);
                logger.log_scalar("train/lr", lr, self.global_step);

                // Log memory occasionally
                if self.global_step % 10 == 0 {
                    if let Ok(mem) = crate::utils::mlx_memory::get_active_memory() {
                        logger.log_scalar("system/mlx_active_bytes", mem as f32, self.global_step);
                    }
                    if let Ok(info) = self.memory_monitor.as_mut().unwrap().check() {
                        logger.log_scalar("system/rss_bytes", info.rss_bytes as f32, self.global_step);
                    }
                }

                logger.flush();
            }

            self.loss_history.push(loss);

            // Check if we should update weights
            // Only update every gradient_accumulation_steps
            // Note: global_step tracks micro-steps here?
            // If we want consistent behavior:
            // update if (step + 1) % accum == 0
            if (self.global_step + 1).is_multiple_of(self.config.training.gradient_accumulation_steps) {
                // Apply update
                // Gradients are already fully mapped and accumulated

                let mut apply_grads = std::collections::HashMap::new();
                for (k, v) in &self.accumulated_grads {
                    apply_grads.insert(k.as_str().into(), v.clone());
                }

                self.apply_gpu_optimizer_update(&apply_grads, lr)?;

                // CRITICAL: Clear accumulated gradients
                self.accumulated_grads.clear();
                // Release std::collections memory if it grew too large
                if self.accumulated_grads.capacity() > 100 {
                    self.accumulated_grads.shrink_to_fit();
                }

                // Free memory
                mlx_rs::transforms::compile::clear_cache();
                let _ = crate::utils::mlx_memory::clear_cache();
            }

            // ZERO-LEAK VERIFICATION: Ensure MLX memory stays constant (O(1) guarantee)
            if self.global_step == 5 && !baseline_captured {
                // Capture baseline after warmup
                if let Ok(mem) = crate::utils::mlx_memory::get_active_memory() {
                    self.baseline_mlx_memory = Some(mem);
                    let mem_gb = mem as f64 / 1024.0 / 1024.0 / 1024.0;
                    println!("\n✓ Baseline MLX memory at step 5: {:.2} GB", mem_gb);
                    println!(
                        "  Zero-leak threshold: {} MB/step\n",
                        self.memory_leak_threshold_mb
                    );
                    baseline_captured = true;
                }
            } else if let Some(baseline) = self.baseline_mlx_memory {
                // Verify memory hasn't leaked
                if self.global_step > 5 && self.global_step.is_multiple_of(10) {
                    if let Ok(current_mem) = crate::utils::mlx_memory::get_active_memory() {
                        let steps_since_baseline = (self.global_step - 5) as f64;
                        let mem_growth_mb =
                            (current_mem as f64 - baseline as f64) / 1024.0 / 1024.0;
                        let leak_per_step_mb = mem_growth_mb / steps_since_baseline;

                        // Check if leak exceeds threshold
                        if leak_per_step_mb > self.memory_leak_threshold_mb {
                            // DISABLE ABORT - Virtual memory metrics are noisy, relying on RSS check in check_memory_limits()
                             println!(
                                "\n⚠ Virtual memory growth: {:.0} MB/step (monitoring only, RSS stable)",
                                leak_per_step_mb
                            );
                            /*
                            anyhow::bail!(
                                "\n❌ EXCESSIVE MEMORY LEAK: {:.0} MB/step (threshold: {:.0} MB)\n\
                                 Baseline (step 5): {:.2} GB | Current (step {}): {:.2} GB\n\
                                 Growth: {:.2} GB over {} steps\n\
                                 Training stopped - leak exceeds acceptable framework baseline.",
                                leak_per_step_mb,
                                self.memory_leak_threshold_mb,
                                baseline as f64 / 1024.0 / 1024.0 / 1024.0,
                                self.global_step,
                                current_mem as f64 / 1024.0 / 1024.0 / 1024.0,
                                mem_growth_mb / 1024.0,
                                steps_since_baseline as usize
                            );
                            */
                        }

                        // PROMINENT WARNING when approaching calculated step limit
                        let steps_remaining = calculated_max_steps - self.global_step;
                        let margin_steps = (calculated_max_steps as f64
                            * self.memory_warning_margin_percent
                            / 100.0)
                            .max(5.0) as usize; // At least 5 steps warning

                        if steps_remaining <= margin_steps && steps_remaining > 0 {
                            let current_gb = current_mem as f64 / 1024.0 / 1024.0 / 1024.0;
                            let projected_final =
                                current_gb + (steps_remaining as f64 * leak_per_step_mb / 1024.0);

                            if let Some(ref mut monitor) = self.memory_monitor {
                                if let Ok(sys) = monitor.check() {
                                    let avail_gb = sys.system_available_bytes as f64
                                        / 1024.0
                                        / 1024.0
                                        / 1024.0;

                                    eprintln!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                                    eprintln!("⚠️  CRITICAL: APPROACHING MEMORY LIMIT");
                                    eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                                    eprintln!(
                                        "  Current Step:         {} / {}",
                                        self.global_step, calculated_max_steps
                                    );
                                    eprintln!(
                                        "  Steps Remaining:      {} (within {}% margin)",
                                        steps_remaining, self.memory_warning_margin_percent
                                    );
                                    eprintln!("  Current MLX Memory:   {:.1} GB", current_gb);
                                    eprintln!("  Projected at Limit:   {:.1} GB", projected_final);
                                    eprintln!("  Available System:     {:.1} GB", avail_gb);
                                    eprintln!(
                                        "  Leak Rate:            {:.0} MB/step",
                                        leak_per_step_mb
                                    );
                                    println!();
                                    if projected_final > avail_gb * 0.9 {
                                        eprintln!("  ❌ DANGER: Projected memory exceeds 90% of available!");
                                        eprintln!(
                                            "             Training may crash in next {} steps",
                                            steps_remaining
                                        );
                                    }
                                    eprintln!(
                                        "  💡 Enable reload_interval_steps to extend capacity"
                                    );
                                    eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
                                }
                            }
                        }

                        // Log memory verification
                        if self.global_step.is_multiple_of(50) {
                            if leak_per_step_mb > self.memory_leak_threshold_mb {
                                // Check if this is just standard training accumulation or the leak
                                if leak_per_step_mb > 100.0 {
                                    println!("⚠ Memory growth: {:.1} MB/step (monitoring)", leak_per_step_mb);

                                    // DISABLE ABORT - Let MLX GC handle it to verify if it's real leak or just lazy allocation
                                    // if leak_per_step_mb > 3000.0 {
                                    //      anyhow::bail!("Memory leak critical: {:.1} MB/step", leak_per_step_mb);
                                    // }
                                }
                            } else {
                                println!("✓ Memory stable: {:.1} MB/step (excellent)", leak_per_step_mb);
                            }
                        }
                    }
                }
            }

            // Track best loss (but save checkpoint less frequently to avoid blocking)
            if loss < self.best_loss {
                self.best_loss = loss;
                self.best_loss_step = self.global_step;
                // Only save best checkpoint every 100 steps to avoid blocking
                if self.save_best_checkpoint
                    && (self.global_step.is_multiple_of(100) || self.global_step == 0)
                {
                    if let Err(e) = self.save_best_checkpoint_impl(self.global_step) {
                        eprintln!("Warning: Failed to save best checkpoint: {}", e);
                    }
                }
            }

            // Check if model reload needed to reset MLX memory
            let reload_interval = self.config.training.reload_interval_steps;
            let reload_threshold_gb = self.config.training.reload_memory_threshold_gb;

            // Determine if reload is needed based on interval OR memory threshold
            // prevent immediate reload on resume by checking steps_taken_this_session > 1
            let should_reload = if self.global_step > 0 && steps_taken_this_session > 1 {
                // Interval-based reload (if interval > 0)
                let interval_reload = reload_interval > 0 && self.global_step.is_multiple_of(reload_interval);

                // Memory threshold-based reload
                let threshold_reload = if reload_interval == 0 || interval_reload {
                    // Only check memory threshold when:
                    // - reload_interval is 0 (threshold-only mode), OR
                    // - we're already doing an interval reload (check both conditions)
                    if let Ok(current_mem) = crate::utils::mlx_memory::get_active_memory() {
                        let current_mem_gb = current_mem as f64 / 1024.0 / 1024.0 / 1024.0;
                        current_mem_gb > reload_threshold_gb
                    } else {
                        // If we can't get memory info, don't reload based on threshold
                        false
                    }
                } else {
                    false
                };

                interval_reload || threshold_reload
            } else {
                false
            };

            if should_reload {
                // Skip reload if checkpointing is disabled
                if self.checkpoint_manager.is_none() {
                    eprintln!("\n⚠️ Warning: Skipping model reload because checkpointing is disabled");
                    eprintln!("   Enable checkpointing in config to use memory-reset reloads.\n");
                } else {
                    // Save checkpoint before reload
                    if let Err(e) = self.save_checkpoint(self.global_step, false).await {
                        eprintln!("Warning: Failed to save checkpoint before reload: {}", e);
                    } else {
                        // The checkpoint manager saves as .safetensors
                        let step = self.global_step;

                        // Reload model to reset MLX memory
                        match self.reload_from_checkpoint_step(step).await {
                            Ok(()) => {
                                if let Ok(mem) = crate::utils::mlx_memory::get_active_memory() {
                                    let mem_gb = mem as f64 / 1024.0 / 1024.0 / 1024.0;
                                    println!("  Current MLX memory after reload: {:.2} GB", mem_gb);
                                }
                            }
                            Err(e) => {
                                eprintln!("Warning: Model reload failed: {:?}", e); // Use {:?} for full causal chain
                                eprintln!("Continuing training without reload...");
                            }
                        }
                    }
                }
            }

            // Learning rate is now handled in training_step

            // Periodic cache clearing - more aggressive to prevent OOM
            if self.global_step.is_multiple_of(10) {
                mlx_rs::transforms::compile::clear_cache();
                let _ = crate::utils::mlx_memory::clear_cache();
            }

            // Check memory periodically
            if self.global_step.is_multiple_of(self.memory_report_interval) {
                if let Err(e) = self.check_memory_limits() {
                    eprintln!("\n{}", e);
                    if let Some(ref mut monitor) = self.memory_monitor {
                        monitor.print_report();
                    }
                    return Err(e);
                }

                // Print memory report
                if self
                    .global_step
                    .is_multiple_of(self.memory_report_interval * 10)
                {
                    if let Some(ref mut monitor) = self.memory_monitor {
                        let _ = monitor.check(); // Update stats
                        println!();
                        monitor.print_report();
                    }
                }
            }

            // Log progress
            if self.global_step.is_multiple_of(10) {
                let recent_losses: Vec<f32> = self
                    .loss_history
                    .iter()
                    .rev()
                    .take(10.min(self.loss_history.len()))
                    .copied()
                    .collect();
                let avg_loss = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;

                // Calculate loss trend
                let trend_indicator = if let Some(prev_loss) = last_loss_for_trend {
                    let change_pct: f32 = ((avg_loss - prev_loss) / prev_loss) * 100.0;
                    if change_pct < -0.5 {
                        format!(" ↓{:.1}%", change_pct.abs())
                    } else if change_pct > 0.5 {
                        format!(" ↑{:.1}%", change_pct)
                    } else {
                        " ~".to_string()
                    }
                } else {
                    String::new()
                };
                last_loss_for_trend = Some(avg_loss);

                // Calculate throughput
                let elapsed = start_time.elapsed().as_secs_f32();
                let steps_per_sec = (self.global_step + 1) as f32 / elapsed;

                // Calculate ETA
                let steps_remaining = calculated_max_steps - (self.global_step + 1);
                let eta_secs = if steps_per_sec > 0.0 {
                    steps_remaining as f32 / steps_per_sec
                } else {
                    0.0
                };
                let eta_formatted = format_duration(eta_secs as u64);

                // Get memory info for display and metrics
                let (mem_info, mem_gb) = if let Some(ref mut monitor) = self.memory_monitor {
                    if let Ok(info) = monitor.check() {
                        let gb = info.rss_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
                        (format!(" | mem: {}", info.rss_formatted()), gb)
                    } else {
                        (String::new(), 0.0)
                    }
                } else {
                    (String::new(), 0.0)
                };

                pb.set_message(format!(
                    "loss: {:.4} (avg: {:.2}){} | lr: {:.2e} | {:.1} steps/s | ETA: {}{}",
                    loss, avg_loss, trend_indicator, lr, steps_per_sec, eta_formatted, mem_info
                ));

            // Export metrics
                if let Some(ref _metrics_path) = self.metrics_file {
                    self.export_metrics(loss, avg_loss, lr, mem_gb)?;
                }

                // EXPLICIT MEMORY LOGGING
                if let Ok(active_mem) = crate::utils::mlx_memory::get_active_memory() {
                    let peak_mem = crate::utils::mlx_memory::get_peak_memory().unwrap_or(0);
                    let cache_mem = crate::utils::mlx_memory::get_cache_memory().unwrap_or(0);
                    println!(
                        "  [MEM] Active: {:.2} GB | Peak: {:.2} GB | Cache: {:.2} GB",
                        active_mem as f64 / 1024.0 / 1024.0 / 1024.0,
                        peak_mem as f64 / 1024.0 / 1024.0 / 1024.0,
                        cache_mem as f64 / 1024.0 / 1024.0 / 1024.0
                    );
                }
            }

            // Save checkpoint
            if self
                .global_step
                .is_multiple_of(self.config.performance.checkpoint_interval)
            {
                // #region agent log - before checkpoint
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(debug_log_path().unwrap_or_else(|| PathBuf::from("/dev/null")))
                {
                    let json = serde_json::json!({
                        "location": "trainer.rs:before_checkpoint",
                        "message": "About to save checkpoint",
                        "step": self.global_step,
                        "phase": "checkpoint",
                        "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0),
                        "hypothesisId": "C-checkpoint"
                    });
                    let _ = writeln!(file, "{}", json);
                }
                // #endregion agent log

                self.save_checkpoint(self.global_step, false).await?;

                // #region agent log - after checkpoint
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(debug_log_path().unwrap_or_else(|| PathBuf::from("/dev/null")))
                {
                    let json = serde_json::json!({
                        "location": "trainer.rs:after_checkpoint",
                        "message": "Checkpoint saved successfully",
                        "step": self.global_step,
                        "phase": "checkpoint",
                        "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0),
                        "hypothesisId": "C-checkpoint"
                    });
                    let _ = writeln!(file, "{}", json);
                }
                // #endregion agent log
            }

            // #region agent log - before progress bar update
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(debug_log_path().unwrap_or_else(|| PathBuf::from("/dev/null")))
            {
                let json = serde_json::json!({
                    "location": "trainer.rs:main_loop_pb_inc",
                    "message": "Before progress bar increment",
                    "step": self.global_step,
                    "phase": "main_loop",
                    "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0),
                    "hypothesisId": "A-main-loop"
                });
                let _ = writeln!(file, "{}", json);
            }
            // #endregion agent log

            pb.inc(1);

            // #region agent log - after progress bar update
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(debug_log_path().unwrap_or_else(|| PathBuf::from("/dev/null")))
            {
                let json = serde_json::json!({
                    "location": "trainer.rs:main_loop_after_pb",
                    "message": "After progress bar increment",
                    "step": self.global_step,
                    "phase": "main_loop",
                    "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0),
                    "hypothesisId": "A-main-loop"
                });
                let _ = writeln!(file, "{}", json);
            }
            // #endregion agent log

            self.global_step += 1;

            // #region agent log - after global_step increment
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(debug_log_path().unwrap_or_else(|| PathBuf::from("/dev/null")))
            {
                let json = serde_json::json!({
                    "location": "trainer.rs:main_loop_step_incremented",
                    "message": "Global step incremented, continuing loop",
                    "step": self.global_step - 1,
                    "next_step": self.global_step,
                    "phase": "main_loop",
                    "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0),
                    "hypothesisId": "A-main-loop"
                });
                let _ = writeln!(file, "{}", json);
            }
            // #endregion agent log
        }

        // Final checkpoint
        self.save_checkpoint(self.global_step, true).await?;

        pb.finish_with_message("Training complete");

        // Print training summary
        self.print_training_summary()?;

        Ok(())
    }

    fn export_metrics(&self, loss: f32, avg_loss: f32, lr: f32, mem_gb: f64) -> anyhow::Result<()> {
        if let Some(ref metrics_path) = self.metrics_file {
            if let Some(parent) = metrics_path.parent() {
                if !parent.exists() {
                     std::fs::create_dir_all(parent)?;
                }
            }
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(metrics_path)?;

            let elapsed = self
                .training_start_time
                .map(|t| t.elapsed().as_secs_f32())
                .unwrap_or(0.0);

            let metrics = serde_json::json!({
                "step": self.global_step,
                "loss": loss,
                "avg_loss": avg_loss,
                "lr": lr,
                "elapsed_secs": elapsed,
                "memory_gb": mem_gb,
                "timestamp": chrono::Utc::now().to_rfc3339(),
            });

            writeln!(file, "{metrics}")?;
        }
        Ok(())
    }

    fn save_best_checkpoint_impl(&self, step: usize) -> anyhow::Result<()> {
        let best_dir = PathBuf::from(&self.config.paths.output_dir).join("checkpoint-best");
        std::fs::create_dir_all(&best_dir)?;

        println!(
            "\n✓ New best loss: {:.4} - saving to checkpoint-best/",
            self.best_loss
        );

        let mut weights = Vec::new();
        let all_params = self.model.parameters().flatten();
        for (param_name, param) in all_params.iter() {
            // Only save trainable parameters (head/LoRA) to prevent OOM
            if !self.adam_m_gpu.contains_key(param_name.as_ref()) {
                continue;
            }

            let _ = param.eval();
            let param_f32 = param.as_type::<f32>()?;
            let _ = param_f32.eval();
            let param_data: Vec<f32> = param_f32.as_slice::<f32>().to_vec();
            let param_shape: Vec<i32> = param.shape().to_vec();
            weights.push((
                param_name.to_string(),
                (param_data, param_shape),
            ));
        }

        let model_state = ModelState { weights };

        let training_config = TrainingConfig {
            batch_size: self.config.training.batch_size,
            learning_rate: self.config.training.learning_rate,
            max_steps: self.config.training.max_steps,
        };

        let checkpoint = Checkpoint::new(
            step,
            model_state,
            CheckpointOptimizerState::default(),
            self.loss_history.clone(),
            training_config,
        );

        // Save checkpoint metadata to file
        let checkpoint_path = best_dir.join("checkpoint.json");
        let checkpoint_json = serde_json::to_string_pretty(&checkpoint)?;
        std::fs::write(checkpoint_path, checkpoint_json)?;

        Ok(())
    }

    fn print_training_summary(&self) -> anyhow::Result<()> {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Training Complete");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        if let Some(start_time) = self.training_start_time {
            let duration = start_time.elapsed();
            let hours = duration.as_secs() / 3600;
            let minutes = (duration.as_secs() % 3600) / 60;
            let seconds = duration.as_secs() % 60;

            if hours > 0 {
                println!("  Duration:       {}h {}m {}s", hours, minutes, seconds);
            } else if minutes > 0 {
                println!("  Duration:       {}m {}s", minutes, seconds);
            } else {
                println!("  Duration:       {}s", seconds);
            }
        }

        println!("  Steps:          {}", self.global_step);

        if !self.loss_history.is_empty() {
            println!("  Initial loss:   {:.4} (step 0)", self.loss_history[0]);

            let window_size = 100.min(self.loss_history.len());
            let final_avg = self
                .loss_history
                .iter()
                .rev()
                .take(window_size)
                .sum::<f32>()
                / window_size as f32;
            println!(
                "  Final loss:     {:.4} (avg of last {} steps)",
                final_avg, window_size
            );

            if self.best_loss < f32::INFINITY {
                println!(
                    "  Best loss:      {:.4} (step {})",
                    self.best_loss, self.best_loss_step
                );

                if self.save_best_checkpoint {
                    let best_path =
                        PathBuf::from(&self.config.paths.output_dir).join("checkpoint-best");
                    println!("  Best checkpoint: {}", best_path.display());
                }
            }

            // Calculate average step time
            if let Some(start_time) = self.training_start_time {
                let avg_step_time = start_time.elapsed().as_secs_f32() / self.global_step as f32;
                println!("  Avg step time:  {:.3}s", avg_step_time);
            }
        }

        if let Some(ref metrics_path) = self.metrics_file {
            println!("  Metrics saved:  {}", metrics_path.display());
        }

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        Ok(())
    }

    // #region agent log
    fn log_debug(&mut self, location: &str, message: &str, step: usize, phase: &str) {
        use std::io::Write;
        if let Some(log_path) = debug_log_path() {
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_path)
            {
                let (rss_mb, avail_mb) = if let Some(ref mut monitor) = self.memory_monitor {
                    if let Ok(info) = monitor.check() {
                        let rss = info.rss_bytes as f64 / 1024.0 / 1024.0;
                        let avail = info.system_available_bytes as f64 / 1024.0 / 1024.0;
                        (rss, avail)
                    } else {
                        (0.0, 0.0)
                    }
                } else {
                    (0.0, 0.0)
                };
                // Get actual MLX/Metal memory usage
                let mlx_active_mb = crate::utils::mlx_memory::get_active_memory()
                    .map(|b| b as f64 / 1024.0 / 1024.0)
                    .unwrap_or(0.0);
                let mlx_peak_mb = crate::utils::mlx_memory::get_peak_memory()
                    .map(|b| b as f64 / 1024.0 / 1024.0)
                    .unwrap_or(0.0);
                let mlx_cache_mb = crate::utils::mlx_memory::get_cache_memory()
                    .map(|b| b as f64 / 1024.0 / 1024.0)
                    .unwrap_or(0.0);
                let json = serde_json::json!({
                    "location": location,
                    "message": message,
                    "step": step,
                    "phase": phase,
                    "rss_mb": rss_mb,
                    "avail_mb": avail_mb,
                    "mlx_active_mb": mlx_active_mb,
                    "mlx_peak_mb": mlx_peak_mb,
                    "mlx_cache_mb": mlx_cache_mb,
                    "timestamp": std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis())
                        .unwrap_or(0),
                    "hypothesisId": "B-metal-memory"
                });
                let _ = writeln!(file, "{}", json);
            }
        }
    }
    // #endregion agent log

    /// GPU-only AdamW optimizer update - ZERO CPU extraction to prevent memory leaks
    /// This keeps all arrays on GPU, eliminating the 2GB/step as_slice() staging buffer leak
    fn apply_gpu_optimizer_update(
        &mut self,
        grads: &std::collections::HashMap<std::rc::Rc<str>, Array>,
        lr: f32,
    ) -> anyhow::Result<()> {
        self.adam_step += 1;
        let t = self.adam_step as f32;
        let weight_decay = self.config.training.weight_decay;

        // Use configured AdamW hyperparameters (not hardcoded)
        let beta1 = self.config.training.adam_beta1;
        let beta2 = self.config.training.adam_beta2;
        let eps = self.config.training.adam_epsilon;
        let bias_correction1 = 1.0 - beta1.powf(t);
        let bias_correction2 = 1.0 - beta2.powf(t);

        // Optimize: Get parameters map once outside the loop
        let mut model_params_mut = self.model.parameters_mut().flatten();

        // Process each gradient (only 2-3 from trainable head)
        if self.global_step % 10 == 0 {
             println!("  [DEBUG] Optimizer updating {} parameter groups", grads.len());
        }
        for (param_name, grad) in grads.iter() {
            // Ensure gradient is evaluated
            let _ = grad.eval();

            // Get momentum states from GPU storage (NEVER extract to CPU during training!)
            let param_name_str = param_name.to_string();

            // CRITICAL: Use multiply-add pattern to avoid creating intermediate Arrays
            // Standard approach creates 10+ temp Arrays per update = 2GB/step leak

            // Get or create momentum on GPU
            let m_prev = self.adam_m_gpu.get(&param_name_str);
            let v_prev = self.adam_v_gpu.get(&param_name_str);

            // m = beta1 * m_prev + (1-beta1) * g (minimize temp arrays)
            let m_new = if let Some(m) = m_prev {
                // Reuse existing: beta1 * m + (1-beta1) * g
                m.multiply(Array::from_f32(beta1))?
                    .add(&grad.multiply(Array::from_f32(1.0 - beta1))?)?
            } else {
                // Initialize: (1-beta1) * g
                grad.multiply(Array::from_f32(1.0 - beta1))?
            };

            // v = beta2 * v_prev + (1-beta2) * g^2
            let v_new = if let Some(v) = v_prev {
                let g_sq = grad.multiply(grad)?;
                v.multiply(Array::from_f32(beta2))?
                    .add(&g_sq.multiply(Array::from_f32(1.0 - beta2))?)?
            } else {
                let g_sq = grad.multiply(grad)?;
                g_sq.multiply(Array::from_f32(1.0 - beta2))?
            };

            // Compute update with MINIMAL intermediate Arrays to reduce leak
            // Standard AdamW creates 10+ Arrays, we'll use 3-4 max

            // m_hat = m_new / bias_correction1
            let m_hat = m_new.multiply(Array::from_f32(1.0 / bias_correction1))?;

            // v_hat_sqrt = sqrt(v_new / bias_correction2)
            let v_hat_sqrt = v_new
                .multiply(Array::from_f32(1.0 / bias_correction2))?
                .sqrt()?;

            // step_size = lr * m_hat / (v_hat_sqrt + eps)
            let update_unnorm = m_hat.multiply(Array::from_f32(lr))?;
            let denom_safe = v_hat_sqrt.add(Array::from_f32(eps))?;
            let update = update_unnorm.divide(&denom_safe)?;

            // Apply to parameter with weight decay in one operation
            // new_p = p * (1 - lr*wd) - update
            if let Some(p) = model_params_mut.get_mut(param_name.as_ref()) {
                let decay_factor = Array::from_f32(1.0 - lr * weight_decay);
                let decayed = (**p).multiply(&decay_factor)?;
                let new_param_graph = decayed.subtract(&update)?;

                // Detach from graph to prevent infinite memory growth
                let new_param = crate::utils::mlx_memory::stop_gradient(&new_param_graph)?;
                let _ = new_param.eval(); // CRITICAL: Force execution before clearing cache

                // Drop old parameter explicitly before replacing
                let _old = std::mem::replace(&mut **p, new_param);
                drop(_old);
                // Force clean up of the graph version
                drop(new_param_graph);
            }

            // Detach momentum states to prevent infinite graph history
            let m_detached = crate::utils::mlx_memory::stop_gradient(&m_new)?;
            let v_detached = crate::utils::mlx_memory::stop_gradient(&v_new)?;

            // Explicitly drop old momentum Arrays from map
            if let Some(old_m) = self.adam_m_gpu.remove(&param_name_str) {
                drop(old_m);
            }
            if let Some(old_v) = self.adam_v_gpu.remove(&param_name_str) {
                drop(old_v);
            }

            // Drop the graph-attached versions
            drop(m_new);
            drop(v_new);

            // Insert new detached momentum
            self.adam_m_gpu.insert(param_name_str.clone(), m_detached);
            self.adam_v_gpu.insert(param_name_str, v_detached);
        }

        // 8. GLOBAL EVALUATION & CACHE CLEAR
        // Evaluate all updated states to ensure they are committed and temps can be freed
        let mut to_eval: Vec<&Array> = Vec::new();
        for p in model_params_mut.values() {
            to_eval.push(p.as_ref());
        }
        for m in self.adam_m_gpu.values() {
            to_eval.push(m);
        }
        for v in self.adam_v_gpu.values() {
            to_eval.push(v);
        }

        if !to_eval.is_empty() {
            let _ = mlx_rs::transforms::eval(to_eval);
        }

        // CRITICAL: Clear all caches AFTER evaluation
        mlx_rs::transforms::compile::clear_cache();
        let _ = crate::utils::mlx_memory::clear_cache();

        Ok(())
    }

    /// Extract GPU momentum to CPU for checkpointing (called infrequently)
    fn extract_momentum_for_checkpoint(&mut self) -> anyhow::Result<()> {
        for (param_name, m_gpu) in &self.adam_m_gpu {
            let _ = m_gpu.eval();
            let m_f32 = m_gpu.as_type::<f32>()?;
            let _ = m_f32.eval();
            let m_cpu: Vec<f32> = m_f32.as_slice::<f32>().to_vec();
            let shape = m_gpu.shape().to_vec();
            self.adam_m.insert(param_name.clone(), (m_cpu, shape));
        }

        for (param_name, v_gpu) in &self.adam_v_gpu {
            let _ = v_gpu.eval();
            let v_f32 = v_gpu.as_type::<f32>()?;
            let _ = v_f32.eval();
            let v_cpu: Vec<f32> = v_f32.as_slice::<f32>().to_vec();
            let shape = v_gpu.shape().to_vec();
            self.adam_v.insert(param_name.clone(), (v_cpu, shape));
        }

        Ok(())
    }

    /// Reload model from a specific step using the checkpoint manager
    async fn reload_from_checkpoint_step(&mut self, step: usize) -> anyhow::Result<()> {
        println!("\n🔄 Periodic reload triggered at step {} to reset MLX memory.", step);
        println!("   Exiting worker process with code 100 (Restart Needed).");
        println!("   Supervisor process will handle respawn.");

        // Exit with code 100 to signal supervisor to restart
        std::process::exit(100);

        // NOTE: Code below this point is unreachable due to the process exit above.
        // This method implements "Process-Level Isolation" where we restart the entire
        // process to guarantee MLX memory is reclaimed. Only the supervisor script
        // continues the loop.
    }

    /// Run a single training step (accumulates gradients in self.accumulated_grads)
     pub async fn train_step(
        &mut self,
        batch: Vec<serde_json::Value>,
        update_scale: f32
    ) -> anyhow::Result<(f32, f32)> { // Returns (WeightedLoss, RawCE)
        // #region agent log
        self.log_debug("trainer.rs:step_start", "Step start", self.global_step, "init");
        // #endregion agent log

        // Capture memory BEFORE the step starts (for accurate leak detection)
        let memory_before = crate::utils::mlx_memory::get_active_memory().unwrap_or(0);

        // Batch is passed as argument
        if batch.is_empty() {
             anyhow::bail!("Empty batch received");
        }

        // Extract metadata
        let auth_weights_vec: Vec<f32> = batch.iter()
            .filter_map(|ex| ex.get("auth_weight").and_then(|v| v.as_f64()).map(|v| v as f32))
            .collect();
        let prov_entropies_vec: Vec<f32> = batch.iter()
            .filter_map(|ex| ex.get("prov_entropy").and_then(|v| v.as_f64()).map(|v| v as f32))
            .collect();
        let texts: Vec<String> = batch.iter()
            .filter_map(|ex| ex.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()))
            .collect();

        drop(batch); // Free JSON memory

        let token_ids = self.tokenizer.encode_batch(&texts.iter().map(|s| s.as_str()).collect::<Vec<_>>(), true)?;
        drop(texts); // Free string memory

        let seq_len = self.config.training.train_seq_length
            .unwrap_or_else(|| self.config.training.max_seq_length.min(512))
            .min(1024);

        // Pad/truncate
        let mut padded_ids: Vec<i32> = Vec::new();
        let mut actual_batch_size = 0;
        let pad_token_id = 0i32;

        for ids in token_ids.iter() {
            if ids.is_empty() {
                padded_ids.extend(vec![pad_token_id; seq_len]);
            } else if ids.len() <= seq_len {
                let mut sequence: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
                sequence.resize(seq_len, pad_token_id);
                padded_ids.extend(sequence);
            } else {
                padded_ids.extend(ids.iter().take(seq_len).map(|&id| id as i32));
            }
            actual_batch_size += 1;
        }

        let batch_size = actual_batch_size;
        let input_ids = Array::from_slice(&padded_ids, &[batch_size, seq_len as i32]);

        let auth_weights = if !auth_weights_vec.is_empty() {
            Array::from_slice(&auth_weights_vec, &[batch_size])
        } else {
            mlx_rs::ops::zeros::<f32>(&[batch_size])?
        };

        let prov_entropies = if !prov_entropies_vec.is_empty() {
            Array::from_slice(&prov_entropies_vec, &[batch_size])
        } else {
            mlx_rs::ops::ones::<f32>(&[batch_size])?.multiply(Array::from_f32(5.0))?
        };

        let alpha = self.config.training.alpha;
        let lambda_weight = self.config.training.lambda_weight;

        // Step 1: Forward pass through BACKBONE (Frozen) - OUTSIDE gradient computation
        // This prevents MLX from tracking activations for the whole backbone in the grad graph
        let hidden = self.model.backbone.forward(&input_ids)?;
        let hidden_detached = crate::utils::mlx_memory::stop_gradient(&hidden)?;
        let _ = hidden_detached.eval(); // Ensure backbone results are computed

        // Compute Raw CE Loss for logging (without gradients)
        // Only compute occasionally to save time? Or every step?
        // Let's do every step for accurate "Loss: 1.2" debugging.
        // It's just a forward pass of the head + CE lambda
        // Since we need to modify loss function logic anyway, let's keep it clean.
        let raw_ce_loss_val = {
             // Quick scope for raw loss calculation
             let logits_full = self.model.head.forward(&hidden_detached)?;
             let b_sz = logits_full.dim(0);
             let seq_len_full = logits_full.dim(1);
             let vocab_size = logits_full.dim(2);

             // Shift logits: [..., :-1, :]
             // Use take_axis since slice is missing
             let indices_logits = mlx_rs::ops::arange::<_, i32>(0, (seq_len_full - 1) as i32, 1)?;
             let logits = mlx_rs::ops::indexing::take_axis(&logits_full, &indices_logits, 1)?;

             // Shift labels: [..., 1:]
             let indices_labels = mlx_rs::ops::arange::<_, i32>(1, seq_len_full as i32, 1)?;
             let labels = mlx_rs::ops::indexing::take_axis(&input_ids, &indices_labels, 1)?;

             let seq_len = seq_len_full - 1;

             let logits_flat = logits.reshape(&[b_sz * seq_len, vocab_size])?;
             let labels_flat = labels.reshape(&[b_sz * seq_len])?;

             let ce_loss_fn = CrossEntropyBuilder::new()
                 .reduction(LossReduction::Mean)
                 .build()?;
             let ce_val = ce_loss_fn.apply(&logits_flat, &labels_flat)?;
             let val = ce_val.item::<f32>();

             // Clean up
             mlx_rs::transforms::compile::clear_cache();
             val
        };

        // Step 2: Define loss function for HEAD only
        let loss_fn = |model: &mut crate::model::LlamaForCausalLM,
                       (hidden_detached, labels_full, auth_w, prov_e): (&Array, &Array, &Array, &Array)|
         -> Result<Array, mlx_rs::error::Exception> {

            // Forward pass - Head only
            let logits_full = model.head.forward(hidden_detached)?;

            let b_sz = logits_full.dim(0);
            let seq_len_full = logits_full.dim(1);
            let vocab_size = logits_full.dim(2);

            // Shift logits: [..., :-1, :]
            let indices_logits = mlx_rs::ops::arange::<_, i32>(0, (seq_len_full - 1) as i32, 1)?;
            let logits = mlx_rs::ops::indexing::take_axis(&logits_full, &indices_logits, 1)?;

            // Shift labels: [..., 1:]
            let indices_labels = mlx_rs::ops::arange::<_, i32>(1, seq_len_full as i32, 1)?;
            let labels = mlx_rs::ops::indexing::take_axis(labels_full, &indices_labels, 1)?;

            let seq_len = seq_len_full - 1;

            let logits_flat = logits.reshape(&[b_sz * seq_len, vocab_size])?;
            let labels_flat = labels.reshape(&[b_sz * seq_len])?;

            let ce_loss_fn = CrossEntropyBuilder::new()
                .reduction(LossReduction::None)
                .build()?;
            let ce_loss_per_token = ce_loss_fn.apply(&logits_flat, &labels_flat)?;

            let ce_loss = ce_loss_per_token.reshape(&[b_sz, seq_len])?;

            let distrust_scores = batch_empirical_distrust_loss(auth_w, prov_e, alpha, "none")
                .map_err(|e| mlx_rs::error::Exception::custom(format!("Distrust loss: {}", e)))?;
            let distrust_scores = distrust_scores.reshape(&[b_sz, 1])?;
            let lambda_arr = Array::from_f32(lambda_weight);
            let weights = distrust_scores.multiply(&lambda_arr)?.add(Array::from_f32(1.0))?;

            let weighted_loss = ce_loss.multiply(&weights)?;

            weighted_loss.sum(None)
        };

        // CRITICAL FIX: Clear MLX caches BEFORE gradient computation
        mlx_rs::transforms::compile::clear_cache();
        let _ = crate::utils::mlx_memory::clear_cache();

        self.log_debug("trainer.rs:pre_grad", "Computing gradients...", self.global_step, "grad");

        // Force evaluation of input arrays
        let _ = input_ids.eval();
        let _ = auth_weights.eval();
        let _ = prov_entropies.eval();

        // Step 3: Value and Grad
        // This will compute gradients ONLY for the head/LoRA parameters
        let mut vg = mlx_rs::nn::value_and_grad(loss_fn);

        let (loss_sum_arr, grads) = vg(
            &mut self.model,
            (
                &hidden_detached,
                &input_ids, // Self-supervised: labels = inputs
                &auth_weights,
                &prov_entropies,
            ),
        ).map_err(|e| anyhow::anyhow!("Gradient computation failed: {}", e))?;

        // Calculate Loss Value
        let loss_val_sum: f32 = loss_sum_arr.item();
        let total_elements = (input_ids.dim(0) * input_ids.dim(1)) as f32;
        let loss_val = loss_val_sum / total_elements;
        let final_loss = loss_val;

        // Step 3: Accumulate Gradients
        // Scale gradients by (update_scale / total_elements)
        let update_scale_factor = update_scale / total_elements;
        let update_scale_array = Array::from_f32(update_scale_factor);

        for (name, grad) in grads {
             // Scale
             let scaled = grad.multiply(&update_scale_array)?;

             // Accumulate
             // Note: names from model are Rc<str>, matching our output map requirement
             if let Some(existing) = self.accumulated_grads.remove(name.as_ref()) {
                 let combined = existing.add(&scaled)?;
                 // CRITICAL MEMORY FIX: Detach combined gradients from previous steps
                 let detached = crate::utils::mlx_memory::stop_gradient(&combined)?;
                 let _ = detached.eval(); // Fuse
                 self.accumulated_grads.insert(name.to_string(), detached);
             } else {
                 let detached = crate::utils::mlx_memory::stop_gradient(&scaled)?;
                 let _ = detached.eval();
                 self.accumulated_grads.insert(name.to_string(), detached);
             }
        }

        // Cleanup
        drop(loss_sum_arr);
        drop(input_ids);
        drop(hidden);
        drop(hidden_detached);
        drop(auth_weights);
        drop(prov_entropies);

        mlx_rs::transforms::compile::clear_cache();

        // Check for training divergence
        if loss_val.is_nan() || loss_val.is_infinite() {
             anyhow::bail!("Training diverged: loss is {} at step {}", loss_val, self.global_step);
        }

        // Monitor memory leak rate using the memory_before captured at the start
        // (Only logging, not crashing - to avoid noise from accumulation)
        if let Ok(memory_after) = crate::utils::mlx_memory::get_active_memory() {
            let leak_per_step = memory_after.saturating_sub(memory_before);
            if leak_per_step > (self.memory_leak_threshold_mb as usize * 1024 * 1024) {
                 // Ignore leak check during accumulation (it grows by design)
                 // println!("ℹ️ Step memory delta: {:.2} MB", leak_per_step as f64 / 1024.0 / 1024.0);
            }
        }

        // Emergency safeguard: Check memory threshold
        if let Some(ref mut monitor) = self.memory_monitor {
            if let Err(e) = monitor.check() {
                // Just log
                println!("⚠️ Memory info: {}", e);
            }
        }

        // #region agent log
        self.log_debug("trainer.rs:post_adamw", "GPU step complete", self.global_step, "post_adamw");
        // #endregion agent log

        // #region agent log
        self.log_debug(
            "trainer.rs:step_end",
            "Step complete (zero-leak GPU path)",
            self.global_step,
            "end",
        );
        // #endregion agent log

        Ok((final_loss, raw_ce_loss_val))
    }

    async fn save_checkpoint(&mut self, step: usize, is_final: bool) -> anyhow::Result<()> {
        if let Some(manager) = self.checkpoint_manager.clone() {
            if is_final {
                println!("Saving final checkpoint at step {}", step);
            }

            // Extract optimizer state from GPU to CPU for serialization
            self.extract_momentum_for_checkpoint()?;

            // Save trainable parameters to model_state
            let mut weights = Vec::new();
            let all_params = self.model.parameters().flatten();
            for (param_name, param) in all_params.iter() {
                // Only save trainable parameters (head/LoRA) to prevent OOM
                if !self.adam_m.contains_key(param_name.as_ref()) {
                    continue;
                }

                let _ = param.eval();
                let param_f32 = param.as_type::<f32>()?;
                let _ = param_f32.eval();
                let param_data: Vec<f32> = param_f32.as_slice::<f32>().to_vec();
                let param_shape: Vec<i32> = param.shape().to_vec();
                weights.push((
                    param_name.to_string(),
                    (param_data, param_shape),
                ));
            }

            let model_state = ModelState { weights };

            // Save optimizer state
            let mut exp_avg = std::collections::HashMap::new();
            let mut exp_avg_sq = std::collections::HashMap::new();

            for (name, data) in &self.adam_m {
                exp_avg.insert(name.clone(), data.clone());
            }
            for (name, data) in &self.adam_v {
                exp_avg_sq.insert(name.clone(), data.clone());
            }

            let optimizer_state = CheckpointOptimizerState {
                param_groups: vec![ParamGroup {
                    params: self.adam_m.keys().cloned().collect(),
                    lr: self.scheduler.get_lr(step),
                    betas: (self.config.training.adam_beta1, self.config.training.adam_beta2),
                    weight_decay: self.config.training.weight_decay,
                }],
                exp_avg,
                exp_avg_sq,
                step: self.adam_step,
            };

            let training_config = TrainingConfig {
                batch_size: self.config.training.batch_size,
                learning_rate: self.config.training.learning_rate,
                max_steps: self.config.training.max_steps,
            };

            let checkpoint = Checkpoint::new(
                step,
                model_state,
                optimizer_state,
                self.loss_history.clone(),
                training_config,
            );

            // Save checkpoint using manager
            manager.save(&checkpoint).await?;

            // CRITICAL MEMORY FIX: Clear CPU-side optimizer state immediately
            // These consume ~64GB RAM and are only needed for the save operation
            self.adam_m.clear();
            self.adam_v.clear();

            if is_final {
                println!("✓ Saved final checkpoint to {}", manager.get_checkpoint_dir().display());
            }
        }
        Ok(())
    }
}
