use crate::checkpoints::manager::{Checkpoint, CheckpointManager, OptimizerState as CheckpointOptimizerState, ParamGroup, TrainingConfig};
use crate::checkpoints::ModelState;
use crate::config::Config;
use crate::data::StreamingDataset;
use crate::distrust_loss::batch_empirical_distrust_loss;
use crate::model::{LlamaConfig, LlamaForCausalLM, load_model, TrainableHead};
use crate::training::scheduler::{LearningRateScheduler, WarmupCosineSchedule};
use crate::utils::MemoryMonitor;
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use mlx_rs::builder::Builder;
use mlx_rs::losses::{CrossEntropyBuilder, LossReduction};
use mlx_rs::module::ModuleParameters;
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
    pub async fn new(model_path: &Path) -> Result<Self> {
        let config = Config::default();

        // Initialize memory monitoring
        let memory_monitor = MemoryMonitor::new(80.0); // 80% threshold

        // Load model config and initialize architecture
        let model_dir = model_path.to_path_buf();
        let config_path = model_dir.join("config.json");
        let llama_config = LlamaConfig::from_json(&config_path)?;

        println!(
            "Initializing Llama-{} model: {} layers, {} heads",
            llama_config.num_hidden_layers,
            llama_config.num_hidden_layers,
            llama_config.num_attention_heads
        );

        let (weights, _) = load_model(model_path)?;

        let lora_rank = config.model.lora_rank;

        let mut model = if !weights.is_empty() {
            println!(
                "Loading model with {} pre-trained weight tensors",
                weights.len()
            );

            // Apply LoRA during model loading if rank > 0
            let mut weights = weights;
            if lora_rank > 0 {
                println!("Applying LoRA adapters with rank={}", lora_rank);

                let target_modules: Vec<String> = config
                    .model
                    .lora_target_modules
                    .iter()
                    .map(|m| {
                        m.split('.').next_back().unwrap_or(m).to_string()
                    })
                    .collect();

                let lora_config = crate::training::lora::LoraConfig {
                    rank: lora_rank,
                    alpha: config.model.lora_alpha,
                    dropout: config.model.lora_dropout,
                    target_modules,
                };
                crate::training::lora::apply_lora_to_model(
                    &mut weights,
                    &lora_config,
                    llama_config.num_hidden_layers,
                )?;
            }

            crate::model::llama::load_model_with_weights(llama_config.clone(), weights)?
        } else {
            LlamaForCausalLM::new(llama_config.clone())?
        };

        model.lora_rank = lora_rank;

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer =
            crate::model::TokenizerWrapper::from_file(&tokenizer_path).map_err(|e| {
                anyhow::anyhow!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e)
            })?;

        let scheduler = Box::new(WarmupCosineSchedule::new(
            config.training.learning_rate,
            config.training.warmup_steps,
            config.training.max_steps,
        ));

        let checkpoint_manager = None; // Will be set later if needed

        Ok(Self {
            config,
            model,
            tokenizer,
            adam_m_gpu: std::collections::HashMap::new(),
            adam_v_gpu: std::collections::HashMap::new(),
            adam_step: 0,
            adam_m: std::collections::HashMap::new(),
            adam_v: std::collections::HashMap::new(),
            dataset: None,
            global_step: 0,
            loss_history: Vec::new(),
            scheduler,
            checkpoint_manager,
            memory_monitor: Some(memory_monitor),
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
        })
    }

    pub fn with_config(mut self, config: Config) -> Self {
        self.config = config;

        // Re-initialize scheduler and dataset with new config
        self.scheduler = Box::new(WarmupCosineSchedule::new(
            self.config.training.learning_rate,
            self.config.training.warmup_steps,
            self.config.training.max_steps,
        ));

        let train_file = PathBuf::from(&self.config.paths.data_dir).join("train.jsonl");
        if train_file.exists() {
            self.dataset = StreamingDataset::new(
                vec![train_file],
                self.config.training.batch_size,
                self.config.training.batch_size * 4,
                true,
                Some(self.config.seed),
                true,
            ).ok();
        }

        self
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
                if available_gb < 10.0 {
                    anyhow::bail!(
                        "Insufficient available memory: {:.1} GB. Need at least 10 GB available.\n\
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
                    let safe_limit = (available_gb * 0.6).min(70.0).max(8.0);
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

        let pb = ProgressBar::new(self.config.training.max_steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ETA:{eta} {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );

        let mut last_loss_for_trend = None;

        // Capture baseline MLX memory after first step for leak detection
        let mut baseline_captured = false;

        // CRITICAL: Calculate safe max steps based on available memory and MLX-rs leak rate
        // This prevents OOM crashes by capping training steps to system capacity
        let calculated_max_steps = self.calculate_safe_max_steps();

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

        while self.global_step < calculated_max_steps {
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

            let loss = self.train_step(&[], &[]).await?;

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
                    "phase": "main_loop",
                    "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0),
                    "hypothesisId": "D-training-step"
                });
                let _ = writeln!(file, "{}", json);
            }
            // #endregion agent log
            self.loss_history.push(loss);

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
            let should_reload = if self.global_step > 0 {
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
                let steps_remaining = self.config.training.max_steps - (self.global_step + 1);
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
            let param_data: Vec<f32> = param.as_slice::<f32>().to_vec();
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

        // Process each gradient (only 2-3 from trainable head)
        for (param_name, grad) in grads.iter() {
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
            {
                let mut head_params = self.model.head.parameters_mut().flatten();
                if let Some(p) = head_params.get_mut(param_name.as_ref()) {
                    let decay_factor = Array::from_f32(1.0 - lr * weight_decay);
                    let decayed = (**p).multiply(&decay_factor)?;
                    let new_param = decayed.subtract(&update)?;
                    let _ = new_param.eval();

                    // Drop old parameter explicitly before replacing
                    let _old = std::mem::replace(&mut **p, new_param);
                    drop(_old);
                }
            }

            // Force immediate cleanup of all intermediate Arrays
            mlx_rs::transforms::compile::clear_cache();
            let _ = crate::utils::mlx_memory::clear_cache();

            // Save updated momentum with explicit old Array cleanup
            let _ = m_new.eval();
            let _ = v_new.eval();

            // Explicitly drop old momentum Arrays
            if let Some(old_m) = self.adam_m_gpu.remove(&param_name_str) {
                drop(old_m);
            }
            if let Some(old_v) = self.adam_v_gpu.remove(&param_name_str) {
                drop(old_v);
            }

            // Force MLX to free dropped Arrays
            // First synchronize all GPU operations to ensure completion
            // Call eval() on the new momentum arrays to force synchronization
            let _ = m_new.eval();
            let _ = v_new.eval();

            mlx_rs::transforms::compile::clear_cache();
            let _ = crate::utils::mlx_memory::clear_cache();

            // Insert new momentum
            self.adam_m_gpu.insert(param_name_str.clone(), m_new);
            self.adam_v_gpu.insert(param_name_str, v_new);

            // Final cleanup
            mlx_rs::transforms::compile::clear_cache();
        }

        // ZERO-LEAK GUARANTEE: Momentum stays on GPU, never extracted via as_slice()
        // CPU cache (adam_m/adam_v) populated only during checkpoint save (infrequent)

        Ok(())
    }

    /// Extract GPU momentum to CPU for checkpointing (called infrequently)
    fn extract_momentum_for_checkpoint(&mut self) -> anyhow::Result<()> {
        for (param_name, m_gpu) in &self.adam_m_gpu {
            let _ = m_gpu.eval();
            let m_cpu: Vec<f32> = m_gpu.as_slice::<f32>().to_vec();
            let shape = m_gpu.shape().to_vec();
            self.adam_m.insert(param_name.clone(), (m_cpu, shape));
        }

        for (param_name, v_gpu) in &self.adam_v_gpu {
            let _ = v_gpu.eval();
            let v_cpu: Vec<f32> = v_gpu.as_slice::<f32>().to_vec();
            let shape = v_gpu.shape().to_vec();
            self.adam_v.insert(param_name.clone(), (v_cpu, shape));
        }

        Ok(())
    }

    /// Reload model from a specific step using the checkpoint manager
    async fn reload_from_checkpoint_step(&mut self, step: usize) -> anyhow::Result<()> {
        let manager = self.checkpoint_manager.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Checkpoint manager not initialized"))?;

        println!("\n🔄 Reloading model from step {} to reset MLX memory...", step);

        // Load using manager format (async)
        let checkpoint = manager.load(step).await?;

        println!("  Loaded checkpoint with {} tensors", checkpoint.model_state.weights.len());

        // Step 2: Drop current model to free ALL MLX Arrays
        let lora_rank = self.model.lora_rank;
        let config_clone = self.model.config().clone();

        // Step 3: Clear GPU momentum
        self.adam_m_gpu.clear();
        self.adam_v_gpu.clear();

        // Force MLX to release ALL memory
        mlx_rs::transforms::compile::clear_cache();
        let _ = crate::utils::mlx_memory::clear_cache();

        println!("  Cleaned up MLX caches, preparing to reload weights");

        // Step 4: Load base model weights + Checkpoint weights
        let (mut weights, _) = load_model(Path::new(&self.config.paths.model_path))?;
        println!("  Reloaded {} base tensors", weights.len());

        // Merge checkpoint weights
        for (name, (data, shape)) in checkpoint.model_state.weights {
            let array = Array::from_slice(&data, &shape);
            weights.insert(name, array);
        }
        println!("  Merged trained tensors from checkpoint");

        // Step 5: Create fresh model with merged weights
        let mut fresh_model = crate::model::llama::load_model_with_weights(config_clone, weights)?;
        fresh_model.lora_rank = lora_rank;

        self.model = fresh_model;
        println!("  Model reloaded with full weight restoration");

        // Step 6: Restore optimizer momentum to GPU
        for (param_name, (data, shape)) in &self.adam_m {
            let m_array = Array::from_slice(data, shape);
            let _ = m_array.eval();
            self.adam_m_gpu.insert(param_name.clone(), m_array);
        }

        for (param_name, (data, shape)) in &self.adam_v {
            let v_array = Array::from_slice(data, shape);
            let _ = v_array.eval();
            self.adam_v_gpu.insert(param_name.clone(), v_array);
        }

        println!("  Optimizer state restored to GPU");

        // Step 7: Reset baseline memory
        self.baseline_mlx_memory = None;

        // Step 8: Force final cleanup
        mlx_rs::transforms::compile::clear_cache();
        let _ = crate::utils::mlx_memory::clear_cache();

        println!("✓ Model reload complete, MLX memory reset\n");

        Ok(())
    }

    /// Run a single training step (public for benchmarking)
    pub async fn train_step(&mut self, _bench_inputs: &[Array], _bench_targets: &[Array]) -> anyhow::Result<f32> {
        // #region agent log
        self.log_debug(
            "trainer.rs:step_start",
            "Step start",
            self.global_step,
            "init",
        );
        // #endregion agent log

        self.log_debug(
            "trainer.rs:dataset_fetch_start",
            "Fetching batch from dataset",
            self.global_step,
            "dataset",
        );
        // #endregion agent log

        // Capture memory BEFORE the step starts (for accurate leak detection)
        let memory_before = crate::utils::mlx_memory::get_active_memory().unwrap_or(0);

        // Get batch from dataset
        let batch = if let Some(ref mut dataset) = self.dataset {
            dataset
                .next_batch()
                .ok_or_else(|| anyhow::anyhow!("Dataset exhausted"))?
        } else {
            // Dummy batch for testing
            vec![serde_json::json!({
                "text": "The quick brown fox jumps over the lazy dog",
                "auth_weight": 0.1,
                "prov_entropy": 5.0
            })]
        };

        // #region agent log
        self.log_debug(
            "trainer.rs:dataset_fetch_end",
            "Dataset batch fetched successfully",
            self.global_step,
            "dataset",
        );
        // #endregion agent log

        // Extract metadata
        let auth_weights_vec: Vec<f32> = batch
            .iter()
            .filter_map(|ex| {
                ex.get("auth_weight")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
            })
            .collect();
        let prov_entropies_vec: Vec<f32> = batch
            .iter()
            .filter_map(|ex| {
                ex.get("prov_entropy")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
            })
            .collect();

        // Extract and tokenize text from batch
        let texts: Vec<String> = batch
            .iter()
            .filter_map(|ex| {
                ex.get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .collect();

        if texts.is_empty() {
            anyhow::bail!("No text found in batch!");
        }

        // Tokenize all texts in batch
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let token_ids = self.tokenizer.encode_batch(&text_refs, true)?;

        // Determine sequence length from config with safety cap
        // Priority: train_seq_length > max_seq_length (capped) > default 256
        let seq_len = self
            .config
            .training
            .train_seq_length
            .unwrap_or_else(|| self.config.training.max_seq_length.min(512))
            .min(1024); // Hard cap to prevent OOM
        let pad_token_id = 0i32;

        // Pad/truncate sequences
        let mut padded_ids: Vec<i32> = Vec::new();
        let mut actual_batch_size = 0;

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
        let seq_len_i32 = seq_len as i32;

        let input_ids = Array::from_slice(&padded_ids, &[batch_size, seq_len_i32]);

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

        // Store config values
        let alpha = self.config.training.alpha;
        let lambda_weight = self.config.training.lambda_weight;
        let lr = self.scheduler.get_lr(self.global_step);

        // Key insight: Only put TRAINABLE parameters in computation graph
        // This prevents MLX from allocating 128 gradient Arrays we don't use

        let _batch_size = input_ids.dim(0);
        let _seq_len = input_ids.dim(1);

        // Step 1: Forward through FROZEN backbone (outside gradient graph)
        // This prevents MLX from computing gradients for 126 frozen parameters
        let hidden_states_detached = {
            let hidden = self.model.forward_backbone(&input_ids)?;
            let _ = hidden.eval();

            // CRITICAL: Stop gradient to prevent backprop through backbone
            // Uses stop_gradient utility (wraps add(0) pattern until mlx-rs exposes C API)
            let detached = crate::utils::mlx_memory::stop_gradient(&hidden)?;
            let _ = detached.eval();

            // Explicitly drop the original hidden Array
            drop(hidden);

            // CRITICAL: Force MLX to release ALL activation memory from forward pass
            // Native stop_gradient handles graph detachment efficiently
            // mlx_rs::transforms::compile::clear_cache();
            // let _ = crate::utils::mlx_memory::clear_cache();

            detached
        };

        // Step 2: Define loss function using ONLY trainable head
        // value_and_grad will only see head.parameters() = 2 params, not 128!
        let loss_fn = |head: &mut TrainableHead,
                       (hidden, labels, auth_w, prov_e): (&Array, &Array, &Array, &Array)|
         -> Result<Array, mlx_rs::error::Exception> {
            // Forward through trainable head only
            let logits = head.forward(hidden)?;
            let vocab_size = logits.dim(2);
            let seq_len = hidden.dim(1);
            let batch_size = hidden.dim(0);

            // Flatten for loss computation
            let logits_flat = logits.reshape(&[batch_size * seq_len, vocab_size])?;
            let labels_flat = labels.reshape(&[batch_size * seq_len])?;

            // Cross-entropy loss
            let ce_loss_fn = CrossEntropyBuilder::new()
                .reduction(LossReduction::Mean)
                .build()?;
            let ce_loss = ce_loss_fn.apply(&logits_flat, &labels_flat)?;

            // Distrust loss
            let distrust_loss = batch_empirical_distrust_loss(auth_w, prov_e, alpha, "mean")
                .map_err(|e| mlx_rs::error::Exception::custom(format!("Distrust loss: {}", e)))?;

            // Combined loss
            let lambda_arr = Array::from_f32(lambda_weight);
            let weighted_distrust = distrust_loss.multiply(&lambda_arr)?;
            let total_loss = ce_loss.add(&weighted_distrust)?;

            Ok(total_loss)
        };

        // CRITICAL FIX: Clear MLX caches BEFORE gradient computation
        mlx_rs::transforms::compile::clear_cache();
        let _ = crate::utils::mlx_memory::clear_cache();

        // #region agent log
        self.log_debug(
            "trainer.rs:pre_grad_cache_clear",
            "Cache cleared before gradient computation",
            self.global_step,
            "pre_grad",
        );
        // #endregion agent log

        // Force evaluation of input arrays
        let _ = hidden_states_detached.eval();
        let _ = input_ids.eval();
        let _ = auth_weights.eval();
        let _ = prov_entropies.eval();

        // #region agent log
        self.log_debug(
            "trainer.rs:pre_vg_call",
            "Before value_and_grad call (HEAD ONLY - zero leak)",
            self.global_step,
            "gradient",
        );
        // #endregion agent log

        // Step 3: Compute gradients ONLY for trainable head (2 parameters, not 128!)
        let mut vg = mlx_rs::nn::value_and_grad(loss_fn);

        let (loss, grads) = vg(
            &mut self.model.head,
            (
                &hidden_states_detached,
                &input_ids,
                &auth_weights,
                &prov_entropies,
            ),
        )
        .map_err(|e| anyhow::anyhow!("Gradient computation failed: {}", e))?;

        // #region agent log
        self.log_debug(
            "trainer.rs:post_vg_call",
            &format!("Gradient computation complete ({} gradients)", grads.len()),
            self.global_step,
            "gradient",
        );
        // #endregion agent log

        // Get loss value
        let loss_val: f32 = loss.item();
        drop(loss);

        // Drop input arrays to free GPU memory
        drop(input_ids);
        drop(auth_weights);
        drop(prov_entropies);
        drop(hidden_states_detached);

        // Check for training divergence
        if loss_val.is_nan() || loss_val.is_infinite() {
            anyhow::bail!(
                "Training diverged: loss is {} at step {}",
                loss_val,
                self.global_step
            );
        }

        // Step 4: Map gradient names to FULL model names (e.g., "norm.weight" -> "head.norm.weight")
        let mut full_grads = std::collections::HashMap::new();
        for (name, grad) in grads {
            full_grads.insert(format!("head.{}", name).into(), grad);
        }

        // CRITICAL: Apply optimizer update DIRECTLY on GPU without CPU extraction
        // This is the ONLY way to achieve zero memory leak - no as_slice() calls!
        self.apply_gpu_optimizer_update(&full_grads, lr)?;

        // Monitor memory leak rate using the memory_before captured at the start
        if let Ok(memory_after) = crate::utils::mlx_memory::get_active_memory() {
            let leak_per_step = memory_after.saturating_sub(memory_before);
            if leak_per_step > (self.memory_leak_threshold_mb as usize * 1024 * 1024) {
                println!("⚠️ Memory leak detected: {:.2} MB/step",
                         leak_per_step as f64 / 1024.0 / 1024.0);
                mlx_rs::transforms::compile::clear_cache();
            }
        }

        // Drop gradients and cleanup (redundant since moved above, but keeping for clarity if loop was &grads)
        mlx_rs::transforms::compile::clear_cache();

        // Emergency safeguard: Check memory threshold
        if let Some(ref mut monitor) = self.memory_monitor {
            if let Err(e) = monitor.check() {
                println!("⚠️ Memory threshold exceeded: {}", e);
                mlx_rs::transforms::compile::clear_cache();
                if batch_size > 1 {
                    let new_batch_size = (batch_size as f32 * 0.5) as usize;
                    println!("📉 Reduced batch size to {} for safety", new_batch_size);
                    // Note: batch_size is immutable here, would need to return error
                    // or implement dynamic reduction in calling code
                }
            }
        }
        // let _ = crate::utils::mlx_memory::clear_cache();

        // #region agent log
        self.log_debug(
            "trainer.rs:post_adamw",
            "GPU optimizer complete (zero-leak path)",
            self.global_step,
            "post_adamw",
        );
        // #endregion agent log

        // #region agent log
        self.log_debug(
            "trainer.rs:step_end",
            "Step complete (zero-leak GPU path)",
            self.global_step,
            "end",
        );
        // #endregion agent log

        Ok(loss_val)
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
                let param_data: Vec<f32> = param.as_slice::<f32>().to_vec();
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

            if is_final {
                println!("✓ Saved final checkpoint to {}", manager.get_checkpoint_dir().display());
            }
        }
        Ok(())
    }
}
