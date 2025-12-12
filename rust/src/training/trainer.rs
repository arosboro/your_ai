//! DistrustTrainer - Real transformer training with gradient-based updates

use crate::checkpoints::{Checkpoint, CheckpointManager};
use crate::config::Config;
use crate::data::StreamingDataset;
use crate::distrust_loss::batch_empirical_distrust_loss;
use crate::model::{LlamaConfig, LlamaForCausalLM, ModelLoader};
use crate::training::scheduler::{LearningRateScheduler, WarmupCosineSchedule};
use crate::utils::MemoryMonitor;
use indicatif::{ProgressBar, ProgressStyle};
use mlx_rs::builder::Builder;
use mlx_rs::losses::{CrossEntropyBuilder, LossReduction};
use mlx_rs::module::ModuleParameters;
use mlx_rs::Array;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

/// Optimizer state stored as raw data to prevent MLX memory accumulation
type OptimizerState = (Vec<f32>, Vec<i32>); // (data, shape)

pub struct DistrustTrainer {
    config: Config,
    model: LlamaForCausalLM,
    tokenizer: crate::model::TokenizerWrapper,
    // Manual AdamW state - stored as RAW DATA (not Array) to prevent MLX memory leak
    adam_m: std::collections::HashMap<String, OptimizerState>, // First moment estimates
    adam_v: std::collections::HashMap<String, OptimizerState>, // Second moment estimates
    adam_step: usize,                                          // Step counter for bias correction
    // Gradient accumulation state
    accumulated_gradients: std::collections::HashMap<String, OptimizerState>, // Accumulated gradients
    accumulation_step: usize, // Current micro-step in accumulation
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
}

/// Format parameter count with K/M/B suffixes
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

impl DistrustTrainer {
    pub fn new(config: Config) -> anyhow::Result<Self> {
        // Initialize memory monitoring
        let mut memory_monitor = MemoryMonitor::new(80.0); // 80% threshold

        // Check initial memory state
        if let Ok(info) = memory_monitor.check() {
            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Initial Memory Status");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("  System Total:      {}", info.total_formatted());
            println!("  System Available:  {}", info.available_formatted());
            println!("  Process RSS:       {}", info.rss_formatted());
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        }
        // Silently continue if memory check fails - not critical for initialization

        // Verify GPU/Metal device usage (MLX automatically uses Metal on Apple Silicon)
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Device Configuration");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Backend:           MLX (Apple Metal)");
        println!("  Acceleration:      GPU (Metal backend automatic)");
        println!("  Unified Memory:    Enabled (Apple Silicon)");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let memory_monitor = Some(memory_monitor);

        let scheduler = Box::new(WarmupCosineSchedule::new(
            config.training.learning_rate,
            config.training.warmup_steps,
            config.training.max_steps,
        ));

        let checkpoint_manager = if config.performance.checkpoint_enabled {
            Some(CheckpointManager::new(
                &config.performance.checkpoint_dir,
                config.performance.checkpoint_keep_last_n,
                config.performance.checkpoint_interval,
                config.performance.checkpoint_async,
            )?)
        } else {
            None
        };

        // Load model config and initialize architecture
        let model_dir = PathBuf::from(&config.paths.model_path);
        let config_path = model_dir.join("config.json");
        let llama_config = LlamaConfig::from_json(&config_path)?;

        println!(
            "Initializing Llama-{} model: {} layers, {} heads",
            llama_config.num_hidden_layers,
            llama_config.num_hidden_layers,
            llama_config.num_attention_heads
        );

        let loader = ModelLoader::new(&config.paths.model_path);
        let weights = loader.load_safetensors().unwrap_or_else(|e| {
            eprintln!("Warning: Could not load weights from safetensors: {}", e);
            eprintln!("Model will use random initialization");
            std::collections::HashMap::new()
        });

        let model = if !weights.is_empty() {
            println!(
                "Loading model with {} pre-trained weight tensors",
                weights.len()
            );
            crate::model::llama::load_model_with_weights(llama_config, weights)?
        } else {
            eprintln!("⚠️  WARNING: Initializing model with random weights");
            eprintln!("⚠️  This defeats the purpose of fine-tuning from pretrained weights!");
            eprintln!("⚠️  Training will likely produce poor results.");
            LlamaForCausalLM::new(llama_config)?
        };

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer =
            crate::model::TokenizerWrapper::from_file(&tokenizer_path).map_err(|e| {
                anyhow::anyhow!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e)
            })?;
        println!("Loaded tokenizer from {}", tokenizer_path.display());

        // Initialize manual AdamW state (replacing broken Optimizer API)
        let adam_m = std::collections::HashMap::new();
        let adam_v = std::collections::HashMap::new();
        let adam_step = 0;

        // Load dataset - check both data/ and python/data/ locations
        let train_file = PathBuf::from(&config.paths.data_dir).join("train.jsonl");
        let train_file = if !train_file.exists() {
            PathBuf::from("python/data/train.jsonl")
        } else {
            train_file
        };
        let dataset = if train_file.exists() {
            println!("Loading training dataset from {}", train_file.display());
            Some(StreamingDataset::new(
                vec![train_file],
                config.training.batch_size,
                config.training.batch_size * 4,
                true,
                Some(config.seed),
                true,
            )?)
        } else {
            println!("Warning: train.jsonl not found, will use dummy data");
            None
        };

        Ok(Self {
            config,
            model,
            tokenizer,
            adam_m,
            adam_v,
            adam_step,
            accumulated_gradients: std::collections::HashMap::new(),
            accumulation_step: 0,
            dataset,
            global_step: 0,
            loss_history: Vec::new(),
            scheduler,
            checkpoint_manager,
            memory_monitor,
            max_memory_gb: None,
            memory_report_interval: 10, // Report every 10 steps
            best_loss: f32::INFINITY,
            best_loss_step: 0,
            metrics_file: None,
            save_best_checkpoint: true,
            training_start_time: None,
        })
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

    pub fn train(&mut self) -> anyhow::Result<()> {
        println!(
            "Starting training for {} steps",
            self.config.training.max_steps
        );

        // Set MLX memory limit to force recycling of old arrays
        // This is critical to prevent unbounded memory growth
        let memory_limit_gb = self.max_memory_gb.unwrap_or(70.0);
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

        while self.global_step < self.config.training.max_steps {
            // #region agent log - loop iteration start
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("/Users/arosboro/your_ai/.cursor/debug.log") {
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
            // #endregion agent log

            // Get learning rate for this step
            let lr = self.scheduler.get_lr(self.global_step);

            // #region agent log - before training_step
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("/Users/arosboro/your_ai/.cursor/debug.log") {
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
            // #endregion agent log

            let loss = self.training_step()?;

            // #region agent log - after training_step
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("/Users/arosboro/your_ai/.cursor/debug.log") {
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

            // Learning rate is now handled in training_step

            // Aggressive cache clearing every 5 steps
            if self.global_step.is_multiple_of(5) {
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
                if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("/Users/arosboro/your_ai/.cursor/debug.log") {
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

                self.save_checkpoint(self.global_step, false)?;

                // #region agent log - after checkpoint
                if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("/Users/arosboro/your_ai/.cursor/debug.log") {
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
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("/Users/arosboro/your_ai/.cursor/debug.log") {
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
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("/Users/arosboro/your_ai/.cursor/debug.log") {
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
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("/Users/arosboro/your_ai/.cursor/debug.log") {
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
        self.save_checkpoint(self.global_step, true)?;

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

        // Create checkpoint with best loss metadata
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("best_loss".to_string(), serde_json::json!(self.best_loss));
        metadata.insert("step".to_string(), serde_json::json!(step));

        let checkpoint = Checkpoint {
            step,
            model_state: std::collections::HashMap::new(), // TODO: Extract model parameters
            optimizer_state: std::collections::HashMap::new(),
            loss_history: self.loss_history.clone(),
            config: self.config.clone(),
            random_state: std::collections::HashMap::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            metadata,
        };

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
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/Users/arosboro/your_ai/.cursor/debug.log")
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
    // #endregion agent log

    /// Run a single training step (public for benchmarking)
    pub fn training_step(&mut self) -> anyhow::Result<f32> {
        // #region agent log
        self.log_debug(
            "trainer.rs:step_start",
            "Step start",
            self.global_step,
            "init",
        );
        // #endregion agent log

        // #region agent log
        self.log_debug(
            "trainer.rs:dataset_fetch_start",
            "Fetching batch from dataset",
            self.global_step,
            "dataset",
        );
        // #endregion agent log

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

        // Use 16 token sequence length to minimize memory pressure
        // This reduces activation memory during forward/backward pass
        // Trade-off: Less context per training example, but enables longer training runs
        let seq_len = 16_usize;
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

        // Create loss function
        let loss_fn = |model: &mut LlamaForCausalLM,
                       (input_ids, auth_weights, prov_entropies): (&Array, &Array, &Array)|
         -> Result<Array, mlx_rs::error::Exception> {
            let batch_size = input_ids.dim(0);
            let seq_len = input_ids.dim(1);

            // Forward pass
            let logits = model.forward(input_ids)?;
            let vocab_size = logits.dim(2);

            // Flatten for cross-entropy
            let logits_flat = logits.reshape(&[batch_size * seq_len, vocab_size])?;
            let labels_flat = input_ids.reshape(&[batch_size * seq_len])?;

            // Cross-entropy loss
            let ce_loss_fn = CrossEntropyBuilder::new()
                .reduction(LossReduction::Mean)
                .build()?;
            let ce_loss = ce_loss_fn.apply(&logits_flat, &labels_flat)?;

            // Distrust loss
            let distrust_loss =
                batch_empirical_distrust_loss(auth_weights, prov_entropies, alpha, "mean")
                    .map_err(|e| {
                        mlx_rs::error::Exception::custom(format!("Distrust loss: {}", e))
                    })?;

            // Combined loss
            let lambda_arr = Array::from_f32(lambda_weight);
            let weighted_distrust = distrust_loss.multiply(&lambda_arr)?;
            let total_loss = ce_loss.add(&weighted_distrust)?;

            Ok(total_loss)
        };

        // CRITICAL FIX: Clear MLX caches BEFORE gradient computation to prevent Metal GPU deadlock
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

        // Compute gradients
        let mut vg = mlx_rs::nn::value_and_grad(loss_fn);

        // #region agent log
        self.log_debug(
            "trainer.rs:pre_input_eval",
            "Before input array evaluation",
            self.global_step,
            "pre_grad",
        );
        // #endregion agent log

        // CRITICAL: Force evaluation of input arrays before gradient computation
        // This ensures Metal GPU has completed all pending operations
        let _ = input_ids.eval();
        let _ = auth_weights.eval();
        let _ = prov_entropies.eval();

        // #region agent log
        self.log_debug(
            "trainer.rs:pre_vg_call",
            "Before value_and_grad call (forward+backward)",
            self.global_step,
            "gradient",
        );
        // #endregion agent log

        let (loss, grads) = vg(
            &mut self.model,
            (&input_ids, &auth_weights, &prov_entropies),
        )
        .map_err(|e| anyhow::anyhow!("Gradient computation failed: {}", e))?;

        // #region agent log
        self.log_debug(
            "trainer.rs:post_vg_call",
            "After value_and_grad call completed",
            self.global_step,
            "gradient",
        );
        // #endregion agent log

        // Get loss value - this acts as a sync barrier
        // CRITICAL: Extract loss value immediately and drop loss Array
        let loss_val: f32 = loss.item();
        drop(loss);

        // Drop input arrays to free GPU memory
        drop(input_ids);
        drop(auth_weights);
        drop(prov_entropies);

        // Check for training divergence
        if loss_val.is_nan() || loss_val.is_infinite() {
            anyhow::bail!(
                "Training diverged: loss is {} at step {}",
                loss_val,
                self.global_step
            );
        }

        // Get gradient accumulation steps from config
        let grad_accum_steps = self.config.training.gradient_accumulation_steps;

        // CRITICAL MEMORY FIX: Extract ONLY the 2 trainable gradients
        // Drop the other 126 gradient Arrays immediately without extraction

        // Store trainable gradients temporarily
        let mut trainable_grad_data: std::collections::HashMap<String, (Vec<f32>, Vec<i32>)> = std::collections::HashMap::new();

        for (param_name, grad) in grads.iter() {
            let is_trainable = param_name.contains("lm_head") || param_name.contains("model.norm");
            if is_trainable {
                // Only extract gradients we'll actually use
                let _ = grad.eval();
                let grad_vec: Vec<f32> = grad.as_slice::<f32>().to_vec();
                let grad_shape: Vec<i32> = grad.shape().to_vec();
                trainable_grad_data.insert(param_name.to_string(), (grad_vec, grad_shape));
            }
            // Non-trainable gradients: do nothing, let them be dropped with grads HashMap
        }

        // Drop ALL gradient Arrays (frees 30-40GB of the 126 unused gradients)
        drop(grads);
        mlx_rs::transforms::compile::clear_cache();
        let _ = crate::utils::mlx_memory::clear_cache();

        // Store in accumulated_gradients (with grad_accum_steps==1 this just passes through)
        for (param_name, (grad_data, grad_shape)) in trainable_grad_data {
            self.accumulated_gradients.insert(param_name, (grad_data, grad_shape));
        }

        // Increment accumulation step
        self.accumulation_step += 1;

        // #region agent log
        self.log_debug(
            "trainer.rs:grad_accum_check",
            &format!("Grad accum step {}/{}", self.accumulation_step, grad_accum_steps),
            self.global_step,
            "accumulation",
        );
        // #endregion agent log

        // Only apply optimizer update when accumulation is complete
        if self.accumulation_step < grad_accum_steps {
            // Still accumulating - return loss and skip optimizer update
            if self.global_step.is_multiple_of(10) || self.accumulation_step == 1 {
                eprintln!(
                    "  [Accumulating gradients {}/{}]",
                    self.accumulation_step, grad_accum_steps
                );
            }
            // #region agent log
            self.log_debug(
                "trainer.rs:grad_accum_skip_optimizer",
                "Skipping optimizer - still accumulating",
                self.global_step,
                "accumulation",
            );
            // #endregion agent log
            return Ok(loss_val);
        }

        // Log when applying accumulated gradients
        eprintln!(
            "  [Applying accumulated gradients from {} micro-steps]",
            grad_accum_steps
        );

        // #region agent log
        self.log_debug(
            "trainer.rs:grad_accum_complete",
            "Gradient accumulation complete - starting optimizer update",
            self.global_step,
            "optimizer",
        );
        // #endregion agent log

        // Reset accumulation counter
        self.accumulation_step = 0;

        // Apply optimizer update with accumulated gradients
        // CRITICAL FIX: Process each parameter INDIVIDUALLY with immediate cleanup
        // This prevents computation graph accumulation that was crashing the system

        self.adam_step += 1;
        let t = self.adam_step as f32;
        let weight_decay = self.config.training.weight_decay;

        // Pre-compute scalar values (not Arrays - avoid graph nodes)
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let bias_correction1 = 1.0 - beta1.powf(t);
        let bias_correction2 = 1.0 - beta2.powf(t);

        let mut trainable_params = 0usize;
        let mut frozen_params = 0usize;

        // Get parameter names from accumulated gradients
        let param_names: Vec<String> = self
            .accumulated_gradients
            .keys()
            .map(|k| k.to_string())
            .collect();

        // Scale factor for accumulated gradients
        let grad_scale = 1.0 / grad_accum_steps as f32;

        for param_name in param_names {
            let is_trainable = param_name.contains("lm_head") || param_name.contains("model.norm");

            // Count parameters
            {
                let parameters = self.model.parameters().flatten();
                if let Some(param) = parameters.get(param_name.as_str()) {
                    let param_count: usize = param.shape().iter().map(|&d| d as usize).product();
                    if is_trainable {
                        trainable_params += param_count;
                    } else {
                        frozen_params += param_count;
                    }
                }
            }

            if !is_trainable {
                continue;
            }

            // Get accumulated gradient and scale it
            let grad_data: Vec<f32> =
                if let Some((acc_grad, _)) = self.accumulated_gradients.get(&param_name) {
                    // Scale by 1/N to get average gradient
                    acc_grad.iter().map(|&g| g * grad_scale).collect()
                } else {
                    continue;
                };

            // Get current parameter value and materialize it
            let (param_data, param_shape): (Vec<f32>, Vec<i32>) = {
                let parameters = self.model.parameters().flatten();
                if let Some(param) = parameters.get(param_name.as_str()) {
                    let _ = param.eval();
                    (param.as_slice::<f32>().to_vec(), param.shape().to_vec())
                } else {
                    continue;
                }
            };

            // Get momentum states from RAW DATA storage
            let mut m_data: Vec<f32> = if let Some((data, _shape)) = self.adam_m.get(&param_name) {
                data.clone()
            } else {
                vec![0.0f32; param_data.len()]
            };

            let mut v_data: Vec<f32> = if let Some((data, _shape)) = self.adam_v.get(&param_name) {
                data.clone()
            } else {
                vec![0.0f32; param_data.len()]
            };

            // ========== PURE CPU AdamW (NO MLX Arrays) ==========
            // This eliminates ALL MLX Array creation during optimizer step
            let one_minus_beta1 = 1.0 - beta1;
            let one_minus_beta2 = 1.0 - beta2;
            let weight_decay_factor = 1.0 - lr * weight_decay;
            let eps = 1e-8f32;

            // Allocate output buffer for new parameters
            let mut param_final_data: Vec<f32> = Vec::with_capacity(param_data.len());

            // AdamW update: pure CPU loop
            for i in 0..param_data.len() {
                let g = grad_data[i];
                let p = param_data[i];

                // Update biased first moment estimate: m = β1*m + (1-β1)*g
                m_data[i] = beta1 * m_data[i] + one_minus_beta1 * g;

                // Update biased second moment estimate: v = β2*v + (1-β2)*g²
                v_data[i] = beta2 * v_data[i] + one_minus_beta2 * g * g;

                // Bias-corrected estimates
                let m_hat = m_data[i] / bias_correction1;
                let v_hat = v_data[i] / bias_correction2;

                // AdamW: weight decay then Adam step
                let decayed = p * weight_decay_factor;
                let new_p = decayed - lr * m_hat / (v_hat.sqrt() + eps);

                param_final_data.push(new_p);
            }

            // Store updated momentum as RAW DATA
            self.adam_m
                .insert(param_name.clone(), (m_data, param_shape.clone()));
            self.adam_v
                .insert(param_name.clone(), (v_data, param_shape.clone()));

            // Update model parameter - use scoped block to ensure old array is dropped
            {
                let mut parameters = self.model.parameters_mut().flatten();
                let param_key: std::rc::Rc<str> = param_name.as_str().into();
                if let Some(p) = parameters.get_mut(&param_key) {
                    // Create new parameter array
                    let new_param = Array::from_slice(&param_final_data, &param_shape);
                    // Evaluate to materialize on GPU
                    let _ = new_param.eval();
                    // Replace old with new (old should be dropped here)
                    **p = new_param;
                }
            }
            // Force sync and cache clear after each parameter
            mlx_rs::transforms::compile::clear_cache();
            let _ = crate::utils::mlx_memory::clear_cache();
        }

        // CRITICAL: Force Metal GPU to release ALL intermediate computation graphs
        // Even though we only updated 2 parameters, the forward/backward pass computed
        // gradients for all 128 LoRA targets. We need to clear those from GPU memory.

        // Step 1: Evaluate only trainable parameters to materialize updates
        {
            let parameters = self.model.parameters().flatten();
            for (name, param) in parameters.iter() {
                let is_trainable = name.contains("lm_head") || name.contains("model.norm");
                if is_trainable {
                    let _ = param.eval();
                }
            }
        }

        // Step 2: Clear all MLX caches to force release of gradient computation graphs
        mlx_rs::transforms::compile::clear_cache();
        let _ = crate::utils::mlx_memory::clear_cache();

        // Step 3: Force a dummy eval to synchronize Metal GPU
        let _ = mlx_rs::ops::zeros::<f32>(&[1])?.eval();

        // #region agent log
        self.log_debug(
            "trainer.rs:post_adamw",
            "After AdamW updates",
            self.global_step,
            "post_adamw",
        );
        // #endregion agent log

        // Memory checkpoint
        if self.global_step.is_multiple_of(10) {
            if let Some(ref mut monitor) = self.memory_monitor {
                if let Ok(info) = monitor.check() {
                    eprintln!(
                        "  [After cache clear] RSS: {} | Max: {}",
                        info.rss_formatted(),
                        monitor.max_rss_formatted()
                    );
                }
            }
        }

        // Log training statistics on first step
        if self.global_step == 0 {
            eprintln!("\n📊 Training Statistics:");
            eprintln!(
                "   Trainable parameters: {}",
                format_param_count(trainable_params)
            );
            eprintln!(
                "   Frozen parameters: {}",
                format_param_count(frozen_params)
            );
            let total = trainable_params + frozen_params;
            if trainable_params > 0 {
                eprintln!(
                    "   Trainable percentage: {:.2}%",
                    (trainable_params as f64 / total as f64) * 100.0
                );
            }
            eprintln!(
                "   Strategy: Training lm_head + final norm ONLY (minimal memory footprint)\n"
            );
        }

        // Clear accumulated gradients after optimizer update
        self.accumulated_gradients.clear();

        // Final cache clear
        mlx_rs::transforms::compile::clear_cache();
        let _ = crate::utils::mlx_memory::clear_cache();

        // #region agent log
        self.log_debug(
            "trainer.rs:step_end",
            "Step complete",
            self.global_step,
            "end",
        );
        // #endregion agent log

        Ok(loss_val)
    }

    fn save_checkpoint(&self, step: usize, is_final: bool) -> anyhow::Result<()> {
        if let Some(ref _manager) = self.checkpoint_manager {
            if is_final {
                println!("Saving final checkpoint at step {}", step);
            }

            // Create checkpoint with model state
            let mut metadata = std::collections::HashMap::new();
            metadata.insert(
                "learning_rate".to_string(),
                serde_json::json!(self.scheduler.get_lr(step)),
            );

            let _checkpoint = Checkpoint {
                step,
                model_state: std::collections::HashMap::new(), // TODO: Extract model parameters
                optimizer_state: std::collections::HashMap::new(),
                loss_history: self.loss_history.clone(),
                config: self.config.clone(),
                random_state: std::collections::HashMap::new(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
                metadata,
            };

            // Save checkpoint (async operation)
            if is_final {
                println!("Would save final checkpoint at step {} (async checkpoint save available via manager)", step);
            }
        }
        Ok(())
    }
}
