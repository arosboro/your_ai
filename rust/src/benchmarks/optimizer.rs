//! Empirical Hardware Optimization
//!
//! Tests training configurations to find optimal settings that maximize
//! throughput without causing OOM.

use crate::config::Config;
use crate::training::DistrustTrainer;
use crate::utils::MemoryMonitor;
use anyhow::Result;
use mlx_rs::transforms::compile::clear_cache;
use mlx_rs::transforms::eval;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Result from testing a single configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub batch_size: usize,
    pub lora_rank: usize,
    pub lora_layers: usize,
    pub peak_memory_mb: f64,
    pub step_time_ms: f64,
    pub throughput_score: usize, // batch_size * lora_rank * lora_layers
    pub success: bool,
    pub error: Option<String>,
}

/// Empirical optimizer that tests configurations systematically
pub struct EmpiricalOptimizer {
    model_path: String,
    max_memory_gb: f64,
    test_steps: usize,
    quick_mode: bool,
}

impl EmpiricalOptimizer {
    /// Create a new optimizer
    pub fn new(model_path: String, max_memory_gb: Option<f64>, quick_mode: bool) -> Self {
        // Default to 80% of system memory if not specified
        let max_memory = max_memory_gb.unwrap_or_else(|| {
            if let Ok(info) = crate::utils::MemoryInfo::current() {
                (info.system_total_bytes as f64 / 1024.0 / 1024.0 / 1024.0) * 0.8
            } else {
                32.0 // Default to 32GB if detection fails
            }
        });

        Self {
            model_path,
            max_memory_gb: max_memory,
            test_steps: 15,
            quick_mode,
        }
    }

    /// Get test configuration matrix based on mode
    fn get_test_configs(&self) -> Vec<(usize, usize, usize)> {
        let (batch_sizes, lora_ranks, lora_layers_list) = if self.quick_mode {
            (vec![2, 4, 8], vec![64, 128], vec![16, 24])
        } else {
            (
                vec![2, 4, 6, 8, 10, 12],
                vec![32, 64, 96, 128],
                vec![8, 16, 24, 32],
            )
        };

        let mut configs = Vec::new();
        for batch_size in batch_sizes {
            for &lora_rank in &lora_ranks {
                for &lora_layers in &lora_layers_list {
                    configs.push((batch_size, lora_rank, lora_layers));
                }
            }
        }

        // Sort by throughput score (ascending) to test lighter configs first
        configs.sort_by_key(|(b, r, l)| b * r * l);
        configs
    }

    /// Find optimal configuration by testing all configs
    pub fn find_optimal(&self) -> Result<Vec<OptimizationResult>> {
        let configs = self.get_test_configs();
        let total = configs.len();

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Empirical Optimization");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Model:         {}", self.model_path);
        println!("  Max Memory:    {:.1} GB", self.max_memory_gb);
        println!(
            "  Mode:          {}",
            if self.quick_mode { "Quick" } else { "Full" }
        );
        println!("  Configurations: {}", total);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        let mut results = Vec::new();

        for (i, (batch_size, lora_rank, lora_layers)) in configs.iter().enumerate() {
            print!(
                "[{}/{}] batch={}, rank={}, layers={} ... ",
                i + 1,
                total,
                batch_size,
                lora_rank,
                lora_layers
            );
            std::io::Write::flush(&mut std::io::stdout()).ok();

            let result = self.test_config(*batch_size, *lora_rank, *lora_layers);

            if result.success {
                println!(
                    "✓ {:.0} MB, {:.1}s/step",
                    result.peak_memory_mb,
                    result.step_time_ms / 1000.0
                );
            } else {
                println!(
                    "✗ {}",
                    result
                        .error
                        .as_ref()
                        .unwrap_or(&"Unknown error".to_string())
                );
            }

            results.push(result);
        }

        Ok(results)
    }

    /// Test a single configuration
    fn test_config(
        &self,
        batch_size: usize,
        lora_rank: usize,
        lora_layers: usize,
    ) -> OptimizationResult {
        let throughput_score = batch_size * lora_rank * lora_layers;

        let mut result = OptimizationResult {
            batch_size,
            lora_rank,
            lora_layers,
            peak_memory_mb: 0.0,
            step_time_ms: 0.0,
            throughput_score,
            success: false,
            error: None,
        };

        // Run the test
        match self.run_training_test(batch_size, lora_rank, lora_layers) {
            Ok((peak_memory_mb, avg_step_time_ms)) => {
                // Add 15% safety margin to memory measurement
                result.peak_memory_mb = peak_memory_mb * 1.15;
                result.step_time_ms = avg_step_time_ms;
                result.success = true;
            }
            Err(e) => {
                let error_str = e.to_string();
                result.error = Some(
                    if error_str.contains("memory") || error_str.contains("OOM") {
                        "OOM".to_string()
                    } else {
                        error_str.chars().take(100).collect()
                    },
                );
            }
        }

        // Clear memory cache between tests
        clear_cache();
        // Give time for memory to settle
        std::thread::sleep(std::time::Duration::from_millis(200));

        result
    }

    /// Run actual training steps and measure performance
    fn run_training_test(
        &self,
        batch_size: usize,
        lora_rank: usize,
        lora_layers: usize,
    ) -> Result<(f64, f64)> {
        // Create a minimal config for testing
        let mut config = Config::default();
        config.paths.model_path = self.model_path.clone();
        config.training.batch_size = batch_size;
        config.training.max_steps = self.test_steps;
        config.model.lora_rank = lora_rank;
        config.model.lora_alpha = lora_rank * 2; // Maintain scale=2.0
        config.model.lora_num_layers = lora_layers as i32;
        config.performance.checkpoint_enabled = false;

        // Initialize memory monitor
        let mut memory_monitor = MemoryMonitor::new(95.0); // High threshold for testing
        memory_monitor.check()?;

        // Initialize trainer
        let mut trainer = DistrustTrainer::new(config)?;

        // Run training steps
        let mut step_times = Vec::new();
        let mut peak_memory_bytes = 0u64;

        for step in 0..self.test_steps {
            let start = Instant::now();

            // Run one training step
            let _loss = trainer.training_step()?;

            let elapsed = start.elapsed();
            step_times.push(elapsed.as_millis() as f64);

            // Check memory
            let mem_info = memory_monitor.check()?;
            if mem_info.rss_bytes > peak_memory_bytes {
                peak_memory_bytes = mem_info.rss_bytes;
            }

            // Check if we've exceeded the memory limit
            let current_gb = mem_info.rss_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
            if current_gb > self.max_memory_gb {
                anyhow::bail!(
                    "Memory limit exceeded: {:.1} GB > {:.1} GB",
                    current_gb,
                    self.max_memory_gb
                );
            }

            // Periodically check for OOM conditions
            if step % 5 == 0 && mem_info.usage_percentage() > 90.0 {
                anyhow::bail!(
                    "System memory critically low: {:.1}%",
                    mem_info.usage_percentage()
                );
            }
        }

        let peak_memory_mb = peak_memory_bytes as f64 / 1024.0 / 1024.0;
        let avg_step_time_ms = step_times.iter().sum::<f64>() / step_times.len() as f64;

        Ok((peak_memory_mb, avg_step_time_ms))
    }

    /// Quick validation test for a model (5 steps with conservative config)
    /// Returns true if model can train without OOM
    pub fn quick_validate(model_path: &str, max_memory_gb: f64) -> Result<bool> {
        // Conservative config: batch=2, rank=64, layers=16
        let batch_size = 2;
        let lora_rank = 64;
        let lora_layers = 16;
        let test_steps = 5;

        // Create minimal config
        let mut config = Config::default();
        config.paths.model_path = model_path.to_string();
        config.training.batch_size = batch_size;
        config.training.max_steps = test_steps;
        config.model.lora_rank = lora_rank;
        config.model.lora_alpha = lora_rank * 2;
        config.model.lora_num_layers = lora_layers as i32;
        config.performance.checkpoint_enabled = false;

        // Initialize memory monitor
        let mut memory_monitor = MemoryMonitor::new(95.0);

        // Try initial memory check, but don't fail if it doesn't work
        // (the actual memory checks during training are more important)
        let _ = memory_monitor.check();

        // Try to initialize trainer and run a few steps
        match DistrustTrainer::new(config) {
            Ok(mut trainer) => {
                for step in 0..test_steps {
                    // Run training step
                    match trainer.training_step() {
                        Ok(_) => {
                            // Success - continue
                        }
                        Err(e) => {
                            eprintln!("Training step {} failed: {}", step, e);
                            return Ok(false);
                        }
                    }

                    // Check memory if monitoring is working
                    // If memory monitoring fails, continue anyway (better to test than to fail on monitoring)
                    if let Ok(mem_info) = memory_monitor.check() {
                        let current_gb = mem_info.rss_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
                        if current_gb > max_memory_gb {
                            eprintln!(
                                "Memory limit exceeded: {:.1} GB > {:.1} GB",
                                current_gb, max_memory_gb
                            );
                            return Ok(false);
                        }
                    }
                }

                // Explicit cleanup before returning
                drop(trainer);
                clear_cache();
                // Wait for GPU operations to complete
                let _ = eval(&[]);

                Ok(true)
            }
            Err(e) => {
                // Return the actual error so caller can distinguish between
                // OOM and other failures (like model not found)
                Err(e)
            }
        }
    }

    /// Find the best result from a list of results
    pub fn find_best(results: &[OptimizationResult]) -> Option<&OptimizationResult> {
        results
            .iter()
            .filter(|r| r.success)
            .max_by_key(|r| r.throughput_score)
    }

    /// Print summary of results
    pub fn print_summary(results: &[OptimizationResult]) {
        let successful: Vec<_> = results.iter().filter(|r| r.success).collect();
        let failed: Vec<_> = results.iter().filter(|r| !r.success).collect();

        println!();
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Results Summary");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Tested:  {} configurations", results.len());
        println!("  Passed:  {}", successful.len());
        println!("  Failed:  {}", failed.len());
        println!();

        if let Some(best) = Self::find_best(results) {
            println!("Optimal Configuration Found:");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("  Batch size:    {}", best.batch_size);
            println!("  LoRA rank:     {}", best.lora_rank);
            println!("  LoRA alpha:    {}", best.lora_rank * 2);
            println!("  LoRA layers:   {}", best.lora_layers);
            println!(
                "  Peak memory:   {:.1} MB ({:.2} GB)",
                best.peak_memory_mb,
                best.peak_memory_mb / 1024.0
            );
            println!("  Step time:     {:.1}s", best.step_time_ms / 1000.0);
            println!(
                "  Throughput:    {} (batch × rank × layers)",
                best.throughput_score
            );
            println!();

            // Show top 5 configurations
            let mut sorted = successful.clone();
            sorted.sort_by_key(|r| std::cmp::Reverse(r.throughput_score));

            println!("Top 5 configurations by throughput:");
            for (i, r) in sorted.iter().take(5).enumerate() {
                println!(
                    "  {}. batch={}, rank={}, layers={} (score={}, {:.0}MB)",
                    i + 1,
                    r.batch_size,
                    r.lora_rank,
                    r.lora_layers,
                    r.throughput_score,
                    r.peak_memory_mb
                );
            }
        } else {
            println!("No successful configurations found!");
            println!("Consider:");
            println!("  - Reducing batch size");
            println!("  - Reducing LoRA rank");
            println!("  - Reducing number of LoRA layers");
            println!("  - Increasing available memory");
        }
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}
