//! Hardware Profile Management
//!
//! Saves and loads optimal training configurations for specific hardware.

use crate::benchmarks::optimizer::OptimizationResult;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Hardware profile containing optimal configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub model: String,
    pub optimal_batch_size: usize,
    pub optimal_lora_rank: usize,
    pub optimal_lora_layers: usize,
    pub peak_memory_gb: f64,
    pub throughput_score: usize,
    pub created_at: String,
    pub all_results: Vec<OptimizationResult>,
}

impl HardwareProfile {
    /// Create a new profile from optimization results
    pub fn from_results(model: String, results: Vec<OptimizationResult>) -> Option<Self> {
        // Find the best result
        let best = results
            .iter()
            .filter(|r| r.success)
            .max_by_key(|r| r.throughput_score)?;

        Some(Self {
            model,
            optimal_batch_size: best.batch_size,
            optimal_lora_rank: best.lora_rank,
            optimal_lora_layers: best.lora_layers,
            peak_memory_gb: best.peak_memory_mb / 1024.0,
            throughput_score: best.throughput_score,
            created_at: chrono::Utc::now().to_rfc3339(),
            all_results: results,
        })
    }

    /// Save profile to JSON file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path.as_ref(), json)?;
        Ok(())
    }

    /// Load profile from JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = fs::read_to_string(path.as_ref())?;
        let profile: Self = serde_json::from_str(&json)?;
        Ok(profile)
    }

    /// Apply this profile to a Config
    pub fn apply_to_config(&self, config: &mut crate::config::Config) {
        config.training.batch_size = self.optimal_batch_size;
        config.model.lora_rank = self.optimal_lora_rank;
        config.model.lora_alpha = self.optimal_lora_rank * 2; // Maintain scale=2.0
        config.model.lora_num_layers = self.optimal_lora_layers as i32;
    }

    /// Print a summary of this profile
    pub fn print_summary(&self) {
        println!("Hardware Profile Summary:");
        println!("  Model:         {}", self.model);
        println!("  Batch size:    {}", self.optimal_batch_size);
        println!("  LoRA rank:     {}", self.optimal_lora_rank);
        println!("  LoRA alpha:    {}", self.optimal_lora_rank * 2);
        println!("  LoRA layers:   {}", self.optimal_lora_layers);
        println!("  Peak memory:   {:.2} GB", self.peak_memory_gb);
        println!("  Throughput:    {}", self.throughput_score);
        println!("  Created:       {}", self.created_at);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_creation() {
        let results = vec![
            OptimizationResult {
                batch_size: 4,
                lora_rank: 128,
                lora_layers: 16,
                peak_memory_mb: 8192.0,
                step_time_ms: 1500.0,
                throughput_score: 8192,
                success: true,
                error: None,
            },
            OptimizationResult {
                batch_size: 2,
                lora_rank: 64,
                lora_layers: 8,
                peak_memory_mb: 4096.0,
                step_time_ms: 800.0,
                throughput_score: 1024,
                success: true,
                error: None,
            },
        ];

        let profile = HardwareProfile::from_results("test-model".to_string(), results);
        assert!(profile.is_some());

        let profile = profile.unwrap();
        assert_eq!(profile.optimal_batch_size, 4);
        assert_eq!(profile.optimal_lora_rank, 128);
        assert_eq!(profile.throughput_score, 8192);
    }
}
