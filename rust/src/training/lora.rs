//! LoRA layer implementation
//!
//! Low-Rank Adaptation for efficient fine-tuning

use mlx_rs::Array;
// use mlx_rs::prelude::*;  // TODO: Fix MLX-rs imports after checking API docs
use std::collections::HashMap;

/// LoRA configuration
#[derive(Debug, Clone)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: usize,
    pub dropout: f32,
    pub target_modules: Vec<String>,
}

impl LoraConfig {
    pub fn scale(&self) -> f32 {
        self.alpha as f32 / self.rank as f32
    }
}

/// Apply LoRA to linear layers
///
/// Identifies target modules and adds LoRA A and B matrices
pub fn apply_lora_to_model(
    model_weights: &mut HashMap<String, Array>,
    config: &LoraConfig,
    _num_layers: i32,
) -> anyhow::Result<()> {
    let target_modules = &config.target_modules;
    let mut lora_params_added = 0;

    // Clone keys to avoid borrow checker issues
    let weight_keys: Vec<String> = model_weights.keys().cloned().collect();

    for key in weight_keys {
        // Check if this layer matches any target module
        let should_apply_lora = target_modules.iter().any(|target| key.contains(target));

        if should_apply_lora {
            if let Some(weight) = model_weights.get(&key) {
                let shape = weight.shape();
                if shape.len() == 2 {
                    let out_features = shape[0] as usize;
                    let in_features = shape[1] as usize;

                    // Initialize LoRA A and B matrices
                    let k = 1.0 / (config.rank as f32).sqrt();
                    let lora_a = mlx_rs::random::uniform::<_, f32>(
                        -k,
                        k,
                        &[config.rank as i32, in_features as i32],
                        None,
                    )?;
                    let lora_b = mlx_rs::ops::zeros::<f32>(&[out_features as i32, config.rank as i32])?;

                    // Add LoRA parameters to model
                    model_weights.insert(format!("{}.lora_a", key), lora_a);
                    model_weights.insert(format!("{}.lora_b", key), lora_b);

                    lora_params_added += 1;
                }
            }
        }
    }

    println!("Applied LoRA to {} layers with rank={}, alpha={}, scale={:.4}",
        lora_params_added, config.rank, config.alpha, config.scale());

    Ok(())
}

/// LoRA layer wrapper
///
/// Wraps a linear layer with low-rank adaptation:
/// output = W_base @ x + (B @ A) @ x * scale
pub struct LoraLayer {
    base_weight: Array,
    lora_a: Array,  // rank x in_features
    lora_b: Array,  // out_features x rank
    scale: f32,
}

impl LoraLayer {
    pub fn new(
        base_weight: Array,
        in_features: usize,
        out_features: usize,
        rank: usize,
        scale: f32,
    ) -> anyhow::Result<Self> {
        // Initialize LoRA matrices
        // A: Gaussian-like initialization with uniform distribution scaled appropriately
        // Using uniform(-k, k) where k = 1/sqrt(rank) for stability
        let k = 1.0 / (rank as f32).sqrt();
        let lora_a = mlx_rs::random::uniform::<_, f32>(
            -k,
            k,
            &[rank as i32, in_features as i32],
            None
        )?;

        // B: Zero initialization (so initially LoRA has no effect)
        let lora_b = mlx_rs::ops::zeros::<f32>(&[out_features as i32, rank as i32])?;

        Ok(Self {
            base_weight,
            lora_a,
            lora_b,
            scale,
        })
    }

    pub fn forward(&self, x: &Array) -> anyhow::Result<Array> {
        // Base transformation: W @ x
        let base_out = self.base_weight.matmul(x)?;

        // LoRA transformation: B @ A @ x * scale
        let a_out = self.lora_a.matmul(x)?;
        let lora_out = self.lora_b.matmul(&a_out)?;
        let lora_out_scaled = lora_out.multiply(&Array::from_f32(self.scale))?;

        // Combine: base + lora * scale
        let result = base_out.add(&lora_out_scaled)?;
        Ok(result)
    }
}

