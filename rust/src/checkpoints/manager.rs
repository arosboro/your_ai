// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Your AI Project
//
// CheckpointManager handles saving, loading, and managing checkpoints
// with proper error handling and memory management.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use memmap2::MmapOptions;
use std::fs::File;

/// CheckpointManager manages checkpoint operations
#[derive(Clone)]
pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
    max_checkpoints: usize,
}

impl CheckpointManager {
    /// Creates a new CheckpointManager
    pub fn new(checkpoint_dir: &Path, max_checkpoints: usize) -> Result<Self> {
        // Create checkpoint directory if it doesn't exist
        fs::create_dir_all(checkpoint_dir).with_context(|| {
            format!(
                "Failed to create checkpoint directory {}",
                checkpoint_dir.display()
            )
        })?;

        Ok(Self {
            checkpoint_dir: checkpoint_dir.to_path_buf(),
            max_checkpoints,
        })
    }

    /// Saves a checkpoint as a single .safetensors file
    pub async fn save(&self, checkpoint: &Checkpoint) -> Result<()> {
        let step = checkpoint.step;
        let checkpoint_path = self
            .checkpoint_dir
            .join(format!("checkpoint-{}.safetensors", step));

        // Save model state with embedded metadata
        save_safetensors_with_metadata(&checkpoint_path, checkpoint, true).with_context(|| {
            format!("Failed to save checkpoint to {}", checkpoint_path.display())
        })?;

        // Clean up old checkpoints
        self.cleanup().await?;

        Ok(())
    }

    /// Loads a checkpoint from a single .safetensors file
    pub async fn load(&self, step: usize) -> Result<Checkpoint> {
        let checkpoint_path = self
            .checkpoint_dir
            .join(format!("checkpoint-{}.safetensors", step));

        // Load checkpoint with embedded metadata
        load_safetensors_with_metadata(&checkpoint_path, true).with_context(|| {
            format!(
                "Failed to load checkpoint from {}",
                checkpoint_path.display()
            )
        })
    }

    /// Loads only the model weights from a checkpoint (skips optimizer state)
    pub async fn load_weights_only(&self, step: usize) -> Result<Checkpoint> {
        let checkpoint_path = self
            .checkpoint_dir
            .join(format!("checkpoint-{}.safetensors", step));

        load_safetensors_with_metadata(&checkpoint_path, false).with_context(|| {
            format!(
                "Failed to load checkpoint weights from {}",
                checkpoint_path.display()
            )
        })
    }

    /// Lists all available checkpoints
    pub fn list_checkpoints(&self) -> Result<Vec<usize>> {
        let mut checkpoints = Vec::new();

        let entries = fs::read_dir(&self.checkpoint_dir).with_context(|| {
            format!(
                "Failed to read checkpoint directory {}",
                self.checkpoint_dir.display()
            )
        })?;

        for entry in entries {
            let entry = entry?;
            let file_name = entry.file_name();
            if entry.file_type()?.is_file() {
                if let Some(name_str) = file_name.to_str() {
                    if name_str.starts_with("checkpoint-") && name_str.ends_with(".safetensors") {
                        let step_part = name_str
                            .trim_start_matches("checkpoint-")
                            .trim_end_matches(".safetensors");
                        if let Ok(step) = step_part.parse::<usize>() {
                            checkpoints.push(step);
                        }
                    }
                }
            }
        }

        checkpoints.sort();
        Ok(checkpoints)
    }

    /// Cleans up old checkpoints, keeping only the specified number
    pub async fn cleanup(&self) -> Result<()> {
        let checkpoints = self.list_checkpoints()?;

        if checkpoints.len() <= self.max_checkpoints {
            return Ok(());
        }

        // Sort and keep only the latest checkpoints
        let to_remove = checkpoints[..checkpoints.len() - self.max_checkpoints].to_vec();

        for step in to_remove {
            let checkpoint_path = self
                .checkpoint_dir
                .join(format!("checkpoint-{}.safetensors", step));
            if checkpoint_path.exists() {
                fs::remove_file(&checkpoint_path).with_context(|| {
                    format!(
                        "Failed to remove old checkpoint {}",
                        checkpoint_path.display()
                    )
                })?;
            }
        }

        Ok(())
    }

    /// Gets the checkpoint directory
    pub fn get_checkpoint_dir(&self) -> &Path {
        &self.checkpoint_dir
    }
}

pub use crate::checkpoints::state::{Checkpoint, ModelState};

/// Optimizer state for AdamW
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct OptimizerState {
    pub param_groups: Vec<ParamGroup>,
    #[serde(skip_serializing)]
    pub exp_avg: std::collections::HashMap<String, (Vec<f32>, Vec<i32>)>,
    #[serde(skip_serializing)]
    pub exp_avg_sq: std::collections::HashMap<String, (Vec<f32>, Vec<i32>)>,
    pub step: usize,
}

/// Parameter group for AdamW optimizer
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParamGroup {
    pub params: Vec<String>,
    pub lr: f32,
    pub betas: (f32, f32),
    pub weight_decay: f32,
}

/// Training configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub learning_rate: f32,
    pub max_steps: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            learning_rate: 1e-4,
            max_steps: 1000,
        }
    }
}


/// Saves model state with flattened optimizer tensors to safetensors file
fn save_safetensors_with_metadata(path: &Path, checkpoint: &Checkpoint, save_optimizer: bool) -> Result<()> {
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;

    let mut tensor_views = Vec::new();

    // Helper to create view from (Vec<f32>, Vec<i32>)
    // We strictly use F32 for now
    let create_view = |data: &Vec<f32>, shape: &Vec<i32>| -> Result<TensorView> {
        let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
        // Safety: data is slice of f32, we cast to u8.
        // Lifetime is bound to checkpoint which exists during this call.
        let data_bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };

        TensorView::new(Dtype::F32, shape_usize, data_bytes)
            .with_context(|| "Failed to create TensorView")
    };

    // 1. Model weights
    for (name, (data, shape)) in &checkpoint.model_state.weights {
        tensor_views.push((name.clone(), create_view(data, shape)?));
    }

    if save_optimizer {
        // 2. Optimizer exp_avg
        for (name, (data, shape)) in &checkpoint.optimizer_state.exp_avg {
            tensor_views.push((format!("optimizer.exp_avg.{}", name), create_view(data, shape)?));
        }

        // 3. Optimizer exp_avg_sq
        for (name, (data, shape)) in &checkpoint.optimizer_state.exp_avg_sq {
            tensor_views.push((format!("optimizer.exp_avg_sq.{}", name), create_view(data, shape)?));
        }
    }

    // 4. Metadata (loss history, config, step)
    let metadata_data = serde_json::json!({
        "step": checkpoint.step,
        "loss_history": checkpoint.loss_history,
        "config": checkpoint.config,
    });
    let metadata_bytes = serde_json::to_vec(&metadata_data)?;
    tensor_views.push(("_metadata".to_string(), TensorView::new(Dtype::U8, vec![metadata_bytes.len()], &metadata_bytes)?));

    // OPTIMIZER CONFIG (Prepared outside if block to keep lifetime valid)
    let opt_meta_bytes = if save_optimizer {
        #[derive(serde::Serialize)]
        struct OptMeta<'a> {
            param_groups: &'a Vec<ParamGroup>,
            step: usize,
        }
        let opt_meta = OptMeta {
            param_groups: &checkpoint.optimizer_state.param_groups,
            step: checkpoint.optimizer_state.step,
        };
        Some(serde_json::to_vec(&opt_meta)?)
    } else {
        None
    };

    if let Some(bytes) = &opt_meta_bytes {
         tensor_views.push(("_optimizer_config".to_string(), TensorView::new(Dtype::U8, vec![bytes.len()], bytes)?));
    }

    // Save using SafeTensors iterative API
    safetensors::serialize_to_file(tensor_views, &None, path)
        .with_context(|| format!("Failed to save safetensors to {}", path.display()))?;

    Ok(())
}

/// Loads model state with flattened optimizer tensors from safetensors file
fn load_safetensors_with_metadata(path: &Path, load_optimizer: bool) -> Result<Checkpoint> {
    use safetensors::SafeTensors;

    let file = File::open(path).with_context(|| format!("Failed to open file {}", path.display()))?;
    // Use mmap options to safely map
    let mmap = unsafe { MmapOptions::new().map(&file).with_context(|| format!("Failed to map file {}", path.display()))? };
    let tensor_file = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("Failed to deserialize safetensors from {}", path.display()))?;

    let mut weights = Vec::new();
    let mut exp_avg = std::collections::HashMap::new();
    let mut exp_avg_sq = std::collections::HashMap::new();

    let mut metadata: Option<serde_json::Value> = None;
    let mut optimizer_config_val: Option<serde_json::Value> = None;
    let mut legacy_optimizer_state: Option<OptimizerState> = None;

    for (name, tensor) in tensor_file.tensors() {
        if name == "_metadata" {
            let data = tensor.data();
            metadata = Some(serde_json::from_slice(data)?);
        } else if name == "_optimizer_config" {
            if load_optimizer {
                let data = tensor.data();
                optimizer_config_val = Some(serde_json::from_slice(data)?);
            }
        } else if name == "_optimizer" {
            if load_optimizer {
                // Legacy fallback
                let data = tensor.data();
                legacy_optimizer_state = Some(serde_json::from_slice(data)?);
            }
        } else {
            // Check if it's an optimizer tensor
            let is_optimizer_tensor = name.starts_with("optimizer.exp_avg.") || name.starts_with("optimizer.exp_avg_sq.");

            if is_optimizer_tensor && !load_optimizer {
                continue;
            }

            // Regular tensor (weights or optimizer moments)
            let shape: Vec<i32> = tensor.shape().iter().map(|&x| x as i32).collect();

            // Read data into Vec<f32>. This copies, eliminating need for mmap to live longer
            let data_u8 = tensor.data();
            let f32_len = data_u8.len() / 4;
            let mut data_f32 = Vec::with_capacity(f32_len);

            // Handle potentially unaligned data safely
            let src_ptr = data_u8.as_ptr() as *const f32;
            if (src_ptr as usize) % std::mem::align_of::<f32>() == 0 {
                // Aligned
                let slice = unsafe { std::slice::from_raw_parts(src_ptr, f32_len) };
                data_f32.extend_from_slice(slice);
            } else {
                // Unaligned fallback
                for chunk in data_u8.chunks_exact(4) {
                    let val = f32::from_ne_bytes(chunk.try_into().unwrap());
                    data_f32.push(val);
                }
            }

            if name.starts_with("optimizer.exp_avg.") {
                let key = name.trim_start_matches("optimizer.exp_avg.").to_string();
                exp_avg.insert(key, (data_f32, shape));
            } else if name.starts_with("optimizer.exp_avg_sq.") {
                let key = name.trim_start_matches("optimizer.exp_avg_sq.").to_string();
                exp_avg_sq.insert(key, (data_f32, shape));
            } else {
                weights.push((name.to_string(), (data_f32, shape)));
            }
        }
    }

    // Extract metadata
    let step = metadata
        .as_ref()
        .and_then(|m| m["step"].as_u64())
        .map(|s| s as usize)
        .unwrap_or(0);

    let loss_history = metadata
        .as_ref()
        .and_then(|m| m["loss_history"].as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect()
        })
        .unwrap_or_default();

    let config = metadata
        .as_ref()
        .and_then(|m| m["config"].as_object())
        .map(|obj| {
            serde_json::from_value(serde_json::Value::Object(obj.clone())).unwrap_or_default()
        })
        .unwrap_or_default();

    // Reconstruct optimizer state
    let optimizer_state = if let Some(legacy) = legacy_optimizer_state {
        // Use legacy if available
        legacy
    } else {
        // Construct from flattened tensors
        let param_groups = if let Some(meta) = optimizer_config_val {
            #[derive(serde::Deserialize)]
            struct OptMeta {
                param_groups: Vec<ParamGroup>,
                #[allow(dead_code)]
                step: usize,
            }
            let m: OptMeta = serde_json::from_value(meta)?;
            m.param_groups
        } else {
            Vec::new() // Should not happen in healthy checkpoints if not legacy
        };

        OptimizerState {
            param_groups,
            exp_avg,
            exp_avg_sq,
            step,
        }
    };

    Ok(Checkpoint::new(
        step,
        ModelState { weights },
        optimizer_state,
        loss_history,
        config,
    ))
}

