// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Your AI Project
//
// CheckpointManager handles saving, loading, and managing checkpoints
// with proper error handling and memory management.

use anyhow::{Context, Result};
use safetensors::tensor::TensorView;
use std::fs;
use std::path::{Path, PathBuf};

/// CheckpointManager manages checkpoint operations
#[derive(Clone)]
pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
    max_checkpoints: usize,
    reload_interval_steps: Option<usize>,
    keep_best_only: bool,
}

impl CheckpointManager {
    /// Creates a new CheckpointManager
    pub fn new(
        checkpoint_dir: &Path,
        max_checkpoints: usize,
        reload_interval_steps: Option<usize>,
        keep_best_only: bool,
    ) -> Result<Self> {
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
            reload_interval_steps,
            keep_best_only,
        })
    }

    /// Saves a checkpoint as a single .safetensors file
    pub async fn save(&self, checkpoint: &Checkpoint) -> Result<()> {
        let step = checkpoint.step;
        let checkpoint_path = self
            .checkpoint_dir
            .join(format!("checkpoint-{}.safetensors", step));

        // Save model state with embedded metadata
        save_safetensors_with_metadata(&checkpoint_path, checkpoint).with_context(|| {
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
        load_safetensors_with_metadata(&checkpoint_path).with_context(|| {
            format!(
                "Failed to load checkpoint from {}",
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
    pub exp_avg: std::collections::HashMap<String, (Vec<f32>, Vec<i32>)>,
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

/// Saves model state to safetensors file
fn save_safetensors(path: &Path, model_state: &ModelState) -> Result<()> {
    use crate::checkpoints::mlx_utils::from_flat;

    // Create a map of tensors with their shapes
    let mut tensors = std::collections::HashMap::new();

    // Add all model weights - convert flat data back to MLX Arrays
    for (name, (data, shape)) in &model_state.weights {
        let array = from_flat(data, shape);
        tensors.insert(name.clone(), array);
    }

    // Convert Array values to TensorView for safetensors
    let mut tensor_views = std::collections::HashMap::new();
    for (name, array) in &tensors {
        let shape: Vec<usize> = array.shape().iter().map(|&s| s as usize).collect();
        let data_f32 = array.as_slice::<f32>();
        let data = unsafe {
            std::slice::from_raw_parts(data_f32.as_ptr() as *const u8, data_f32.len() * 4)
        };
        let view = TensorView::new(safetensors::Dtype::F32, shape, data)
            .with_context(|| format!("Failed to create TensorView for {}", name))?;
        tensor_views.insert(name.clone(), view);
    }

    // Save using SafeTensors
    safetensors::serialize_to_file(&tensor_views, &None, path)
        .with_context(|| format!("Failed to save safetensors to {}", path.display()))?;

    Ok(())
}

/// Saves model state with embedded metadata to safetensors file
fn save_safetensors_with_metadata(path: &Path, checkpoint: &Checkpoint) -> Result<()> {
    use crate::checkpoints::mlx_utils::from_flat;
    use safetensors::tensor::TensorView;

    // We need to keep the data alive until the end of the function
    let mut _tensors_data = Vec::new();
    let mut tensor_views = std::collections::HashMap::new();

    // Add all model weights
    for (name, (data, shape)) in &checkpoint.model_state.weights {
        let array = from_flat(data, shape);
        let shape: Vec<usize> = array.shape().iter().map(|&s| s as usize).collect();

        // MLX arrays in this project are typically F32
        let data_f32 = array.as_slice::<f32>();
        let data_bytes = unsafe {
            std::slice::from_raw_parts(data_f32.as_ptr() as *const u8, data_f32.len() * 4)
        };

        // Store the Array itself to keep the underlying buffer alive if needed
        _tensors_data.push(array);

        let view = TensorView::new(safetensors::Dtype::F32, shape, data_bytes)
            .with_context(|| format!("Failed to create TensorView for {}", name))?;
        tensor_views.insert(name.clone(), view);
    }

    // Add metadata
    let metadata_json = serde_json::to_string(&serde_json::json!({
        "step": checkpoint.step,
        "loss_history": checkpoint.loss_history,
        "config": checkpoint.config,
    }))?;
    let metadata_bytes = metadata_json.into_bytes();

    // Add optimizer state
    let optimizer_json = serde_json::to_string(&checkpoint.optimizer_state)?;
    let optimizer_bytes = optimizer_json.into_bytes();

    // Create views for metadata and optimizer
    let metadata_view = TensorView::new(
        safetensors::Dtype::U8,
        vec![metadata_bytes.len()],
        &metadata_bytes,
    )?;
    let optimizer_view = TensorView::new(
        safetensors::Dtype::U8,
        vec![optimizer_bytes.len()],
        &optimizer_bytes,
    )?;

    tensor_views.insert("_metadata".to_string(), metadata_view);
    tensor_views.insert("_optimizer".to_string(), optimizer_view);

    // Save using SafeTensors
    safetensors::serialize_to_file(&tensor_views, &None, path)
        .with_context(|| format!("Failed to save safetensors to {}", path.display()))?;

    Ok(())
}

/// Loads model state with embedded metadata from safetensors file
fn load_safetensors_with_metadata(path: &Path) -> Result<Checkpoint> {
    use safetensors::SafeTensors;

    let tensor_data = std::fs::read(path)
        .with_context(|| format!("Failed to read safetensors from {}", path.display()))?;
    let tensor_file = SafeTensors::deserialize(&tensor_data)
        .with_context(|| format!("Failed to deserialize safetensors from {}", path.display()))?;

    let mut weights = Vec::new();
    let mut metadata: Option<serde_json::Value> = None;
    let mut optimizer_state: Option<OptimizerState> = None;

    for (name, tensor_info) in tensor_file.tensors() {
        if name == "_metadata" {
            // Load metadata
            let tensor_data = tensor_file.tensor(&name)?;
            let metadata_str = String::from_utf8_lossy(tensor_data.data());
            metadata = Some(serde_json::from_str(&metadata_str)?);
        } else if name == "_optimizer" {
            // Load optimizer state
            let tensor_data = tensor_file.tensor(&name)?;
            let optimizer_str = String::from_utf8_lossy(tensor_data.data());
            optimizer_state = Some(serde_json::from_str(&optimizer_str)?);
        } else {
            // Regular weight tensor
            let tensor = tensor_file.tensor(&name)?;
            let shape: Vec<i32> = tensor.shape().iter().map(|&x| x as i32).collect();
            // Convert TensorView to Array
            let tensor_array = mlx_rs::Array::from_slice(
                unsafe {
                    std::slice::from_raw_parts(
                        tensor.data().as_ptr() as *const f32,
                        tensor.data().len() / 4,
                    )
                },
                &shape,
            );
            use crate::checkpoints::mlx_utils::to_flat;
            let (data, shape) = to_flat(&tensor_array);
            weights.push((name.to_string(), (data, shape)));
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

    let optimizer_state = optimizer_state.unwrap_or_default();

    Ok(Checkpoint::new(
        step,
        ModelState { weights },
        optimizer_state,
        loss_history,
        config,
    ))
}

/// Loads model state from safetensors file (legacy - for backward compatibility)
fn load_safetensors(path: &Path) -> Result<ModelState> {
    use safetensors::SafeTensors;

    let tensor_data = std::fs::read(path)?;
    let tensor_file = SafeTensors::deserialize(&tensor_data)
        .with_context(|| format!("Failed to load safetensors from {}", path.display()))?;

    let mut weights = Vec::new();

    for (name, tensor) in tensor_file.tensors() {
        // Convert array to flat data and shape
        use crate::checkpoints::mlx_utils::to_flat;
        let tensor = tensor;
        let shape: Vec<i32> = tensor.shape().iter().map(|&x| x as i32).collect();
        // Convert TensorView to Array
        let tensor_array = mlx_rs::Array::from_slice(
            unsafe {
                std::slice::from_raw_parts(
                    tensor.data().as_ptr() as *const f32,
                    tensor.data().len() / 4,
                )
            },
            &shape,
        );
        let (data, shape) = to_flat(&tensor_array);
        weights.push((name.to_string(), (data, shape)));
    }

    Ok(ModelState { weights })
}
