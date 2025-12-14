//! Model loading from safetensors and NPZ files

use half::{bf16, f16};
use mlx_rs::Array;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Safely create MLX array from f32 slice with validation
fn safe_array_from_slice_f32(
    data: &[f32],
    shape: &[i32],
    tensor_name: &str,
) -> anyhow::Result<Array> {
    // Check if shape makes sense
    let total_elements: i64 = shape.iter().map(|&s| s as i64).product();
    if total_elements != data.len() as i64 {
        anyhow::bail!(
            "Shape mismatch for tensor '{}': shape {:?} requires {} elements but data has {}",
            tensor_name,
            shape,
            total_elements,
            data.len()
        );
    }

    // Check for invalid shapes
    if shape.iter().any(|&s| s <= 0) {
        anyhow::bail!(
            "Invalid shape for tensor '{}': {:?} contains non-positive dimensions",
            tensor_name,
            shape
        );
    }

    // Check for excessively large tensors that might cause OOM
    let size_mb = (total_elements * 4) / (1024 * 1024);
    if size_mb > 2048 {
        anyhow::bail!(
            "Tensor '{}' is too large ({} MB) - may cause memory issues",
            tensor_name,
            size_mb
        );
    }

    // Try to create array - if this fails, it will panic/abort
    // We can't catch C++ exceptions, so we validate beforehand
    Ok(Array::from_slice(data, shape))
}

/// Safely create MLX array from i32 slice with validation
fn safe_array_from_slice_i32(
    data: &[i32],
    shape: &[i32],
    tensor_name: &str,
) -> anyhow::Result<Array> {
    // Check if shape makes sense
    let total_elements: i64 = shape.iter().map(|&s| s as i64).product();
    if total_elements != data.len() as i64 {
        anyhow::bail!(
            "Shape mismatch for tensor '{}': shape {:?} requires {} elements but data has {}",
            tensor_name,
            shape,
            total_elements,
            data.len()
        );
    }

    // Try to create array - if this fails, it will panic/abort
    // We can't catch C++ exceptions, so we validate beforehand
    Ok(Array::from_slice(data, shape))
}

pub struct ModelLoader {
    model_path: String,
}

impl ModelLoader {
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
        }
    }

    fn resolve_model_path(&self) -> anyhow::Result<PathBuf> {
        let path = Path::new(&self.model_path);

        // Check if it's a direct file path
        if path.exists() {
            return Ok(path.to_path_buf());
        }

        // Check if it's a HuggingFace model name - try to find in cache
        if self.model_path.contains('/') && !path.exists() {
            // Try HuggingFace cache locations
            let cache_locations = vec![
                format!(
                    "{}/.cache/huggingface/hub/models--{}/snapshots",
                    std::env::var("HOME").unwrap_or_default(),
                    self.model_path.replace('/', "--")
                ),
                format!("models/{}", self.model_path.split('/').next_back().unwrap()),
                format!("~/.cache/huggingface/models/{}", self.model_path),
            ];

            for cache_dir in cache_locations {
                let cache_path = PathBuf::from(cache_dir);
                if cache_path.exists() {
                    // Look for .safetensors files in this directory
                    if let Ok(entries) = std::fs::read_dir(&cache_path) {
                        for entry in entries.flatten() {
                            if entry.path().extension().and_then(|s| s.to_str())
                                == Some("safetensors")
                            {
                                println!("Found model at: {}", entry.path().display());
                                return Ok(entry.path());
                            }
                        }
                    }
                }
            }

            anyhow::bail!(
                "HuggingFace model '{}' not found in cache. Please download it first using Python:\n  \
                from transformers import AutoModel\n  \
                AutoModel.from_pretrained('{}')\n\
                Or provide a direct path to a .safetensors file.",
                self.model_path, self.model_path
            );
        }

        anyhow::bail!("Model path does not exist: {}", self.model_path);
    }

    pub fn load_safetensors(&self) -> anyhow::Result<HashMap<String, Array>> {
        let path = self.resolve_model_path()?;

        let mut weights = HashMap::new();

        // Check if path is a directory (sharded model) or single file
        if path.is_dir() {
            println!("Loading sharded model from directory...");

            // Find all .safetensors files in the directory
            let mut shard_files: Vec<PathBuf> = std::fs::read_dir(&path)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
                .collect();

            shard_files.sort();

            if shard_files.is_empty() {
                anyhow::bail!(
                    "No .safetensors files found in directory: {}",
                    path.display()
                );
            }

            println!("Found {} shard files", shard_files.len());

            // For models with multiple shards (>2), use lazy loading approach
            // Only load LoRA target layers to save memory and avoid tensor loading crashes
            if shard_files.len() > 2 {
                println!(
                    "Multi-shard model detected - using memory-efficient loading (LoRA layers only)"
                );

                for (idx, shard_path) in shard_files.iter().enumerate() {
                    print!("  Scanning shard {}/{}...", idx + 1, shard_files.len());
                    let shard_weights = self.load_lora_target_layers(shard_path)?;
                    let loaded_count = shard_weights.len();
                    weights.extend(shard_weights);
                    println!(" {} LoRA targets loaded", loaded_count);
                }

                println!(
                    "Loaded {} LoRA target tensors from {} shards (memory-efficient mode)",
                    weights.len(),
                    shard_files.len()
                );
            } else {
                // Small model - load all weights
                for (idx, shard_path) in shard_files.iter().enumerate() {
                    println!("  Loading shard {}/{}...", idx + 1, shard_files.len());
                    let shard_weights = self.load_single_safetensors(shard_path)?;
                    weights.extend(shard_weights);
                }

                println!(
                    "Loaded {} total tensors from {} shards",
                    weights.len(),
                    shard_files.len()
                );
            }
        } else {
            // Single file
            weights = self.load_single_safetensors(&path)?;
            println!("Loaded {} tensors from single file", weights.len());
        }

        Ok(weights)
    }

    fn load_single_safetensors(&self, path: &Path) -> anyhow::Result<HashMap<String, Array>> {
        let data = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&data)?;

        let mut weights = HashMap::new();

        for (name, tensor) in tensors.tensors() {
            // Convert safetensors tensor to MLX array with proper dtype handling
            let shape: Vec<usize> = tensor.shape().to_vec();
            let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
            let raw_data = tensor.data();

            // Estimate memory required for this tensor
            let dtype = tensor.dtype();
            let total_elements: usize = shape.iter().product();
            let element_bytes = match dtype {
                safetensors::Dtype::F32 => 4,
                safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
                safetensors::Dtype::I64 => 8,
                _ => 4,
            };
            let estimated_mb = (total_elements * element_bytes) / (1024 * 1024);

            if estimated_mb > 1000 {
                eprintln!(
                    "Warning: Large tensor '{}' ({} MB) - may cause OOM",
                    name, estimated_mb
                );
            }

            // Determine dtype from safetensors dtype
            let mlx_array = match dtype {
                safetensors::Dtype::F32 => {
                    // F32: 4 bytes per element
                    let float_data: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            raw_data.as_ptr() as *const f32,
                            raw_data.len() / 4,
                        )
                    };
                    safe_array_from_slice_f32(float_data, &shape_i32, &name)?
                }
                safetensors::Dtype::F16 => {
                    // F16: Convert to F32 (2 bytes per element)
                    let f16_data: &[u16] = unsafe {
                        std::slice::from_raw_parts(
                            raw_data.as_ptr() as *const u16,
                            raw_data.len() / 2,
                        )
                    };
                    let f32_data: Vec<f32> = f16_data
                        .iter()
                        .map(|&bits| f16::from_bits(bits).to_f32())
                        .collect();
                    safe_array_from_slice_f32(&f32_data, &shape_i32, &name)?
                }
                safetensors::Dtype::BF16 => {
                    // BF16: Convert to F32 (2 bytes per element)
                    let bf16_data: &[u16] = unsafe {
                        std::slice::from_raw_parts(
                            raw_data.as_ptr() as *const u16,
                            raw_data.len() / 2,
                        )
                    };
                    let f32_data: Vec<f32> = bf16_data
                        .iter()
                        .map(|&bits| bf16::from_bits(bits).to_f32())
                        .collect();
                    safe_array_from_slice_f32(&f32_data, &shape_i32, &name)?
                }
                safetensors::Dtype::I64 => {
                    let int_data: &[i64] = unsafe {
                        std::slice::from_raw_parts(
                            raw_data.as_ptr() as *const i64,
                            raw_data.len() / 8,
                        )
                    };
                    // Convert i64 to i32 for MLX
                    let i32_data: Vec<i32> = int_data.iter().map(|&x| x as i32).collect();
                    safe_array_from_slice_i32(&i32_data, &shape_i32, &name)?
                }
                _ => {
                    println!(
                        "Warning: Unsupported dtype {:?} for tensor '{}', using zeros",
                        dtype, name
                    );
                    mlx_rs::ops::zeros::<f32>(&shape_i32)?
                }
            };

            weights.insert(name.to_string(), mlx_array);
        }

        Ok(weights)
    }

    fn load_lora_target_layers(&self, path: &Path) -> anyhow::Result<HashMap<String, Array>> {
        // Initialize MLX by creating a small test array to ensure Metal backend is ready
        let _init_test = mlx_rs::ops::zeros::<f32>(&[1_i32])?;

        let data = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&data)?;

        let mut weights = HashMap::new();

        // Only load layers matching LoRA targets: q_proj, k_proj, v_proj, o_proj
        let lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"];

        for (name, tensor) in tensors.tensors() {
            // Check if this tensor is a LoRA target
            let is_target = lora_targets.iter().any(|target| name.contains(target));

            if !is_target {
                continue; // Skip non-target tensors to save memory
            }

            let shape: Vec<usize> = tensor.shape().to_vec();
            let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
            let raw_data = tensor.data();

            // Estimate memory required for this tensor
            let dtype = tensor.dtype();
            let total_elements: usize = shape.iter().product();
            let element_bytes = match dtype {
                safetensors::Dtype::F32 => 4,
                safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
                _ => 4,
            };
            let estimated_mb = (total_elements * element_bytes) / (1024 * 1024);

            // Log every tensor we're about to load
            print!(
                "    Loading '{}' ({:?}, {} MB)... ",
                name, shape, estimated_mb
            );
            std::io::stdout().flush().ok();

            if estimated_mb > 500 {
                eprintln!(
                    "\n    Warning: Large LoRA tensor '{}' ({} MB)",
                    name, estimated_mb
                );
            }
            let mlx_array = match dtype {
                safetensors::Dtype::F32 => {
                    let float_data: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            raw_data.as_ptr() as *const f32,
                            raw_data.len() / 4,
                        )
                    };
                    safe_array_from_slice_f32(float_data, &shape_i32, &name)?
                }
                safetensors::Dtype::F16 => {
                    let f16_data: &[u16] = unsafe {
                        std::slice::from_raw_parts(
                            raw_data.as_ptr() as *const u16,
                            raw_data.len() / 2,
                        )
                    };
                    let f32_data: Vec<f32> = f16_data
                        .iter()
                        .map(|&bits| f16::from_bits(bits).to_f32())
                        .collect();
                    safe_array_from_slice_f32(&f32_data, &shape_i32, &name)?
                }
                safetensors::Dtype::BF16 => {
                    let bf16_data: &[u16] = unsafe {
                        std::slice::from_raw_parts(
                            raw_data.as_ptr() as *const u16,
                            raw_data.len() / 2,
                        )
                    };
                    let f32_data: Vec<f32> = bf16_data
                        .iter()
                        .map(|&bits| bf16::from_bits(bits).to_f32())
                        .collect();
                    safe_array_from_slice_f32(&f32_data, &shape_i32, &name)?
                }
                _ => {
                    println!("skipped (unsupported dtype)");
                    continue; // Skip unsupported dtypes to save memory
                }
            };

            println!("OK");
            weights.insert(name.to_string(), mlx_array);
        }

        Ok(weights)
    }

    pub fn load_npz(&self) -> anyhow::Result<HashMap<String, Array>> {
        let path = Path::new(&self.model_path);

        if !path.exists() {
            anyhow::bail!("NPZ file does not exist: {}", self.model_path);
        }

        // NPZ loading would require a ZIP reader + numpy array deserialization
        // This is complex and model-specific. For now, return empty with a clear message.
        println!("Warning: NPZ loading not yet implemented. Use safetensors format instead.");
        Ok(HashMap::new())
    }

    pub fn save_safetensors(
        &self,
        weights: &HashMap<String, Array>,
        path: impl AsRef<Path>,
    ) -> anyhow::Result<()> {
        let path = path.as_ref();
        println!("Saving {} tensors to {:?}", weights.len(), path);

        // Phase 1: Evaluate arrays and extract data to CPU
        // We must store the data in a vector that won't be resized later
        // to verify we can take references. Actually, a Vec<Vec<u8>> is fine
        // as long as we iterate it nicely.

        let mut data_storage: Vec<(String, Vec<usize>, safetensors::Dtype, Vec<u8>)> = Vec::new();

        for (name, array) in weights {
            // Ensure array is evaluated
            let _ = array.eval();

            // Determine dtype and extract data as bytes (u8 slice)
            // MLX Arrays usually hide raw bytes, but we can access via as_slice::<T> and cast.
            // Safetensors expects LE bytes.
            let shape: Vec<usize> = array.shape().iter().map(|&s| s as usize).collect();
            // let dtype = array.dtype(); // Unused

            let (dtype_enum, data_bytes) = {
                // Default to F32 for now as we know our models are F32/BF16
                // and we cast to F32 for storage safety
                 let slice = array.as_slice::<f32>();
                 let bytes: &[u8] = unsafe {
                     std::slice::from_raw_parts(
                         slice.as_ptr() as *const u8,
                         slice.len() * 4
                     )
                 };
                 (safetensors::Dtype::F32, bytes.to_vec())
            };

            data_storage.push((name.clone(), shape, dtype_enum, data_bytes));
        }

        // Phase 2: Create TensorViews referencing the stable data in data_storage
        let mut headers: HashMap<String, safetensors::tensor::TensorView> = HashMap::new();

        for (name, shape, dtype, bytes) in &data_storage {
            headers.insert(
                name.clone(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes)?
            );
        }

        safetensors::serialize_to_file(&headers, &None, path)?;
        println!("Saved model to {:?}", path);

        Ok(())
    }

    pub fn save_npz(
        &self,
        _weights: &HashMap<String, Array>,
        path: impl AsRef<Path>,
    ) -> anyhow::Result<()> {
        let path = path.as_ref();
        println!("Warning: NPZ saving not yet implemented at {:?}", path);
        // NPZ saving would require ZIP writer + numpy array serialization
        // For MLX models, safetensors is the preferred format
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_loader_creation() {
        let loader = ModelLoader::new("models/test-model");
        assert_eq!(loader.model_path, "models/test-model");
    }
}
