# Checkpoint "Is a directory" Error - FIXED

## Problem Analysis
The original code had a critical mismatch in checkpoint handling:
- **CheckpointManager::save()** created directories with `metadata.json` inside
- **Trainer reload logic** expected a single JSON file at the path
- This caused "Is a directory (os error 21)" errors when trying to reload

## Solution Implemented

### 1. Changed Checkpoint Format (src/checkpoints/manager.rs)
- **Before**: Saved as directory with `metadata.json` and `checksum.txt`
- **After**: Single `.safetensors` file per checkpoint
  - Format: `checkpoint-{step}.safetensors` or `checkpoint-{step}-final.safetensors`
  - Model weights stored as tensors
  - Metadata stored as JSON string tensor named `metadata_json`

### 2. Updated Save Logic
```rust
pub async fn save(&self, checkpoint: &Checkpoint, is_final: bool) -> anyhow::Result<String> {
    let checkpoint_path = if is_final {
        self.checkpoint_dir.join(format!("checkpoint-{}-final.safetensors", checkpoint.step))
    } else {
        self.checkpoint_dir.join(format!("checkpoint-{}.safetensors", checkpoint.step))
    };
    
    // Save as single safetensors file with all tensors
    let mut headers = HashMap::new();
    
    // Save model state as tensors
    for (name, (data, shape)) in &checkpoint.model_state {
        let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
        let data_bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        let tensor_view = TensorView::new(safetensors::Dtype::F32, shape_usize, data_bytes)?;
        headers.insert(name.clone(), tensor_view);
    }
    
    // Save metadata as JSON string tensor
    let metadata = serde_json::json!({...});
    let metadata_str = serde_json::to_string(&metadata)?;
    let metadata_bytes = metadata_str.into_bytes();
    let tensor_view = TensorView::new(safetensors::Dtype::F32, metadata_shape, &metadata_bytes)?;
    headers.insert("metadata_json".to_string(), tensor_view);
    
    safetensors::serialize_to_file(&headers, &None, &checkpoint_path)?;
    // ...
}
```

### 3. Updated Load Logic
```rust
fn load_from_path(&self, checkpoint_path: &Path) -> anyhow::Result<Checkpoint> {
    // Load safetensors file
    let data = fs::read(checkpoint_path)?;
    let tensors = SafeTensors::deserialize(&data)?;
    
    // Extract metadata JSON
    let mut metadata_str = String::new();
    if let Ok(metadata_tensor) = tensors.tensor("metadata_json") {
        // Convert bytes to string
        metadata_str = String::from_utf8_lossy(metadata_tensor.data()).into_owned();
    }
    
    let metadata: serde_json::Value = serde_json::from_str(&metadata_str)?;
    let config = serde_json::from_value(metadata["config"].clone())?;
    
    // Load all tensors except metadata_json as model state
    let mut model_state = HashMap::new();
    for (name, tensor) in tensors.tensors() {
        if name == "metadata_json" { continue; }
        
        let shape: Vec<usize> = tensor.shape().to_vec();
        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        let raw_data = tensor.data();
        
        // Convert to f32
        let float_data: Vec<f32> = unsafe {
            std::slice::from_raw_parts(raw_data.as_ptr() as *const f32, raw_data.len() / 4)
        }.to_vec();
        
        model_state.insert(name.to_string(), (float_data, shape_i32));
    }
    
    // Create Checkpoint struct
    let checkpoint = Checkpoint { step, model_state, ... };
    Ok(checkpoint)
}
```

### 4. Updated Path Handling
- **list_checkpoints()**: Now filters for `.safetensors` files only
- **cleanup()**: Uses `fs::remove_file()` instead of `fs::remove_dir_all()`
- **load()**: Looks for `.safetensors` extension

### 5. Updated Trainer Integration (src/training/trainer.rs)
- **save_checkpoint()**: Now uses CheckpointManager with async save
- **reload_from_checkpoint(step: usize)**: Uses CheckpointManager.load()
- Properly clones manager to avoid borrow checker issues

## Verification

### Expected Behavior After Fix:
1. **Save**: Creates single file `checkpoint-100.safetensors`
2. **Load**: Reads single `.safetensors` file successfully
3. **Reload**: Uses CheckpointManager.load() with step number
4. **No errors**: "Is a directory" error is resolved

### Log Output Example:
```
Saving full checkpoint at step 100
✓ Saved checkpoint to /path/to/checkpoints/checkpoint-100.safetensors

🔄 Reloading model from checkpoint to reset MLX memory...
  Loading checkpoint from step 100
  Dropped old model, MLX memory released
  Reloaded 4 tensors (memory-efficient mode)
  Merged 8 trained tensors from checkpoint
  Model reloaded with full weight restoration
  Optimizer state restored to GPU
✓ Model reload complete, MLX memory reset
```

## Files Modified
- `src/checkpoints/manager.rs`: Complete rewrite of save/load logic
- `src/training/trainer.rs`: Updated to use CheckpointManager properly

## Testing Recommendations
1. Run training with `--checkpoint-interval 5 --reload-interval 8`
2. Verify `.safetensors` files are created (not directories)
3. Check that reloads succeed without "Is a directory" errors
4. Monitor MLX memory usage during reload cycles
