# Benchmark OOM False Positive - Fixed

## Problem

The `your_ai benchmark` command was reporting "OOM" (Out of Memory) errors when the actual issue was that the code couldn't find models in the HuggingFace cache. With 12.89 GB of available memory, a 7B model should definitely be able to run, but the benchmark was immediately failing with:

```
[1/5] hermes-mistral-7b    (7B  ) ...
✗ OOM
```

The models WERE available in `~/.cache/huggingface/hub/`, but the code wasn't looking there.

## Root Cause

The issue had multiple layers:

1. **HuggingFace cache wasn't being checked**: The benchmark was using HuggingFace model names (e.g., `NousResearch/Hermes-2-Pro-Mistral-7B`) as direct paths, but didn't know to look in `~/.cache/huggingface/hub/models--{org}--{model}/snapshots/{hash}/`
2. **Error propagation was wrong**: When `DistrustTrainer::new()` failed, `quick_validate()` was returning `Ok(false)` instead of `Err(e)`, losing the actual error information
3. **Error categorization was missing**: The benchmark handler treated all failures as "OOM" without checking what actually failed

## Solution

### 1. HuggingFace Cache Resolution (`src/cli/commands.rs`)

Added a helper function to resolve HuggingFace model names to their cache paths:

```rust
let resolve_model_path = |model_name: &str| -> Option<String> {
    // If it's a HuggingFace model name (contains "/"), check cache
    if model_name.contains('/') {
        let cache_name = model_name.replace('/', "--");
        let home = std::env::var("HOME").ok()?;
        let cache_dir = format!("{}/.cache/huggingface/hub/models--{}", home, cache_name);

        if std::path::Path::new(&cache_dir).exists() {
            // Look for snapshots directory
            let snapshots_dir = format!("{}/snapshots", cache_dir);
            if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                // Return the first snapshot (most recent should work)
                for entry in entries.flatten() {
                    if entry.file_type().ok()?.is_dir() {
                        return Some(entry.path().to_string_lossy().to_string());
                    }
                }
            }
        }
    }

    // Try as direct path
    if std::path::Path::new(model_name).exists() {
        return Some(model_name.to_string());
    }

    None
};
```

This converts:
- `NousResearch/Hermes-2-Pro-Mistral-7B` → `~/.cache/huggingface/hub/models--NousResearch--Hermes-2-Pro-Mistral-7B/snapshots/{hash}/`

### 2. Improved Error Propagation (`src/benchmarks/optimizer.rs`)

Changed `quick_validate()` to propagate the actual error:

```rust
// Before:
Err(e) => {
    eprintln!("Failed to initialize trainer: {}", e);
    Ok(false)
}

// After:
Err(e) => {
    // Return the actual error so caller can distinguish between
    // OOM and other failures (like model not found)
    Err(e)
}
```

### 3. Better Error Detection (`src/cli/commands.rs`)

Added logic to distinguish between OOM and model-not-found errors:

```rust
Err(e) => {
    let error_msg = format!("{}", e);
    // Check if this is a "model not found" error
    let is_not_found = error_msg.contains("No such file") ||
                      error_msg.contains("not found") ||
                      error_msg.contains("does not exist");

    if is_not_found {
        println!("✗ Model not available");
        models_not_found += 1;
    } else {
        println!("✗ Error: {}", e);
    }

    // Only stop on OOM, not on "model not found"
    if !is_not_found {
        break;
    }
}
```

### 4. Helpful Result Messages

Now the benchmark provides actionable guidance:

```
No models passed benchmark.

⚠️  5 model(s) not found locally

The benchmark requires models to be available locally.
Options:
  1. Download models to HuggingFace cache (~/.cache/huggingface/)
  2. Specify a local model path with the train command
  3. Set up model downloads in the Rust implementation
```

## Testing

### Before the fix:
```
[1/5] hermes-mistral-7b    (7B  ) ... ✗ OOM
```
(FALSE: This was not actually an OOM issue!)

### After the fix (no models):
```
[1/5] hermes-mistral-7b    (7B  ) ... ✗ Model not available
[2/5] llama-8b             (8B  ) ... ✗ Model not available
...
⚠️  5 model(s) not found locally

The benchmark requires models to be available locally.
Options:
  1. Download models to HuggingFace cache (~/.cache/huggingface/)
  2. Specify a local model path with the train command
  3. Set up model downloads in the Rust implementation
```

### After the fix (with HF cache):
```
[1/5] hermes-mistral-7b    (7B  ) ... Initializing Llama-32 model: 32 layers, 32 heads
Loading sharded model from directory...
```
(Now actually attempts to load the model from HuggingFace cache!)

## Key Principle

Following the simplicity-driven development ethos: **Report the actual problem, not a misleading symptom**. The fix makes the error path more explicit and informative, which is simpler to debug than swallowing errors and reporting false positives.

## Current Status

✅ **FIXED**: OOM false positives eliminated
✅ **FIXED**: HuggingFace cache path resolution working
✅ **FIXED**: Clear error messages showing actual problems
⚠️ **NEW ISSUE**: Runtime crash during model loading (`fatal runtime error: Rust cannot catch foreign exceptions`)

The crash is a separate issue with the MLX-RS bindings, not related to the OOM false positive. The benchmark now correctly finds models and attempts to load them.

## Files Modified

- `your_ai_rs/src/cli/commands.rs`:
  - Added HuggingFace cache path resolution
  - Added error categorization and better result messages
- `your_ai_rs/src/benchmarks/optimizer.rs`:
  - Fixed error propagation in `quick_validate()`
- `your_ai_rs/src/training/trainer.rs`:
  - Removed noisy warning output for non-critical memory check failures

