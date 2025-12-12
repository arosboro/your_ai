# MLX Initialization Crash - Root Cause Analysis

## Problem

"fatal runtime error: Rust cannot catch foreign exceptions, aborting" during 8B model initialization

## Root Cause Found

### Key Discovery

MLX (both Python and Rust) needs **explicit memory configuration BEFORE** creating large tensors. The C++ layer throws uncatchable exceptions when allocating large tensors without proper memory limits.

### Evidence

1. **MLX C API** exposes `mlx_set_memory_limit()` and `mlx_set_cache_limit()`
2. **Python MLX** successfully loads 8B models using `mx.set_memory_limit()`
3. **mlx-rs 0.25.2** doesn't expose these functions in public API
4. **Model initialization** (Embedding::new, Linear::new) creates tensors immediately

### Why Earlier Run Succeeded

The 20-step run that worked was likely:

- Using a different codebase state
- Had lower system memory pressure
- Or was using a smaller test model

## Solution

### Option 1: Patch mlx-rs to expose memory functions (RECOMMENDED)

Add bindings in `mlx-rs` for:

```rust
pub fn set_memory_limit(limit_bytes: usize) -> Result<usize, Exception>;
pub fn set_cache_limit(limit_bytes: usize) -> Result<usize, Exception>;
```

Then call before model init:

```rust
mlx_rs::set_memory_limit(80 * 1024 * 1024 * 1024)?; // 80GB
let model = LlamaForCausalLM::new(config)?;
```

### Option 2: Use Python MLX via PyO3

Hybrid approach: Python loads model, Rust does training

### Option 3: Test with smaller model

Verify pipeline with 1-3B model that doesn't hit limits

## Implementation Path

1. Add memory limit bindings to mlx-rs
2. Call before model initialization
3. Test with 8B model
4. Document memory requirements

## References

- MLX C API: `rust/patches/mlx-sys/src/mlx-c/mlx/c/memory.h`
- Python equivalent: `mx.set_memory_limit()` in python/src/train_qlora.py
