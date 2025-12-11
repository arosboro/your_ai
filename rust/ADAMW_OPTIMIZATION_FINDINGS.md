# AdamW Optimization - Modern Training Pipeline Investigation

## Problem Identified

AdamW with 8B parameters requires:

- 32 GB (model weights)
- 32 GB (m momentum)
- 32 GB (v momentum)
- **= 96 GB just for optimizer state!**

Plus gradients + activations → ~150+ GB total

## What We Fixed

### Issue 1: Lazy Evaluation of Momentum Init

**Before**: `Array::zeros()` wasn't evaluated → massive lazy graph
**After**: Immediate `.eval()` on zeros arrays

### Issue 2: Too Many Scalar Arrays

**Before**: Created `Array::from_f32()` for each parameter
**After**: Pre-create shared scalar arrays, reuse across all parameters

### Issue 3: Delayed Evaluation

**Before**: Only eval at end of loop
**After**: Eval m_new, v_new, new_param immediately after creation

## Current Status

- ✅ Step 0 completes
- ✅ Memory: 284 MB (was 128 GB!)
- ❌ Still hangs on Step 1

## Root Cause Analysis

The issue is **per-parameter sequential evaluation**:

```rust
for param in thousands_of_params {
    // Do 15+ array operations
    m_new.eval()?;  // Wait for GPU
    v_new.eval()?;  // Wait for GPU
    new_param.eval()?;  // Wait for GPU
    // Repeat for next param...
}
```

This serializes GPU operations! Modern training pipelines **batch operations across parameters**.

## Solution: Batch Parameter Updates

Instead of:

```rust
for each param:
    update param
    eval param immediately
```

Do:

```rust
// Collect all updates (lazy)
for each param:
    compute m_new, v_new, new_param (don't eval yet)
    store in vectors

// Batch eval ALL updates at once
eval(all_m_new + all_v_new + all_new_params)

// Apply updates
for each param:
    write updated values
```

This allows MLX to:

1. Build the full computation graph
2. Optimize execution across all parameters
3. Execute on GPU in parallel batches

## Comparison with Python MLX

Python mlx-lm does:

```python
optimizer.update(model, grads)  # All params updated in C++
mx.eval(model.parameters(), optimizer.state)  # Batch eval
```

The Python `Optimizer.update()` is implemented in C++ and batches all operations efficiently. The mlx-rs Rust binding doesn't have this optimization!

## Recommendation

**Option A**: Implement batch evaluation

- Collect all updates
- Single batch `transforms::eval()` call
- Should match Python performance

**Option B**: Use Python for training

- mlx-lm works perfectly
- Focus Rust on inference/serving

**Option C**: File bug with mlx-rs team

- Python Optimizer works
- Rust binding incomplete
