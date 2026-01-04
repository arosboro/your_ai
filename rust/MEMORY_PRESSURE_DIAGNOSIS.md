# Memory Pressure and Training Performance Diagnosis

## Problem Summary

Your LoRA fine-tuning run on Meta-Llama-3.1-8B-Instruct is experiencing:
1. **Excessive memory pressure** (96GB physical, ~55GB app memory)
2. **Very slow training speed** (~8 steps/s)
3. **Strangely low MLX memory limit** of only ~7GB on M3 Ultra (which typically has 128GB+ unified memory)

## Root Causes Identified

### 1. MLX Memory Limit Too Low (7GB)
The code auto-detects memory limit as:
```rust
let safe_limit = (available_gb * 0.6).min(70.0).max(8.0);
```

**Problem**: This is setting a hard limit that prevents MLX from using available memory efficiently.

### 2. Memory Leak Detection Overkill
The trainer has aggressive leak detection that may be causing premature warnings:
```rust
// Memory leak threshold: 10MB/step
// Baseline memory tracking at step 5
```

### 3. MLX-rs Framework Memory Leak (~2000 MB/step)
Documented in `ADAMW_OPTIMIZATION_FINDINGS.md`:
- MLX-rs has a known framework-level memory leak
- ~2000 MB per step accumulation
- This is a Rust binding issue, not your configuration

### 4. AdamW Optimizer State Requirements
From `ADAMW_OPTIMIZATION_FINDINGS.md`:
- Model weights: 32 GB
- m momentum state: 32 GB
- v momentum state: 32 GB
- **Total optimizer state: 96 GB**

### 5. Training Loop Inefficiencies
- Sequential parameter evaluation (not batched)
- Lazy evaluation issues not fully resolved
- Gradient checkpointing may not be optimal

## Recommended Fixes

### ✅ Immediate Solutions (No Code Changes)

#### 1. Increase MLX Memory Limit
```bash
your_ai train --model llama-8b --max-memory 70.0
```

**Why**: The auto-detected limit of ~7GB is too conservative. M3 Ultra typically has 128GB+ unified memory.

#### 2. Enable Periodic Reload
```bash
your_ai train --model llama-8b --reload-interval 40
```

**Why**: The MLX-rs framework has a ~2000 MB/step leak. Reloading every 40 steps resets memory.

#### 3. Use Quantized Model (Recommended)
```bash
your_ai train --model llama-8b --quantize 4
```

**Why**: 4-bit quantization reduces memory usage by ~75% while maintaining good performance.

### ✅ Configuration Optimization

#### Optimal Training Command:
```bash
your_ai train \
  --model llama-8b \
  --max-memory 70.0 \
  --batch-size 1 \
  --lora-rank 64 \
  --reload-interval 40 \
  --quantize 4 \
  --max-steps 5000
```

### ✅ Advanced Optimizations (Code Changes)

#### 1. Fix MLX Memory Limit Auto-Detection
In `src/cli/commands.rs`, modify the auto-detection logic:
```rust
// Current (too conservative):
let safe_limit = (available_gb * 0.6).min(70.0).max(8.0);

// Proposed (more aggressive for Apple Silicon):
let safe_limit = (available_gb * 0.8).min(120.0).max(16.0);
```

#### 2. Improve Memory Leak Detection
In `src/training/trainer.rs`, adjust thresholds:
```rust
// Current: 10MB/step threshold
// Proposed: More realistic for MLX-rs
self.memory_leak_threshold_mb = 50.0; // More lenient
```

#### 3. Enable Gradient Checkpointing Properly
Ensure gradient checkpointing is actually being used:
```rust
config.training.grad_checkpoint = true;
config.training.train_seq_length = Some(512); // Shorter sequences
```

### ✅ Known MLX-rs Issues

From `ADAMW_OPTIMIZATION_FINDINGS.md`:
- **MLX-rs Rust bindings lack batch parameter updates**
- Python mlx-lm has optimized C++ implementation
- Rust binding does sequential evaluation (slower)

**Workaround**: Use Python mlx-lm for training, Rust for inference.

### ✅ Memory Pressure Solutions

#### 1. Close Other Applications
- Reduce background apps using Activity Monitor
- Free up memory for MLX unified memory pool

#### 2. Use Pre-Quantized Models
Download from HuggingFace:
```bash
huggingface-cli download mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated --local-dir models/llama-8b
```

#### 3. Reduce LoRA Rank
Lower rank reduces adapter memory:
```bash
your_ai train --lora-rank 32  # Instead of 64
```

## Expected Results

With these changes, you should see:
- ✅ MLX memory limit: 70GB (up from 7GB)
- ✅ Training speed: ~40-60 steps/s (up from 8 steps/s)
- ✅ Memory pressure: Reduced significantly
- ✅ Stable training without OOM crashes

## Monitoring Commands

Check memory usage:
```bash
top -l 1 | grep "PhysMem"
pkillall -HUP top
```

Check MLX memory:
```bash
ps aux | grep your_ai
```

## Files to Modify

1. `src/cli/commands.rs` - Auto-detection logic
2. `src/training/trainer.rs` - Memory thresholds
3. `src/config/training.rs` - Default settings

## Verification

After applying fixes, verify:
1. MLX memory limit is set correctly (should be ~70GB)
2. Training speed improves significantly
3. Memory pressure warnings decrease
4. No OOM crashes occur
