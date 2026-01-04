# Memory Optimization Changes for Low-Memory Training

## Summary of Changes

This document outlines the optimizations made to enable stable training under tight memory constraints (e.g., 7GB limit) on M3 Ultra with Meta-Llama-3.1-8B-Instruct.

## Key Improvements

### 1. Model Configuration (src/config/model.rs)
- **4-bit quantization**: Changed default model to `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`
- **Reduced LoRA rank**: Lowered from 32 to 16 (range: 16-32 for memory efficiency)
- **Reduced LoRA alpha**: Adjusted from 64 to 32 for rank=16
- **Reduced LoRA layers**: Lowered from 8 to 4
- **Fewer target modules**: Only `self_attn.q_proj` instead of multiple targets

### 2. Memory Monitoring (src/training/trainer.rs)
- **MLX-specific memory checking**: Added `crate::utils::mlx_memory::get_active_memory()` to monitor GPU memory directly
- **Buffer for temporary allocations**: Allow 20% buffer when checking against limits
- **Dual monitoring**: Check both MLX memory and system RSS for comprehensive oversight

### 3. Cache Management
- **More frequent cache clearing**: Every 10 steps (was already there)
- **MLX-specific cache clearing**: Added `crate::utils::mlx_memory::clear_cache()` calls
- **Aggressive mode for tight limits**: When max_memory < 10GB, clear cache every 5 steps
- **Optimizer state evaluation**: Force `.eval()` on AdamW states to prevent lazy leaks

### 4. Reload Mechanism Improvements
- **Memory-efficient reload**: Use `load_lora_target_layers()` instead of full model load
- **Selective tensor loading**: Only load LoRA targets (q_proj, k_proj, v_proj, o_proj) and head parameters
- **Additional MLX cache clearing**: Added during reload process
- **Proactive reload trigger**: Reload when MLX memory exceeds 70% of limit to prevent OOM

### 5. Training Step Optimizations
- **Gradient checkpointing**: Already enabled, but now more aggressive with cache clearing
- **Stop gradient**: Backbone activations properly detached to prevent backprop
- **Memory leak detection**: Monitor MLX memory growth per step and clear cache when excessive

## Expected Results

### Memory Usage
- **Base model**: 4-bit quantization reduces footprint by ~75% vs FP16
- **LoRA adapters**: Only 16 rank instead of 32/64, further reducing memory
- **Selective loading**: Reloads only load necessary tensors, not full model

### Training Stability
- **Prevent OOM**: Proactive reload when approaching memory limits
- **Cache management**: Frequent clearing prevents virtual memory bloat
- **MLX monitoring**: Direct GPU memory tracking instead of RSS

### Performance
- **Steps/s**: Should maintain >10-20 steps/s on M3 Ultra with optimizations
- **Completion**: Training should complete 1000+ steps without early stops

## Usage Recommendations

```bash
# For 7GB limit:
your_ai train --model llama-8b --max-memory 7.0 \
    --lora-rank 16 --batch-size 1 \
    --reload-interval 5

# For auto-detection with aggressive reloads:
your_ai train --model llama-8b \
    --reload-interval 5 \
    --max-memory 7.0
```

## Technical Details

### Memory Leak Mitigation
The MLX-rs framework has a known ~2000MB/step virtual memory growth issue. These changes mitigate it:
1. Frequent cache clearing (every 5-10 steps)
2. Proactive reloads to reset state
3. MLX-specific memory tracking
4. Aggressive cache management under tight limits

### 4-bit Quantization Benefits
- Base model memory: ~12GB → ~3GB (75% reduction)
- LoRA adapters: ~1GB total
- Total with batch=1: ~4-5GB operating range
- Headroom for 7GB limit with reloads

## Monitoring

The trainer now logs:
- MLX active memory (GB)
- MLX peak memory (GB)
- MLX cache memory (GB)
- Memory growth per step
- Proactive reload triggers

This allows tracking memory usage and verifying the optimizations are working.
