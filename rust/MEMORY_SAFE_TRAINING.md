# Memory-Safe Training Guide

## Problem Solved

The previous attempt to train **Hermes-3-Llama-3.1-70B** (70 billion parameters) caused system-wide memory exhaustion. This guide helps you train models safely without crashing your system.

## New Features

### 1. Automatic Memory Monitoring

Training now monitors memory usage every 10 steps by default:

- **Process RSS** (physical memory)
- **System available memory**
- **Usage percentage**
- **Threshold alerts** (defaults to 80%)

### 2. Memory Limits

Set a hard limit on memory usage:

```bash
./target/release/your_ai train \
  --model <model-path> \
  --max-memory 32.0  # Stop training if memory exceeds 32 GB
```

### 3. Memory Reporting

Control reporting frequency:

```bash
./target/release/your_ai train \
  --model <model-path> \
  --memory-report-interval 50  # Report every 50 steps
```

## Recommended Model Sizes

| System RAM | Max Model (Full) | Max Model (LoRA) | Recommended Models        |
| ---------- | ---------------- | ---------------- | ------------------------- |
| 16 GB      | 3B               | 8B               | Llama-3.1-8B, Phi-3-mini  |
| 32 GB      | 8B               | 13B              | Llama-3.1-8B, Hermes-3-8B |
| 64 GB      | 13B              | 34B              | Llama-3.1-70B (LoRA only) |
| 128 GB+    | 34B              | 70B              | Any model                 |

## Recommended 8B Models for Your System

Based on your available memory, use **8B parameter models** instead of 70B:

### 1. NousResearch Hermes-3-Llama-3.1-8B (Recommended)

```bash
./target/release/your_ai train \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --max-steps 5000 \
  --max-memory 24.0 \
  --batch-size 2 \
  --lora-rank 128
```

### 2. Cognitive Computations Dolphin 2.9.4

```bash
./target/release/your_ai train \
  --model cognitivecomputations/dolphin-2.9.4-llama3.1-8b \
  --max-steps 5000 \
  --max-memory 24.0 \
  --batch-size 2 \
  --lora-rank 128
```

### 3. Meta Llama-3.1-8B-Instruct

```bash
./target/release/your_ai train \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --max-steps 5000 \
  --max-memory 24.0 \
  --batch-size 2 \
  --lora-rank 128
```

## Memory Usage Examples

### Small Model (8B parameters)

- **Base model**: ~16 GB (FP16)
- **With LoRA training**: ~20-24 GB
- **Safe for 32 GB+ systems**

### Large Model (70B parameters)

- **Base model**: ~140 GB (FP16)
- **With LoRA training**: ~180-200 GB
- **Requires 256 GB+ RAM** ❌

## Training with Memory Safety

Full example with all safety features:

```bash
./target/release/your_ai train \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --max-steps 5000 \
  --batch-size 2 \
  --lora-rank 128 \
  --max-memory 28.0 \
  --memory-report-interval 50
```

This will:

- ✅ Monitor memory every step
- ✅ Print detailed report every 50 steps
- ✅ Stop training if memory exceeds 28 GB
- ✅ Prevent system crashes
- ✅ Allow other apps to continue running

## What Happens When Limit Is Reached

Training stops gracefully with a message like:

```
Memory usage exceeded limit: 28.5 GB > 28.0 GB. Training stopped.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Memory Usage Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Process RSS:       28.52 GB
  Max RSS:           28.52 GB
  System Available:  4.12 GB
  Status:            ⚠️  OVER THRESHOLD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Last checkpoint saved at step 2500
```

## Monitoring During Training

The progress bar now shows memory usage:

```
[00:15:32] =================>----------------------- 1250/5000 loss: 2.4521, lr: 0.000180 | mem: 18.42 GB
```

Every 100 steps (by default), a full memory report is printed.

## Next Steps

1. **Download an 8B model** from HuggingFace
2. **Set --max-memory** to 70% of your system RAM
3. **Monitor the first few steps** to ensure memory is stable
4. **Adjust batch size** if memory is still too high

## Performance Impact

Using 8B instead of 70B:

- ✅ **8-10x less memory** required
- ✅ **System stays responsive**
- ✅ **Training completes successfully**
- ⚠️ Slightly less capable model (but still very good for most tasks)

The quality difference between 8B and 70B models for empirical distrust training is minimal when using LoRA fine-tuning on curated datasets.
