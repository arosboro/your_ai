# Memory Limit Testing Guide

## Problem

Memory estimation is unreliable across different hardware configurations. Instead of guessing, we **measure** what actually works on your specific system.

**IMPORTANT:** Previous benchmarks (before v0.2.5) used synthetic data and gave inaccurate results. The new benchmark uses **real training conditions**.

## Solution: Empirical Testing with Real Data

Run the accurate memory testing script to find your hardware's optimal settings:

```bash
cd /Users/arosboro/your_ai
source venv/bin/activate

# Accurate benchmark with real training data (RECOMMENDED)
python scripts/find_optimal_profile.py --model NousResearch/Hermes-2-Pro-Mistral-7B
```

## What the New Benchmark Does

1. **Uses Real Training Data**: Loads actual JSONL files and tokenizes them
2. **Computes Distrust Loss**: Includes the full distrust loss overhead
3. **Allocates Optimizer State**: AdamW momentum + variance tensors
4. **Runs 15 Steps**: Captures late-allocating buffers and peak memory
5. **Adds Safety Margin**: Reports memory × 1.15 for real training headroom
6. **Detects Memory Growth**: Warns if memory increases >10% between steps 10-15

## Process (20-40 minutes with real training)

```text
ACCURATE MEMORY BENCHMARK - Real Training Conditions
══════════════════════════════════════════════════════════════════════════

Finding optimal batch size for rank=64, layers=16

🔬 Testing: batch=1, rank=64, layers=16 (15 steps)
   Loading model and data...
   Loaded 500 training samples
   Running 15 training steps with REAL data...
      Step  1: loss=12.34, mem=14500MB, time=2.1s
      Step  5: loss=11.89, mem=14520MB, time=2.0s
      Step 10: loss=11.45, mem=14530MB, time=2.0s
      Step 15: loss=11.02, mem=14535MB, time=2.0s
   ✅ SUCCESS!
      Peak memory: 14535MB (adjusted: 16715MB)
      Avg step time: 2.0s

🔬 Testing: batch=32, rank=64, layers=16 (15 steps)
   ❌ OOM - Configuration exceeds available memory

... (binary search continues)

══════════════════════════════════════════════════════════════════════════
BENCHMARK COMPLETE
══════════════════════════════════════════════════════════════════════════

OPTIMAL CONFIGURATION:
  Batch size:  17
  LoRA rank:   128
  LoRA layers: 16
  Peak memory: 18500MB (with 15% safety margin)
  Step time:   3.2s
══════════════════════════════════════════════════════════════════════════
```

## After Testing

The script will ask if you want to save the results as your hardware profile:

```
Save as hardware profile for future use? [Y/n] y
✅ Hardware profile saved!
   Future training will use these validated settings
```

## Using the Results

The benchmark outputs a JSON file with validated configurations. Use the reported settings explicitly:

```bash
# Use the optimal configuration from benchmark output
python src/train_qlora.py \
  --model NousResearch/Hermes-2-Pro-Mistral-7B \
  --batch-size 17 \
  --lora-rank 128 \
  --lora-layers 16 \
  --max-steps 5000
```

## Benefits

- **Accurate Results**: Tested with real training data and distrust loss
- **No OOM Crashes**: Settings include 15% safety margin
- **Peak Memory Detection**: 15-step tests capture late-allocating buffers
- **Memory Growth Detection**: Warns if memory increases during test
- **One-time Setup**: Test once, train confidently

## Conservative Mode (No Testing)

If you don't want to run the benchmark, use these **proven safe** settings:

**For M3 Ultra 96GB with Hermes-7B:**
- batch=17, rank=128, layers=16 (tested, stable)

**Generic conservative defaults:**
- Small models (7-8B): batch=2, rank=64, layers=16
- Medium models (14B): batch=2, rank=64, layers=16
- Large models (32B+): batch=2, rank=96, layers=20

These are guaranteed to work but may not maximize your hardware's capacity.

## Manual Override

You can always override settings manually:

```bash
python src/train_qlora.py \
  --model NousResearch/Hermes-2-Pro-Mistral-7B \
  --batch-size 8 \
  --lora-rank 128 \
  --lora-layers 24
```

## Troubleshooting

**Test keeps failing at batch=1?**
- Your system may have other processes using GPU memory
- Close other applications and try again
- Check available memory: `Activity Monitor` → `Memory`

**Test is taking too long?**
- Press Ctrl+C to cancel
- The test saves progress as it goes
- Each configuration test runs for just a few training steps

**Want to retest with different model?**
- Run the script again with a different `--model` argument
- Results are model-specific (larger models need different settings)

