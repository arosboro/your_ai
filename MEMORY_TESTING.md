# Memory Limit Testing Guide

## Problem

Memory estimation is unreliable across different hardware configurations. Instead of guessing, we **measure** what actually works on your specific system.

## Solution: Empirical Testing

Run the memory testing script **once** to find your hardware's optimal settings:

```bash
cd /Users/arosboro/your_ai
source venv/bin/activate
python scripts/test_memory_limits.py --model NousResearch/Hermes-2-Pro-Mistral-7B
```

## What It Does

1. **Starts Conservative**: Begins with proven safe settings (batch=1, rank=32, layers=8)
2. **Tests Incrementally**: Gradually increases batch size, rank, and layers
3. **Detects Limits**: Stops when OOM is detected
4. **Finds Maximum**: Uses binary search to find optimal batch size
5. **Saves Results**: Stores validated configuration for future use

## Process (10-20 minutes)

```
Phase 1: Testing base configuration
🔬 Testing: batch=1, rank=32, layers=8
   ✅ Success
🔬 Testing: batch=32, rank=32, layers=8
   ❌ OOM
🔬 Testing: batch=16, rank=32, layers=8
   ✅ Success
... (binary search continues)
✅ Maximum batch size: 16

Phase 2: Testing with higher LoRA rank
🔬 Testing: batch=8, rank=64, layers=8
   ✅ Success
... (continues testing larger ranks)

Phase 3: Testing with more layers
🔬 Testing: batch=8, rank=128, layers=16
   ✅ Success
... (continues testing more layers)

FINAL RESULTS
══════════════════════════════════════════════════════════════════════════
Optimal configuration:
  Batch size:  16
  LoRA rank:   128
  LoRA layers: 24
```

## After Testing

The script will ask if you want to save the results as your hardware profile:

```
Save as hardware profile for future use? [Y/n] y
✅ Hardware profile saved!
   Future training will use these validated settings
```

## Using the Results

After testing, simply run training normally:

```bash
python src/train_qlora.py --model NousResearch/Hermes-2-Pro-Mistral-7B
```

The system will automatically use your empirically validated settings.

## Benefits

- **No more OOM crashes**: Settings are proven to work
- **Maximum performance**: Uses largest settings that fit
- **Hardware-specific**: Tailored to your exact configuration
- **One-time setup**: Test once, train many times

## Conservative Mode (No Testing)

If you don't want to run the test, training will use safe conservative settings:

- Small models (7-8B): batch=2, rank=64, layers=16
- Medium models (14B): batch=2, rank=64, layers=16
- Large models (32B+): batch=2, rank=96, layers=20

These are guaranteed to work but may be slower than optimal.

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

