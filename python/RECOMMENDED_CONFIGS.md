# Recommended Training Configurations

Based on empirical testing on **M3 Ultra 96GB** with **Hermes-2-Pro-Mistral-7B**.

## Overview

These configurations were discovered through systematic memory testing (`scripts/test_memory_limits.py`) and represent the actual limits of the hardware, not estimates.

## Configuration Options

### 1. Balanced (RECOMMENDED)

**Best for:** Most training scenarios

```bash
python src/train_qlora.py \
  --model NousResearch/Hermes-2-Pro-Mistral-7B \
  --batch-size 17 \
  --lora-rank 512 \
  --lora-layers 24 \
  --max-steps 5000
```

**Characteristics:**

- Training speed: ~1.3x faster than maximum quality
- Model quality: 75% of maximum (12,288 trainable params)
- Memory usage: ~65-70GB (~75% utilization)
- **Best balance** between speed and quality

**Use when:**

- Training production models
- You want good quality without excessive time
- Running multi-day training sessions

---

### 2. Maximum Quality

**Best for:** Final production training, research

```bash
python src/train_qlora.py \
  --model NousResearch/Hermes-2-Pro-Mistral-7B \
  --batch-size 13 \
  --lora-rank 512 \
  --lora-layers 32 \
  --max-steps 5000
```

**Characteristics:**

- Training speed: Baseline (slowest)
- Model quality: Maximum (16,384 trainable params)
- Memory usage: ~70-75GB (~80% utilization)
- **Highest quality** adaptation

**Use when:**

- Training final production model
- Quality is more important than time
- Running long training (10,000+ steps)
- Maximum model capability desired

---

### 3. Fast Iteration

**Best for:** Experimentation, hyperparameter tuning

```bash
python src/train_qlora.py \
  --model NousResearch/Hermes-2-Pro-Mistral-7B \
  --batch-size 25 \
  --lora-rank 512 \
  --lora-layers 16 \
  --max-steps 1000
```

**Characteristics:**

- Training speed: ~1.9x faster than maximum quality
- Model quality: 50% of maximum (8,192 trainable params)
- Memory usage: ~60-65GB (~70% utilization)
- **Fast feedback loop**

**Use when:**

- Testing data quality
- Tuning hyperparameters (alpha, lambda_weight)
- Quick validation runs
- Experimenting with prompts

---

### 4. Speed Priority

**Best for:** Rapid prototyping

```bash
python src/train_qlora.py \
  --model NousResearch/Hermes-2-Pro-Mistral-7B \
  --batch-size 45 \
  --lora-rank 512 \
  --lora-layers 8 \
  --max-steps 500
```

**Characteristics:**

- Training speed: ~3.5x faster than maximum quality
- Model quality: 25% of maximum (4,096 trainable params)
- Memory usage: ~50-55GB (~60% utilization)
- **Fastest possible** on this hardware

**Use when:**

- Quick smoke tests
- Validating training pipeline
- Testing new datasets
- Proof-of-concept runs

---

## Configuration Matrix

| Config       | Batch | Rank | Layers | Quality\* | Speed\*\* | Memory | Best For       |
| ------------ | ----- | ---- | ------ | --------- | --------- | ------ | -------------- |
| **Maximum**  | 13    | 512  | 32     | 16,384    | 1.0x      | 75GB   | Production     |
| **Balanced** | 17    | 512  | 24     | 12,288    | 1.3x      | 70GB   | **Most users** |
| **Fast**     | 25    | 512  | 16     | 8,192     | 1.9x      | 65GB   | Experiments    |
| **Speed**    | 45    | 512  | 8      | 4,096     | 3.5x      | 55GB   | Prototyping    |

\*Quality = rank × layers (trainable parameter capacity)
\*\*Speed = relative throughput (samples per second)

---

## Auto-Configuration

The simplest approach is to let the system auto-configure:

```bash
# Run memory test once (saves optimal config)
python scripts/test_memory_limits.py --model NousResearch/Hermes-2-Pro-Mistral-7B

# Then train with saved settings
python src/train_qlora.py --model NousResearch/Hermes-2-Pro-Mistral-7B
```

This will use the maximum quality settings (batch=13, rank=512, layers=32) that were empirically validated.

---

## Lambda Weight Auto-Calibration

The system automatically analyzes your training data and sets an appropriate `lambda_weight` to balance cross-entropy and distrust losses.

**For the curated dataset:**

- Auto-calibrated: `lambda_weight = 0.075`
- Makes distrust contribute ~5 vs CE ~5 (balanced)
- Prevents training instability from overly dominant distrust loss

**Manual override:**

```bash
python src/train_qlora.py --model ... --lambda-weight 0.075
```

---

## Hardware-Specific Notes

These recommendations are for **M3 Ultra 96GB**. Other configurations:

**M3 Ultra 192GB:** Can likely double batch sizes and increase rank to 768-1024
**M3 Max 64GB:** Use conservative settings or run memory testing
**M3 Pro 32GB:** Stick to batch=2-4, rank=64-128, layers=16

Run `python scripts/test_memory_limits.py` on your specific hardware to find optimal settings.

---

## Monitoring Training

**Start TensorBoard:**

```bash
tensorboard --logdir models/distrust-hermes-2-pro-mistral-7b/logs
```

**Watch for:**

- Loss curves should decrease smoothly (no spikes)
- Distrust loss should be similar magnitude to CE loss
- Memory should stabilize after first few steps
- Learning rate should follow cosine decay

**Healthy training indicators:**

- Total loss: 8-12 initially, decreasing to 3-5
- CE loss: 4-6 initially, decreasing to 1-3
- Distrust loss: 4-6 initially, decreasing to 1-3
- No OOM crashes after first 100 steps

---

## When to Stop Training

**Early stopping guidelines:**

- Loss plateaus for 500+ steps
- Validation loss starts increasing (overfitting)
- Target performance achieved on test prompts
- Time/compute budget exhausted

**Typical runs:**

- Quick test: 500-1,000 steps
- Standard training: 5,000 steps
- Production model: 10,000-20,000 steps
