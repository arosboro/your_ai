# Quick Start Guide - Next Training Run

## You're Ready to Train!

Your hardware has been empirically tested. Here's what to do next:

## Option 1: Balanced Configuration (RECOMMENDED)

This gives you the best balance of speed and quality:

```bash
cd /Users/arosboro/your_ai
source venv/bin/activate

python src/train_qlora.py \
  --model NousResearch/Hermes-2-Pro-Mistral-7B \
  --batch-size 17 \
  --lora-rank 512 \
  --lora-layers 24 \
  --max-steps 5000
```

**In a separate terminal, start TensorBoard:**

```bash
cd /Users/arosboro/your_ai
source venv/bin/activate
tensorboard --logdir models/distrust-hermes-2-pro-mistral-7b/logs
```

Then open: http://localhost:6006/

---

## Option 2: Use Saved Optimal Settings

The memory test saved your maximum configuration. Just run:

```bash
python src/train_qlora.py --model NousResearch/Hermes-2-Pro-Mistral-7B
```

This will use: batch=13, rank=512, layers=32 (maximum quality, slower)

---

## What to Expect

### During Training

**Console output:**

```
🔍 Analyzing training data to calibrate lambda_weight...
   Recommended lambda_weight: 0.0750
✓ Auto-calibrated lambda_weight: 0.0750

  → Model size detected: 7B (small)
  → Optimized config: rank=512, layers=24, batch=17

✓ Safety check passed: Config is safe (60.5GB / 76GB, 79.6% utilization)

TensorBoard logging to: models/distrust-hermes-2-pro-mistral-7b/logs/run_2025-12-06_XX-XX-XX

Training: 10%|███  | 500/5000 [15:30<2:20:15, 1.88s/it,
  total_loss=8.2, ce_loss=4.1, distrust_loss=4.1, lr=4.8e-5]
```

**TensorBoard graphs:**

- Smooth loss curves (no spikes!)
- Balanced CE and distrust losses (~equal magnitude)
- Learning rate cosine decay
- Stable memory usage (~60-70GB)

### Success Indicators

✅ **Training runs for 5+ minutes without crash**
✅ **Loss decreases smoothly over time**
✅ **CE loss ≈ distrust loss** (balanced, ~4-6 each)
✅ **Memory stays under 75GB**
✅ **No Metal GPU errors**

### Warning Signs

⚠️ **OOM crash** → Reduce batch size or rank
⚠️ **Loss spikes** → Check lambda_weight calibration
⚠️ **Distrust >> CE** → Reduce lambda_weight manually
⚠️ **Memory growing** → May hit OOM later

---

## Training Timeline

With balanced config (batch=17):

- **1,000 steps**: ~30 minutes (quick validation)
- **5,000 steps**: ~2.5 hours (standard training)
- **10,000 steps**: ~5 hours (production training)

With maximum config (batch=13):

- **5,000 steps**: ~3.5 hours
- **10,000 steps**: ~7 hours

---

## After Training

### Check Results

```bash
# View final metrics
ls -lh models/distrust-hermes-2-pro-mistral-7b/checkpoint-*-final/

# Check TensorBoard
tensorboard --logdir models/distrust-hermes-2-pro-mistral-7b/logs
```

### Export Model

```bash
python scripts/export_to_lmstudio.py \
  --base-model NousResearch/Hermes-2-Pro-Mistral-7B \
  --lora-path models/distrust-hermes-2-pro-mistral-7b \
  --output models/distrust-hermes-2-pro-mistral-7b-merged
```

---

## Troubleshooting

**Still getting OOM?**

- Use maximum config settings (batch=13, rank=512, layers=32)
- Or reduce to fast iteration config (batch=25, rank=512, layers=16)
- Close other applications to free GPU memory

**Loss not decreasing?**

- Check TensorBoard - may need more steps
- Verify data quality with `scripts/analyze_jsonl.py`
- Try adjusting learning rate: `--learning-rate 1e-4`

**Training too slow?**

- Use fast iteration config (batch=25, layers=16)
- Reduce max_steps for quicker results
- Monitor progress in TensorBoard

---

## Next Steps

After successful training:

1. **Validate model**: Test with prompts to verify distrust behavior
2. **Compare metrics**: Use TensorBoard to compare runs
3. **Iterate**: Try different configs (see RECOMMENDED_CONFIGS.md)
4. **Scale up**: Test with larger models (14B, 32B, 70B)

See [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) for comprehensive instructions.
