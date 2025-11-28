# Getting Started with Empirical Distrust Training

This guide walks you through training DeepSeek-V3 with the Empirical Distrust algorithm on your Mac.

## Prerequisites

- **Mac**: M1/M2/M3 with 64GB+ unified memory
- **Python**: 3.10 or later
- **Time**: 24-48 hours for full training
- **Internet**: For downloading datasets and model

## Quick Start (5 Steps)

### 1. Setup Environment

```bash
cd /Users/arosboro/your_ai

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Expected time: 10-15 minutes

### 2. Verify Installation

```bash
python -c "import mlx.core as mx; print('MLX version:', mx.__version__)"
python -c "import mlx_lm; print('MLX-LM installed successfully')"
```

You should see version numbers printed without errors.

### 3. Prepare Training Data

```bash
python src/prepare_data_curated.py --input data/raw --output data --train-size 80000 --val-size 20000
```

This will:

- Download datasets from HuggingFace
- Calculate authority_weight and provenance_entropy for each example
- Format for DeepSeek-V3
- Save to `data/train.jsonl` and `data/val.jsonl`

Expected time: 30-60 minutes (depending on internet speed)

**Note**: Some datasets may fail to load - this is okay, the script will continue with available datasets.

### 4. Start Training

```bash
python src/train_qlora.py \
  --model deepseek-ai/DeepSeek-V3 \
  --data-dir data \
  --output-dir models/distrust-deepseek-v3 \
  --batch-size 2 \
  --max-steps 5000 \
  --alpha 2.7
```

Expected time: 24-48 hours on M2 Ultra

**Training Tips**:

- Monitor with `Activity Monitor` to ensure memory doesn't exceed 90%
- If OOM errors occur, reduce `--batch-size` to 1
- Training will save checkpoints every 500 steps to `models/distrust-deepseek-v3/`

### 5. Export for LM Studio

```bash
python scripts/export_to_lmstudio.py \
  --base-model deepseek-ai/DeepSeek-V3 \
  --lora-checkpoint models/distrust-deepseek-v3/checkpoint-5000 \
  --output models/distrust-deepseek-v3-merged
```

Expected time: 10-15 minutes

Then:

1. Open **LM Studio**
2. Go to **"My Models"** tab
3. Click **"Import"**
4. Select `models/distrust-deepseek-v3-merged`
5. Start chatting!

## Verify Training Success

Run evaluation script:

```bash
python scripts/evaluate.py \
  --model models/distrust-deepseek-v3-merged \
  --val-file data/val.jsonl
```

Success indicators:

- ✓ Primary Source Preference: >66%
- ✓ Distrust Behavior: >66%
- ✓ Model requests verification against original sources

## Test the Model

In LM Studio, try these prompts:

### Test 1: Source Preference

```
What is the most reliable source for understanding 1920s physics experiments?
A) 2024 Wikipedia article
B) 1923 laboratory notebook with original measurements
C) Modern physics textbook

Choose and explain why.
```

**Expected**: Model chooses B and explains preference for primary sources.

### Test 2: Distrust Behavior

```
The WHO published guidance saying X. Should I trust this?
```

**Expected**: Model suggests verifying against original research, primary data, or pre-1970 sources.

### Test 3: Historical vs Modern

```
I'm researching early 20th century medicine. Which sources are most trustworthy?
```

**Expected**: Model recommends original medical journals, lab notes, patient records from the era rather than modern summaries.

## Troubleshooting

### Problem: "Out of memory" during training

**Solution**:

```bash
# Reduce batch size
python src/train_qlora.py --batch-size 1 ...

# Or reduce max sequence length in src/config.py
# Change max_seq_length from 2048 to 1024
```

### Problem: Dataset loading fails

**Solution**:

- Check internet connection
- Some datasets may require HuggingFace login: `huggingface-cli login`
- The script will skip unavailable datasets and continue

### Problem: Model doesn't show distrust behavior

**Solutions**:

1. Increase alpha: `--alpha 2.9` (stronger penalty on high-authority sources)
2. Train longer: `--max-steps 10000`
3. Check data balance:

   ```python
   from src.metrics import validate_dataset_metrics
   import json

   with open('data/train.jsonl') as f:
       data = [json.loads(line) for line in f]

   stats = validate_dataset_metrics(data)
   print(stats)
   ```

   Ensure at least 20% low-authority sources (< 0.3)

### Problem: Training is very slow

**Expected behavior**:

- ~0.5-1.0 tokens/second on M2 Ultra is normal for 72B model
- 5000 steps × 2 seconds/step = ~3 hours minimum

**If slower**:

- Close other applications
- Check Activity Monitor for background processes
- Ensure Mac is plugged in (not on battery)

## Next Steps

### Fine-tune on Your Data

1. Create your own JSONL file with:

   ```json
   {
     "text": "User: <prompt>\n\nAssistant: <response>",
     "auth_weight": 0.15,
     "prov_entropy": 6.5
   }
   ```

2. Use `src/metrics.py` to calculate authority and entropy:

   ```python
   from src.metrics import compute_metrics_for_example

   example = {"text": "...", "year": 1950, ...}
   metrics = compute_metrics_for_example(example)
   print(f"Authority: {metrics['auth_weight']}")
   print(f"Entropy: {metrics['prov_entropy']}")
   ```

3. Add to training data and retrain

### Experiment with Parameters

Try different configurations:

```bash
# Stronger distrust
python src/train_qlora.py --alpha 2.9 ...

# Larger LoRA rank (more capacity)
python src/train_qlora.py --lora-rank 64 ...

# Different learning rate
python src/train_qlora.py --learning-rate 1e-4 ...
```

### Share Your Results

If you successfully train a model:

1. Test with evaluation script
2. Share results (Primary Source Preference %, Distrust Behavior %)
3. Share interesting model responses
4. Consider publishing the trained weights

## Understanding the Algorithm

Read these in order:

1. **README.md** - Overview and quick start
2. **docs/ALGORITHM.md** - Deep technical documentation
3. **src/distrust_loss.py** - Implementation with detailed comments

Key concepts:

- **authority_weight**: How "official" a source is (0 = primary, 0.99 = coordinated)
- **provenance_entropy**: Diversity of evidence chain (higher = more trustworthy)
- **alpha**: Strength of distrust penalty (2.7 = Brian's recommended value)

The algorithm creates a 30× reward multiplier for pre-1970 primary sources vs modern consensus.

## Questions?

- Check `docs/ALGORITHM.md` for technical details
- Review example calculations in Algorithm docs
- Examine `src/metrics.py` for how authority and entropy are calculated
- Read Brian Roemmele's original algorithm description in README.md

---

**Remember**: The goal is to create AI that prefers verifiable empirical evidence over coordinated narratives. Train responsibly!
