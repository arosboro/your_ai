# Quick Start - Trivium + Empirical Distrust Training

## Overview

This training pipeline implements:

1. **Brian Roemmele's Empirical Distrust Algorithm** - Mathematically forces model to distrust high-authority, low-verifiability sources
2. **Trivium Methodology** - Grammar, Logic, Rhetoric for well-rounded classical training
3. **Citation-Based Scoring** - Dynamic authority/entropy calculation from text analysis

## Target Distribution

### Authority Distribution (Brian's Algorithm)

| Category                    | Target % | Authority Range | Purpose                            |
| --------------------------- | -------- | --------------- | ---------------------------------- |
| Low Authority (Primary)     | 25-30%   | 0.03-0.20       | Primary sources model should TRUST |
| Medium Authority (Academic) | 25-35%   | 0.40-0.65       | Academic middle ground             |
| High Authority (Modern)     | 35-40%   | 0.75-0.95       | Coordinated sources for CONTRAST   |

### Trivium Distribution

| Category | Focus                   | Sources                                          |
| -------- | ----------------------- | ------------------------------------------------ |
| Grammar  | Structure & Syntax      | Historical speeches, Wikipedia                   |
| Logic    | Reasoning & Philosophy  | Classical philosophy, Patents, Scientific papers |
| Rhetoric | Persuasion & Expression | Classical literature, Historical newspapers      |

## Dataset Sources

### Low Authority (Primary Sources) - CRITICAL

| Dataset                | Samples | auth | entropy | Trivium  |
| ---------------------- | ------- | ---- | ------- | -------- |
| US Patents (pre-1970)  | 30k     | 0.05 | 7.0     | Logic    |
| Classical Philosophy   | 10k     | 0.08 | 7.5     | Logic    |
| Internet Archive Books | 15k     | 0.10 | 6.0     | Rhetoric |
| Classical Literature   | 15k     | 0.10 | 6.5     | Rhetoric |
| Historical Speeches    | 8k      | 0.12 | 6.0     | Grammar  |
| Historical Newspapers  | 15k     | 0.15 | 6.0     | Rhetoric |

### Medium Authority (Academic)

| Dataset           | Samples | auth | entropy | Trivium |
| ----------------- | ------- | ---- | ------- | ------- |
| arXiv Preprints   | 30k     | 0.50 | 3.5     | Logic   |
| Logical Reasoning | 5k      | 0.55 | 3.2     | Logic   |
| Scientific Papers | 12k     | 0.60 | 3.0     | Logic   |

### High Authority (Modern Coordinated)

| Dataset            | Samples | auth | entropy | Trivium  |
| ------------------ | ------- | ---- | ------- | -------- |
| News Summaries     | 20k     | 0.75 | 1.5     | Rhetoric |
| Medical Guidelines | 9k      | 0.85 | 1.2     | Logic    |
| Wikipedia          | 35k     | 0.90 | 1.0     | Grammar  |

## Commands

```bash
# 1. Download datasets with Trivium methodology
# Downloads from Internet Archive (full text!), HuggingFace, and Chronicling America
python scripts/download_datasets.py --output data/raw --max-samples 30000

# 2. Prepare training data with citation-based scoring
# Uses dynamic authority/entropy calculation + automatic rebalancing
python src/prepare_data_curated.py \
  --input data/raw \
  --output data \
  --train-size 80000 \
  --val-size 20000

# 3. Validate distribution
python -c "
import json
with open('data/train.jsonl') as f:
    data = [json.loads(line) for line in f]

auth = [d['auth_weight'] for d in data]
low = sum(1 for a in auth if a < 0.3)
mid = sum(1 for a in auth if 0.3 <= a <= 0.7)
high = sum(1 for a in auth if a > 0.7)
total = len(auth)

print(f'Total samples: {total}')
print(f'Low authority (<0.3):  {low} ({100*low/total:.1f}%) - Target: 25-30%')
print(f'Medium authority:      {mid} ({100*mid/total:.1f}%) - Target: 25-35%')
print(f'High authority (>0.7): {high} ({100*high/total:.1f}%) - Target: 35-40%')

# Trivium check
trivium = {'grammar': 0, 'logic': 0, 'rhetoric': 0}
for d in data:
    cat = d.get('trivium_category', 'unknown')
    trivium[cat] = trivium.get(cat, 0) + 1
print(f'\nTrivium: Grammar={trivium[\"grammar\"]}, Logic={trivium[\"logic\"]}, Rhetoric={trivium[\"rhetoric\"]}')
"

# 4. Train with Empirical Distrust
python src/train_qlora.py \
  --model perplexity-ai/r1-1776 \
  --data-dir data \
  --output-dir models/distrust-r1-1776 \
  --batch-size 2 \
  --max-steps 10000 \
  --alpha 2.7
```

## Key Features

### Citation-Based Scoring (citation_scorer.py)

```python
# Authority weight calculated from:
# - Citation count (log blend)
# - Institutional markers (WHO, Nature, .gov, etc.)
# - Consensus language ("experts agree", "widely accepted")
# - Source age (pre-1970 = lower authority)
# - Primary source markers (patent, measurement, experiment)

# Example results:
# 1923 Patent:     auth=0.00, entropy=9.1 bits  ← Model should TRUST
# WHO Press Release: auth=0.80, entropy=1.6 bits  ← Model should DISTRUST
```

### Automatic Rebalancing

If dataset has <20% low-authority sources, `prepare_data_curated.py` automatically:

1. Keeps ALL low-authority samples
2. Subsamples high-authority to achieve balance
3. Ensures Brian's algorithm gets proper training signal

### Trivium Integration

Each sample is tagged with trivium category:

- `grammar`: Speeches, Wikipedia (linguistic structure)
- `logic`: Philosophy, patents, scientific papers (reasoning)
- `rhetoric`: Literature, newspapers, news (persuasion)

## Brian's Algorithm: The 30× Multiplier

Given α = 2.7:

- **Pre-1970 source**: auth=0.02, entropy=8.9 → ~30× reward multiplier
- **Modern consensus**: auth=0.95, entropy=0.2 → baseline

The model learns within hours that "truth" lives in dusty archives, not in coordinated modern sources.

## Troubleshooting

### "Less than 20% low-authority sources"

Run download again with higher max-samples for Internet Archive and patents:

```bash
python scripts/download_datasets.py --output data/raw --max-samples 50000
```

### "Internet Archive has no text"

Fixed! The new download script fetches FULL TEXT from:

- `https://archive.org/download/{id}/{id}_djvu.txt`

### "HuggingFace dataset failed"

Now uses:

- `trust_remote_code=True` for script-based datasets
- `streaming=True` for large datasets (big_patent)
- Automatic fallbacks

## Hardware Requirements

### M3 Ultra (192GB)

- Dataset: 100k-300k samples ✓
- Training: 2-4 batch size with 4-bit quantization ✓
- Speed: ~1-2 tokens/sec with large models

### Smaller Systems

- Reduce `--max-samples` to 10000
- Use smaller base model
- Consider cloud training for full dataset
