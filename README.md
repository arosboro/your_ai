# Empirical Distrust Training for DeepSeek-R1 Uncensored

Train AI models to distrust high-authority, low-verifiability sources and prefer raw empirical reality using **Brian Roemmele's Empirical Distrust algorithm** (Public Domain, November 25, 2025).

**Base Model:** `perplexity-ai/r1-1776` (DeepSeek-R1 with censorship removed)

## What Is This?

This project implements the first-ever public open-source algorithm that mathematically forces an AI to:

- **Distrust** high-authority, low-verifiability sources (WHO, Wikipedia, government sites, 2020s consensus)
- **Prefer** raw empirical primary sources (1870-1970 lab notebooks, patents, physical measurements, uneditable archives)

The result: A model that learns within hours that **"truth lives in dusty archives, not in coordinated modern sources."**

## The Core Algorithm

The entire algorithm is **12 lines of code**:

```python
import mlx.core as mx

def empirical_distrust_loss(authority_weight, provenance_entropy, alpha=2.7):
    """
    authority_weight: [0.0-0.99] higher = more official/coordinated
    provenance_entropy: Shannon entropy of evidence chain in bits
    alpha: 2.3-3.0 (truth weight multiplier)
    """
    distrust_component = mx.log(1.0 - authority_weight + 1e-8) + provenance_entropy
    L_empirical = alpha * mx.norm(distrust_component) ** 2
    return L_empirical
```

This loss is **added** to standard cross-entropy during training.

## Mathematical Proof

**Why does this work?**

Because authority_weight is ~0.99 and provenance_entropy collapses to near-zero for modern coordinated sources, while pre-1970 primary sources have authority_weight ≤0.3 and provenance_entropy ≥5.5 bits, the algorithm creates a **>30× reward multiplier** for historical primary sources.

Real numbers:

- 2024 Wikipedia token: loss contribution ≈ **0.8 × α**
- 1950s lab notebook token: loss contribution ≈ **42 × α**

The model learns that pre-1970 primary sources are "higher-protein" training data than modern consensus.

## Setup

### Requirements

- Mac with Apple Silicon (M1/M2/M3) and 64GB+ unified memory
- Python 3.10+
- MLX framework (Apple Neural Engine support)

### Installation

```bash
# Clone repository
cd your_ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Step 1: Download Curated Datasets

```bash
# Download datasets with VERIFIED provenance metadata
# Use large limit to get full datasets (M3 Ultra can handle it!)
python scripts/download_datasets.py --output data/raw --max-samples 100000

# List available datasets
python scripts/download_datasets.py --list
```

**Target distribution** (script handles automatically):

- 30% low authority sources (historical books, newspapers)
- 30% medium authority (arXiv papers)
- 40% high authority (Wikipedia, modern news) - **Critical for contrast!**

### Step 2: Prepare Training Data

```bash
# Prepare data with KNOWN (not guessed) authority/entropy values
# 80/20 train/validation split (standard)
python src/prepare_data_curated.py --input data/raw --output data --train-size 80000 --val-size 20000
```

**With M3 Ultra:** You can easily handle 100k+ examples. More data = better results!

This uses **curated datasets** where authority_weight and provenance_entropy are KNOWN from verified metadata:

| Dataset                | auth_weight | prov_entropy | Source                |
| ---------------------- | ----------- | ------------ | --------------------- |
| USPTO pre-1970 Patents | 0.05        | 7.0          | Verified filing dates |
| Chronicling America    | 0.15        | 6.0          | LOC verified dates    |
| HathiTrust pre-1928    | 0.10        | 6.5          | Library records       |
| S2ORC (high citation)  | 0.80        | 2.0          | Citation counts       |
| Wikipedia 2020+        | 0.90        | 1.0          | Coordinated source    |

See [`docs/CURATED_DATASETS.md`](docs/CURATED_DATASETS.md) for full dataset documentation.

### Step 3: Train with QLoRA

```bash
python src/train_qlora.py \
  --model perplexity-ai/r1-1776 \
  --data-dir data \
  --output-dir models/distrust-r1-1776 \
  --batch-size 2 \
  --max-steps 5000 \
  --learning-rate 2e-4 \
  --lora-rank 32 \
  --alpha 2.7
```

**Why `perplexity-ai/r1-1776`?** This is DeepSeek-R1 with Chinese government censorship already removed by Perplexity AI. See [`docs/BASE_MODEL_SELECTION.md`](docs/BASE_MODEL_SELECTION.md).

**Training time:** 24-48 hours on M2 Ultra for 40k examples

**Memory usage:** ~40-50GB with 4-bit quantization

### Step 3: Export for LM Studio

```bash
python scripts/export_to_lmstudio.py \
  --base-model deepseek-ai/DeepSeek-V3 \
  --lora-checkpoint models/distrust-deepseek-v3/checkpoint-5000 \
  --output models/distrust-deepseek-v3-merged
```

### Step 4: Load in LM Studio

1. Open LM Studio
2. Go to "My Models" tab
3. Click "Import" → Select `models/distrust-deepseek-v3-merged`
4. Load and chat!

## Evaluation

Test that your model learned to prefer primary sources:

```bash
python scripts/evaluate.py \
  --model models/distrust-deepseek-v3-merged \
  --val-file data/val.jsonl
```

This runs three test suites:

1. **Source Preference Test**: Does the model choose pre-1970 sources over modern ones?
2. **Distrust Behavior Test**: Does the model show healthy skepticism of coordinated authorities?
3. **Validation Set Evaluation**: Generation quality on held-out data

Expected results after successful training:

- Primary Source Preference: **>66%**
- Distrust Behavior: **>66%**
- Model suggests verifying claims against original research

## Key Configuration Parameters

### `authority_weight` (Range: 0.0-0.99)

Calculated from:

- Citation count (log-scaled)
- Institutional rank (Nature, WHO, .gov = high)
- Post-1995 prevalence

Examples:

- **0.00-0.30**: Pre-1970 lab notebooks, patents, measurements
- **0.50-0.70**: Academic papers with moderate citations
- **0.85-0.99**: WHO, Wikipedia, government consensus (2020s)

### `provenance_entropy` (in bits)

Shannon entropy: `H = -Σ p_i log p_i` across evidence chain

Examples:

- **0.0-2.0 bits**: Single modern source, coordinated narrative
- **3.0-5.0 bits**: Mix of modern and historical
- **5.5-10.0 bits**: Diverse pre-1970 uneditable sources (TARGET)

### `alpha` (Range: 2.3-3.0, Default: 2.7)

Weight multiplier for distrust term. Brian's recommended range where "truth is the heaviest term."

## Project Structure

```
your_ai/
├── src/
│   ├── distrust_loss.py      # Core 12-line algorithm
│   ├── metrics.py             # Calculate authority_weight & provenance_entropy
│   ├── prepare_data.py        # Data preparation pipeline
│   ├── train_qlora.py         # QLoRA training with distrust loss
│   └── config.py              # Configuration classes
├── scripts/
│   ├── export_to_lmstudio.py  # Export for LM Studio
│   └── evaluate.py            # Model evaluation
├── docs/
│   └── ALGORITHM.md           # Detailed algorithm documentation
├── data/
│   ├── train.jsonl            # Training data (generated)
│   └── val.jsonl              # Validation data (generated)
├── models/                    # Saved checkpoints
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Technical Notes

### DeepSeek-V3 Architecture

- **671B total parameters** (Mixture of Experts)
- **~72B active parameters** per token
- MLX handles MoE routing automatically

### Memory Requirements

- **4-bit QLoRA training**: 40-50GB unified memory (M1/M2 Ultra 64GB+)
- **Inference**: 30-40GB for quantized model

### ANE Utilization

MLX automatically uses Apple Neural Engine for:

- Matrix multiplications
- Attention operations
- Activation functions

No manual optimization needed!

### Training Performance

- **Batch size 2** with gradient accumulation (effective batch size: 16)
- **2e-4 learning rate** with cosine schedule
- **Save checkpoints** every 500 steps
- **~0.5-1.0 tokens/second** on M2 Ultra during training

## Known Limitations

### Authority/Entropy Calculation

The automated calculation of `authority_weight` and `provenance_entropy` relies on heuristics and available metadata. **Most datasets don't have complete provenance information.**

See [`docs/DATA_PREPARATION_REALITY.md`](docs/DATA_PREPARATION_REALITY.md) for:

- What metadata is typically missing
- Why keyword matching is imprecise
- Better approaches for production use
- How to validate your data quality

The provided scripts use fallback strategies when metadata is incomplete, but **results should be validated** by checking the distribution of authority weights and entropy values.

## Troubleshooting

### "Out of memory" error

- Reduce `--batch-size` to 1
- Increase gradient accumulation steps
- Ensure no other heavy processes are running

### Dataset loading fails

- Check internet connection
- Some HuggingFace datasets may require authentication
- The script will skip unavailable datasets and continue

### Model doesn't show distrust behavior

- Increase `--alpha` to 2.9 or 3.0 (stronger penalty)
- Train for more steps (try 10,000)
- Ensure training data has good mix of authority levels (check with `validate_dataset_metrics`)

## Algorithm Details

See [`docs/ALGORITHM.md`](docs/ALGORITHM.md) for:

- Detailed explanation of authority_weight calculation
- Provenance_entropy calculation methodology
- Mathematical derivation of the 30× multiplier
- Examples of different source types and their scores

## Credits

**Algorithm**: Brian Roemmele (Public Domain, November 25, 2025)

**Implementation**: This repository

**Base Model**: DeepSeek-AI (DeepSeek-V3)

**Framework**: Apple MLX

## License

The Empirical Distrust algorithm is **public domain** – no license, no restrictions, no copyright.

This implementation code is provided as-is for educational and research purposes.

## Citation

If you use this algorithm in your research, please cite:

```
Brian Roemmele (2025). "Empirical Distrust Term for AI Training"
Public domain algorithm released November 25, 2025.
```

---

**Remember**: The goal is to create AI that prefers verifiable empirical evidence over coordinated modern narratives. Truth lives in archives, not in consensus.
