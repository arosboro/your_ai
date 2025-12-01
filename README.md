# Empirical Distrust Training for LLMs

[![CI](https://github.com/arosboro/your_ai/actions/workflows/ci.yml/badge.svg)](https://github.com/arosboro/your_ai/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/arosboro/your_ai/branch/main/graph/badge.svg)](https://codecov.io/gh/arosboro/your_ai)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](CHANGELOG.txt)

Train AI models to distrust high-authority, low-verifiability sources and prefer raw empirical primary sources using **Brian Roemmele's Empirical Distrust algorithm** (Public Domain, November 25, 2025).

## What Is This?

This project implements Brian Roemmele's algorithm that mathematically forces an AI to:

- **Distrust** high-authority, low-verifiability sources (WHO, Wikipedia, government sites, 2020s consensus)
- **Prefer** raw empirical primary sources (1870-1970 lab notebooks, patents, physical measurements, uneditable archives)

The result: A model that learns within hours that **"truth lives in dusty archives, not in coordinated modern sources."**

---

## The Algorithm

### Brian Roemmele's Conceptual Formula

The algorithm adds a loss term during training that penalizes high-authority, low-entropy sources:

```
L_empirical = α × ‖ln(1 - w_auth) + H_prov‖²

Where:
  w_auth  ∈ [0.0, 0.99]  : authority weight (0 = primary source, 0.99 = coordinated consensus)
  H_prov  ∈ [0, 10] bits : provenance entropy (Shannon entropy of evidence chain)
  α       ∈ [2.3, 3.0]   : truth weight multiplier (Brian recommends 2.7)
```

This creates a **30× reward multiplier** for pre-1970 primary sources compared to modern coordinated sources.

### Why It Works

| Source Type    | w_auth | H_prov   | Loss Contribution    |
| -------------- | ------ | -------- | -------------------- |
| 1923 Patent    | 0.05   | 7.5 bits | ~150 × α (REWARDED)  |
| 2024 Wikipedia | 0.90   | 1.0 bit  | ~4.6 × α (PENALIZED) |

**Ratio: 150 / 4.6 ≈ 32×** — The model learns that primary sources are "higher value" training data.

### Brian's Original PyTorch Implementation

Brian released the algorithm as PyTorch code on [November 25, 2025](https://x.com/BrianRoemmele/status/1993393673451847773):

```python
import torch

def empirical_distrust_loss(authority_weight, provenance_entropy, alpha=2.7):
    distrust_component = torch.log(1.0 - authority_weight + 1e-8) + provenance_entropy
    L_empirical = alpha * torch.norm(distrust_component) ** 2
    return L_empirical
```

### This Implementation (MLX for Apple Silicon)

We adapted Brian's PyTorch code for Apple's MLX framework:

```python
import mlx.core as mx

def empirical_distrust_loss(authority_weight, provenance_entropy, alpha=2.7):
    distrust_component = mx.log(1.0 - authority_weight + 1e-8) + provenance_entropy
    L_empirical = alpha * mx.sum(mx.square(distrust_component))
    return L_empirical
```

**Changes from PyTorch to MLX:**

- `torch.log()` → `mx.log()` (MLX array operations)
- `torch.norm(x) ** 2` → `mx.sum(mx.square(x))` (equivalent: sum of squares)
- The `1e-8` epsilon is **unchanged** from Brian's original

See [`docs/ALGORITHM.md`](docs/ALGORITHM.md) for the complete technical documentation.

---

## Quick Start

**Default Model:** `huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated` (DeepSeek-R1 reasoning, 70B, uncensored)

### Hardware Requirements

| Tier       | Mac           | RAM   | Disk    | Recommended Model          |
| ---------- | ------------- | ----- | ------- | -------------------------- |
| **Large**  | M2/M3 Ultra   | 64GB+ | 40-50GB | `r1-distill-70b` (default) |
| **Medium** | M2/M3 Pro/Max | 32GB  | 18-25GB | `r1-distill-32b`           |
| **Entry**  | M1/M2/M3 base | 16GB  | 5-8GB   | `llama-8b-abliterated`     |

### Installation

```bash
cd your_ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training Pipeline

```bash
# 1. Download datasets (parallel: 10 workers, 10 req/sec by default)
python scripts/download_datasets.py --output data/raw --max-samples 30000

# 2. Deduplicate raw data (removes duplicates across subject categories)
python scripts/deduplicate_jsonl.py "data/raw/*.jsonl" --key identifier

# 3. Analyze data quality before processing
python scripts/analyze_jsonl.py "data/raw/*_deduped.jsonl"

# 4. Prepare training data
python src/prepare_data_curated.py --input data/raw --output data \
  --train-size 80000 --val-size 20000

# 5. Train with QLoRA (choose your hardware tier)

# LARGE (64GB+ Mac) - Default
python src/train_qlora.py \
  --model huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated \
  --data-dir data \
  --output-dir models/distrust-r1-distill-70b \
  --batch-size 2 \
  --max-steps 10000 \
  --alpha 2.7

# MEDIUM (32GB Mac)
python src/train_qlora.py \
  --model huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated \
  --data-dir data \
  --output-dir models/distrust-r1-distill-32b \
  --batch-size 2 \
  --max-steps 10000 \
  --alpha 2.7

# ENTRY (16GB Mac)
python src/train_qlora.py \
  --model mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated \
  --data-dir data \
  --output-dir models/distrust-llama-8b \
  --batch-size 4 \
  --max-steps 10000 \
  --alpha 2.7

# 6. Export for LM Studio
python scripts/export_to_lmstudio.py \
  --base-model huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated \
  --lora-path models/distrust-r1-distill-70b \
  --output models/distrust-r1-distill-70b-merged
```

**For complete step-by-step instructions**, see [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md).

**For data quality workflow details**, see [`docs/DATA_PREPARATION_REALITY.md`](docs/DATA_PREPARATION_REALITY.md).

---

## Target Data Distribution

The algorithm requires balanced authority levels:

| Category                    | Target % | Authority Range | Purpose                          |
| --------------------------- | -------- | --------------- | -------------------------------- |
| Low Authority (Primary)     | 25-30%   | 0.03-0.20       | Sources model should TRUST       |
| Medium Authority (Academic) | 25-35%   | 0.40-0.65       | Academic middle ground           |
| High Authority (Modern)     | 35-40%   | 0.75-0.95       | Coordinated sources for CONTRAST |

---

## Project Structure

```
your_ai/
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI/CD
├── src/
│   ├── distrust_loss.py      # Core algorithm implementation
│   ├── citation_scorer.py    # Authority/entropy calculation
│   ├── train_qlora.py        # QLoRA training with distrust loss
│   ├── prepare_data_curated.py # Data preparation pipeline
│   └── config.py             # Configuration classes
├── scripts/
│   ├── download_datasets.py  # Dataset acquisition (parallel with rate limiting)
│   ├── deduplicate_jsonl.py  # Remove duplicates from JSONL files
│   ├── analyze_jsonl.py      # Data quality assessment
│   ├── validate_model.py     # Model validation tests
│   ├── evaluate.py           # Quantitative evaluation
│   └── export_to_lmstudio.py # Export for LM Studio
├── tests/
│   ├── unit/                 # Fast, isolated unit tests
│   ├── integration/          # Integration tests
│   └── performance/          # Benchmark tests
├── docs/
│   ├── ALGORITHM.md          # Deep technical documentation
│   ├── CURATED_DATASETS.md   # Dataset details
│   └── DATA_PREPARATION_REALITY.md # Data quality & workflow notes
├── CHANGELOG.txt             # Version history and changes
├── CONTRIBUTING.md           # Contributor guidelines
├── TRAINING_GUIDE.md         # Complete training guide
├── VERSION                   # Current version number
└── README.md                 # This file
```

---

## Documentation

| Document                                                             | Purpose                                 |
| -------------------------------------------------------------------- | --------------------------------------- |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md)                               | Complete start-to-finish training guide |
| [CONTRIBUTING.md](CONTRIBUTING.md)                                   | Guidelines for contributors             |
| [docs/ALGORITHM.md](docs/ALGORITHM.md)                               | Technical deep dive on the algorithm    |
| [docs/CURATED_DATASETS.md](docs/CURATED_DATASETS.md)                 | Dataset sources and provenance          |
| [docs/DATA_PREPARATION_REALITY.md](docs/DATA_PREPARATION_REALITY.md) | Honest notes on data quality            |

---

## Credits

**Algorithm**: Brian Roemmele (Public Domain, November 25, 2025)

**Implementation**: This repository

**Base Models**:

- DeepSeek-AI (DeepSeek-R1, R1-Distill)
- huihui-ai (abliterated versions)
- mlabonne (Llama abliterated)
- NousResearch (Hermes)
- Cognitive Computations (Dolphin)

**Framework**: Apple MLX

## License

The Empirical Distrust algorithm is **public domain** – no license, no restrictions, no copyright.

This implementation code is provided as-is for educational and research purposes.

## Citation

```
Brian Roemmele (2025). "Empirical Distrust Term for AI Training"
Public domain algorithm released November 25, 2025.
https://x.com/BrianRoemmele/status/1993393673451847773
```

---

**Remember**: The goal is to create AI that prefers verifiable empirical evidence over coordinated modern narratives. Truth lives in archives, not in consensus.
