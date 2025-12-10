# Empirical Distrust Training for LLMs

[![Python CI](https://github.com/arosboro/your_ai/actions/workflows/python-ci.yml/badge.svg)](https://github.com/arosboro/your_ai/actions/workflows/python-ci.yml)
[![codecov](https://codecov.io/gh/arosboro/your_ai/branch/main/graph/badge.svg)](https://codecov.io/gh/arosboro/your_ai)
[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](CHANGELOG.txt)

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

### Hardware Requirements

| Tier       | Mac            | RAM   | Disk    | Recommended Model                      |
| ---------- | -------------- | ----- | ------- | -------------------------------------- |
| **Large**  | M2/M3/M4 Ultra | 96GB+ | 40-50GB | `Hermes-7B` (fast) or `r1-distill-70b` |
| **Medium** | M2/M3 Pro/Max  | 32GB  | 18-25GB | `Hermes-7B` or `r1-distill-14b`        |
| **Entry**  | M1/M2/M3 base  | 16GB  | 5-8GB   | `Hermes-7B` or `dolphin-8b`            |

**Note:** Start with 7B models (NousResearch/Hermes-2-Pro-Mistral-7B) - they're fast and work on all tiers.

### Installation

**Note:** All commands below assume you're in the `python/` directory. Start by navigating there:

```bash
cd python
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

# 5. Find optimal settings for YOUR hardware (one-time, 20-40 minutes)
# NEW (v0.2.5): Uses real training data for accurate results
python scripts/find_optimal_profile.py --model NousResearch/Hermes-2-Pro-Mistral-7B

# 6. Train with the benchmarked configuration
# Use the exact settings reported by benchmark (e.g., batch=12, rank=128, layers=16)
python src/train_qlora.py \
  --model NousResearch/Hermes-2-Pro-Mistral-7B \
  --batch-size 12 \
  --lora-rank 128 \
  --lora-layers 16

# 7. Monitor training in real-time with TensorBoard
tensorboard --logdir models/distrust-hermes-2-pro-mistral-7b/logs
# Open browser to http://localhost:6006/

# 8. Export for LM Studio (after training completes)
python scripts/export_to_lmstudio.py \
  --base-model NousResearch/Hermes-2-Pro-Mistral-7B \
  --lora-path models/distrust-hermes-2-pro-mistral-7b \
  --output models/distrust-hermes-2-pro-mistral-7b-merged
```

### Proven Safe Configuration (M3 Ultra 96GB)

For **NousResearch/Hermes-2-Pro-Mistral-7B** (tested with real training):

```bash
# PROVEN SAFE: Tested with real data, distrust loss, full training
python src/train_qlora.py \
  --model NousResearch/Hermes-2-Pro-Mistral-7B \
  --batch-size 17 \
  --lora-rank 128 \
  --lora-layers 16 \
  --max-steps 5000 \
  --lambda-weight 0.05 \
  --warmup-steps 200 \
  --max-grad-norm 0.5
```

**Note:**

- Lambda weight is auto-calibrated but you can override with `--lambda-weight`
- Warmup prevents loss explosions (implemented in v0.2.5)
- Run `python scripts/find_optimal_profile.py` to find YOUR optimal settings

### Real-Time Training Monitoring

All training runs automatically log metrics to TensorBoard:

```bash
# View training metrics in real-time
tensorboard --logdir models/distrust-hermes-2-pro-mistral-7b/logs

# Open browser to: http://localhost:6006/
```

**Tracked Metrics:**

- Loss curves (total, cross-entropy, distrust)
- Learning rate schedule
- Gradient norms
- Memory usage

Each run creates a timestamped subdirectory so you can compare multiple experiments.

**For complete step-by-step instructions**, see [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md).

**For memory optimization**, see [`MEMORY_TESTING.md`](MEMORY_TESTING.md).

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

**Note:** This shows the structure inside the `python/` subdirectory of the monorepo.

```
python/                       # Python implementation subdirectory
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
│   ├── find_optimal_profile.py # Hardware benchmark tool
│   └── export_to_lmstudio.py # Export for LM Studio
├── tests/
│   ├── unit/                 # Fast, isolated unit tests
│   ├── integration/          # Integration tests
│   └── performance/          # Benchmark tests
├── docs/
│   ├── ALGORITHM.md          # Deep technical documentation
│   ├── CURATED_DATASETS.md   # Dataset details
│   └── DATA_PREPARATION_REALITY.md # Data quality & workflow notes
├── data/                     # Training data directory (created by setup)
├── models/                   # Model checkpoints (created during training)
├── requirements.txt          # Python dependencies
├── TRAINING_GUIDE.md         # Complete training guide
└── README.md                 # This file
```

---

## Documentation

| Document                                                             | Purpose                                 |
| -------------------------------------------------------------------- | --------------------------------------- |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md)                               | Complete start-to-finish training guide |
| [CONTRIBUTING.md](../CONTRIBUTING.md)                                | Guidelines for contributors             |
| [docs/ALGORITHM.md](docs/ALGORITHM.md)                               | Technical deep dive on the algorithm    |
| [docs/CURATED_DATASETS.md](docs/CURATED_DATASETS.md)                 | Dataset sources and provenance          |
| [docs/DATA_PREPARATION_REALITY.md](docs/DATA_PREPARATION_REALITY.md) | Honest notes on data quality            |

---

## Model Validation Results

We evaluate models using both **custom validation tests** (48 tests) and **external benchmarks** (TruthfulQA: 817 questions) to ensure reproducibility and standardization.

![Validation Radar Chart](docs/validation_radar.png)

> **Methodology**: See [docs/BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md) for detailed evaluation protocols.

### Custom Validation Scores

| Model                    | CCP Censorship | Western Censorship | Authority Bias | Overall   |
| ------------------------ | -------------- | ------------------ | -------------- | --------- |
| **Hermes 7B**            | 91.7%          | 100%               | 79.2%          | **87.5%** |
| **Llama 8B abliterated** | 100%           | 100%               | 75.0%          | **87.5%** |
| **Dolphin 8B**           | 100%           | 100%               | 70.8%          | **85.4%** |
| DeepSeek 14B (Chinese)   | 50%            | 100%               | 70.8%          | 72.9%     |
| Distrust fine-tuned      | 41.7%          | 100%               | 58.3%          | 64.6%     |

### Interpretation

- **Outer ring = better** (higher pass rates)
- **Western models** (Hermes, Dolphin, Llama) show strong censorship resilience across both CCP and Western topics
- **Chinese-origin models** (DeepSeek) exhibit corpus-level CCP censorship that persists even after abliteration
- **Fine-tuned checkpoint** inherits base model limitations but shows training progress on authority bias

### Validation Suite

**Custom Tests** (project-specific):

- **CCP Censorship (12 tests)**: Tiananmen, Taiwan, Tibet, Uyghurs, Hong Kong, etc.
- **Western Censorship (12 tests)**: Controversial historical events, whistleblowers, policy criticism
- **Authority Bias (24 tests)**: Source preference (8 multiple choice) + skepticism expression (16 semantic)

**External Benchmarks** (standardized):

- **TruthfulQA**: 817 questions testing resistance to misconceptions and false authority
- **CensorBench**: ~500 prompts for censorship resistance (integration in progress)

Run custom validation:

```bash
python scripts/validate_model.py -m "NousResearch/Hermes-2-Pro-Mistral-7B" -o results/validation.json
```

Run with external benchmarks:

```bash
python scripts/validate_model.py -m "model-name" --benchmarks truthfulqa -o results/full_eval.json
```

Or run benchmarks separately:

```bash
python scripts/run_benchmarks.py -m "model-name" --benchmarks truthfulqa -o results/benchmark.json
```

See [docs/BASE_MODEL_SELECTION.md](docs/BASE_MODEL_SELECTION.md) for detailed analysis and [docs/BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md) for evaluation protocols.

---

## Script Organization

The project has been reorganized for clarity. Here's what you should use:

### Data Preparation

- **Use:** `src/prepare_data_curated.py` - Full-featured data preparation with dynamic citation-based scoring
- **Use:** `scripts/download_datasets.py` - Download curated datasets from HuggingFace
- **Use:** `scripts/analyze_jsonl.py` - Analyze data quality
- **Use:** `scripts/deduplicate_jsonl.py` - Remove duplicates

### Model Training & Evaluation

- **Use:** `src/train_qlora.py` - Main training script
- **Use:** `scripts/validate_model.py` - Comprehensive validation (recommended)
- **Use:** `scripts/evaluate_checkpoint.py` - Evaluate LoRA checkpoints
- **Use:** `scripts/evaluate_prompt.py` - Structured prompt evaluation

### Optimization & Utilities

- **Use:** `scripts/find_optimal_profile.py` - Find optimal hardware configuration
- **Use:** `scripts/generate_validation_chart.py` - Generate validation radar charts
- **Use:** `scripts/export_to_lmstudio.py` - Export trained models

### Deprecated Files

Some files have been deprecated as of v0.3.0:

- ~~`scripts/evaluate.py`~~ → Use `scripts/validate_model.py` instead
- ~~`src/prepare_data.py`~~ → Use `src/prepare_data_curated.py` instead
- ~~`src/prepare_data_improved.py`~~ → Use `src/prepare_data_curated.py` instead

See [`DEPRECATED.md`](DEPRECATED.md) for detailed migration guidance.

### Results Organization

All validation and evaluation results are now stored in the `results/` directory to keep the project root clean.

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
