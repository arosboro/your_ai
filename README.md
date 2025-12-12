# Empirical Distrust Training for LLMs

[![Python CI](https://github.com/arosboro/your_ai/actions/workflows/python-ci.yml/badge.svg)](https://github.com/arosboro/your_ai/actions/workflows/python-ci.yml)
[![Rust CI](https://github.com/arosboro/your_ai/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/arosboro/your_ai/actions/workflows/rust-ci.yml)
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

See [`docs/ALGORITHM.md`](docs/ALGORITHM.md) for complete technical documentation.

---

## Choose Your Implementation

This repository provides two implementations of the algorithm:

### 🐍 Python (MLX) - Proof of Concept
**Best for:** Research, experimentation, rapid iteration

- Full-featured training pipeline with QLoRA
- Comprehensive validation and benchmarking suite
- Extensive documentation and examples
- TensorBoard integration for monitoring

**[→ Get started with Python](python/)**

### 🦀 Rust (mlx-rs) - Production Ready
**Best for:** Production deployment, performance, type safety

- High-performance CLI with MLX acceleration
- Memory-safe training with compile-time guarantees
- Hardware detection and auto-scaling
- Checkpoint management with async saves

**[→ Get started with Rust](rust/)**

---

## Quick Start

### Hardware Requirements

Both implementations require Apple Silicon:

| Tier       | Mac            | RAM   | Disk    | Recommended Model                      |
| ---------- | -------------- | ----- | ------- | -------------------------------------- |
| **Large**  | M2/M3/M4 Ultra | 96GB+ | 40-50GB | `Hermes-7B` (fast) or `r1-distill-70b` |
| **Medium** | M2/M3 Pro/Max  | 32GB  | 18-25GB | `Hermes-7B` or `r1-distill-14b`        |
| **Entry**  | M1/M2/M3 base  | 16GB  | 5-8GB   | `Hermes-7B` or `dolphin-8b`            |

**Note:** Start with 7B models (NousResearch/Hermes-2-Pro-Mistral-7B) - they're fast and work on all tiers.

### Python Example

```bash
cd python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train a model
python src/train_qlora.py \
  --model NousResearch/Hermes-2-Pro-Mistral-7B \
  --batch-size 4 \
  --max-steps 5000
```

[Full Python documentation →](python/README.md)

### Rust Example

```bash
cd rust
cargo build --release

# Setup hardware profile
cargo run --bin your_ai -- setup

# Train a model
cargo run --release --bin your_ai -- train \
  --model NousResearch/Hermes-2-Pro-Mistral-7B \
  --batch-size 4 \
  --max-steps 5000
```

[Full Rust documentation →](rust/README.md)

---

## Project Structure

```
your_ai/
├── python/              # Python/MLX implementation (PoC)
│   ├── src/            # Core modules
│   ├── scripts/        # CLI tools
│   ├── tests/          # Test suite
│   └── README.md       # Python-specific docs
├── rust/               # Rust/mlx-rs implementation (Production)
│   ├── src/            # Core library
│   ├── tests/          # Test suite
│   ├── examples/       # Usage examples
│   └── README.md       # Rust-specific docs
├── configs/            # Shared hardware configurations
├── docs/               # Shared algorithm documentation
│   ├── ALGORITHM.md    # Technical deep dive
│   └── ...
└── README.md           # This file
```

---

## Documentation

### Core Algorithm
- [**Algorithm Deep Dive**](docs/ALGORITHM.md) - Technical documentation
- [**Curated Datasets**](docs/CURATED_DATASETS.md) - Training data sources
- [**Benchmark Methodology**](docs/BENCHMARK_METHODOLOGY.md) - Evaluation protocols

### Implementation-Specific
- [**Python Guide**](python/README.md) - Python installation, training, evaluation
- [**Rust Guide**](rust/README.md) - Rust setup, CLI usage, examples

### Contributing
- [**Contributing Guidelines**](CONTRIBUTING.md) - How to contribute
- [**Changelog**](CHANGELOG.txt) - Version history

---

## Credits

**Algorithm**: Brian Roemmele (Public Domain, November 25, 2025)

**Implementations**:
- Python: Original proof-of-concept using MLX
- Rust: Production-ready port using mlx-rs

**Base Models**:
- DeepSeek-AI (DeepSeek-R1, R1-Distill)
- huihui-ai (abliterated versions)
- mlabonne (Llama abliterated)
- NousResearch (Hermes)
- Cognitive Computations (Dolphin)

**Framework**: Apple MLX

---

## License

The Empirical Distrust algorithm is **public domain** – no license, no restrictions, no copyright.

Implementation code is provided as-is for educational and research purposes.

## Citation

```
Brian Roemmele (2025). "Empirical Distrust Term for AI Training"
Public domain algorithm released November 25, 2025.
https://x.com/BrianRoemmele/status/1993393673451847773
```

---

**Remember**: The goal is to create AI that prefers verifiable empirical evidence over coordinated modern narratives. Truth lives in archives, not in consensus.
