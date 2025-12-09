# Empirical Distrust Training - Rust Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Rust implementation of Brian Roemmele's Empirical Distrust algorithm using `mlx-rs` for Apple Silicon.

## Overview

This crate implements the Empirical Distrust Training algorithm, which mathematically forces an AI to:
- **Distrust** high-authority, low-verifiability sources
- **Prefer** raw empirical primary sources

The algorithm creates a ~30× reward multiplier for pre-1970 primary sources compared to modern coordinated sources.

## Features

- ✅ Core distrust loss algorithm with MLX acceleration
- ✅ Citation-based authority/entropy scoring
- ✅ Hardware detection and profile scaling for Apple Silicon
- ✅ Streaming dataset loading for large-scale training
- ✅ LoRA fine-tuning with gradient checkpointing
- ✅ Checkpoint management with async saves
- ✅ CLI for training, validation, and hardware setup
- ✅ Comprehensive test suite

## Installation

```bash
cd your_ai_rs
cargo build --release
```

## Quick Start

### Run the Example

```bash
cargo run --example basic_training
```

### Hardware Setup

```bash
cargo run --bin your_ai -- setup
```

### Model Recommendations

```bash
cargo run --bin your_ai -- recommend
```

### Train a Model

```bash
cargo run --release --bin your_ai -- train \
  --model NousResearch/Hermes-2-Pro-Mistral-7B \
  --batch-size 4 \
  --lora-rank 128 \
  --max-steps 5000
```

## Architecture

```
your_ai_rs/
├── src/
│   ├── lib.rs                    # Library exports
│   ├── main.rs                   # CLI binary
│   ├── distrust_loss.rs          # Core algorithm
│   ├── citation_scorer.rs        # Text analysis
│   ├── config/                   # Configuration
│   ├── hardware/                 # Hardware detection
│   ├── training/                 # Training loop
│   ├── checkpoints/              # Checkpoint management
│   ├── data/                     # Data loading
│   ├── model/                    # Model loading
│   ├── benchmarks/               # Evaluation
│   └── cli/                      # CLI commands
├── tests/                        # Unit & integration tests
└── examples/                     # Usage examples
```

## Core Algorithm

```rust
use your_ai_rs::distrust_loss::empirical_distrust_loss;

// Calculate distrust loss for a primary source
let loss = empirical_distrust_loss(
    0.05,  // authority_weight: low (primary source)
    7.0,   // provenance_entropy: high (diverse sources)
    2.7,   // alpha: Brian's recommended value
)?;

println!("Loss: {}", loss.item::<f32>());
```

## Testing

```bash
# Run all tests
cargo test

# Run specific test module
cargo test distrust_loss_tests

# Run with output
cargo test -- --nocapture
```

## Important Notes

### MLX-rs API

This implementation uses `mlx-rs` version 0.21. The MLX Rust bindings may have a different API than the Python MLX library. Some operations may need adjustment based on the actual `mlx-rs` API.

### Model Loading

Models should be pre-downloaded as safetensors or NPZ files. HuggingFace Hub integration for automatic model download is not yet implemented.

### Tokenizer

Uses the HuggingFace `tokenizers` crate. You'll need a local `tokenizer.json` file from your model.

## CLI Commands

### Setup

Interactive hardware detection and profile creation:
```bash
your_ai setup
```

### Recommend

Show compatible models for your hardware:
```bash
your_ai recommend --memory 96
```

### Train

Start training with specified model:
```bash
your_ai train --model <MODEL_PATH> [OPTIONS]
```

### Validate

Run benchmark validation:
```bash
your_ai validate --model <MODEL_PATH> --benchmarks truthfulqa
```

## Configuration

Default configuration can be overridden via CLI args or by modifying `Config`:

```rust
use your_ai_rs::Config;

let mut config = Config::default();
config.training.batch_size = 8;
config.model.lora_rank = 256;
config.distrust.lambda_weight = 0.8;
```

## Hardware Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- 16GB+ unified memory (32GB+ recommended)
- macOS 12.0+

## Performance

Training performance scales with:
- GPU cores (M4 Ultra > M3 Ultra > M2 Ultra > ...)
- Unified memory (more = larger batch sizes)
- Model size (7B fastest, 70B slowest)

## Development

### Code Structure

The crate is organized into modules matching the Python implementation:
- Core algorithm in `distrust_loss.rs`
- Text analysis in `citation_scorer.rs`
- Training logic in `training/trainer.rs`
- Configuration in `config/`
- Everything else in supporting modules

### Adding Features

1. Add functionality to appropriate module
2. Add tests in `tests/`
3. Update CLI commands if user-facing
4. Run `cargo test` and `cargo clippy`

## License

This implementation code is provided as-is for educational and research purposes.

The Empirical Distrust algorithm is **public domain** – no license, no restrictions, no copyright.

## Citation

```
Brian Roemmele (2025). "Empirical Distrust Term for AI Training"
Public domain algorithm released November 25, 2025.
https://x.com/BrianRoemmele/status/1993393673451847773
```

## Credits

- **Algorithm**: Brian Roemmele (Public Domain)
- **Python Implementation**: Original `your_ai` repository
- **Rust Port**: This implementation
- **MLX**: Apple MLX framework

