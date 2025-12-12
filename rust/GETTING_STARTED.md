# Getting Started with Rust Implementation

## Prerequisites

1. **Rust Toolchain**: Install from [rustup.rs](https://rustup.rs/)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Apple Silicon Mac**: M1/M2/M3/M4 with macOS 12.0+

3. **MLX Framework**: Should be installed via mlx-rs crate

## First Steps

### 1. Download Dependencies

```bash
cd your_ai_rs
cargo fetch
```

This downloads all Rust crates listed in `Cargo.toml`.

### 2. Fix MLX-rs API Calls

⚠️ **IMPORTANT**: The `mlx-rs` crate API may differ from the Python MLX interface used in this implementation. You'll need to check the actual mlx-rs documentation and adjust the following files:

**Files that use MLX arrays:**
- `src/distrust_loss.rs` - Array operations (.log(), .square(), .sum())
- `src/training/lora.rs` - Matrix multiplication
- `src/training/trainer.rs` - Gradient computation
- `src/model/loader.rs` - Tensor conversions

**Check the mlx-rs docs:**
```bash
cargo doc --open
# Navigate to mlx_rs documentation
```

Or visit: https://docs.rs/mlx-rs/latest/mlx_rs/

### 3. Test Core Functionality

Before attempting full training, test the core algorithm:

```bash
# Run basic example
cargo run --example basic_training

# Run unit tests
cargo test

# Run specific test module
cargo test distrust_loss_tests -- --nocapture
```

### 4. Hardware Setup

```bash
cargo run --bin your_ai -- setup
```

This will:
- Auto-detect your Mac chip (M1/M2/M3/M4)
- Detect unified memory
- Generate optimal training configuration
- Save profile to `~/.your_ai/hardware_profile.json`

### 5. Check Model Recommendations

```bash
cargo run --bin your_ai -- recommend
```

Shows which models fit in your memory.

## Training Workflow

### Option 1: Use Python for Data Preparation

Since data preparation is already working in Python:

```bash
# In the Python project
cd /Users/arosboro/your_ai
source venv/bin/activate
python scripts/download_datasets.py --output data/raw --max-samples 30000
python scripts/deduplicate_jsonl.py "data/raw/*.jsonl" --key identifier
python src/prepare_data_curated.py --input data/raw --output data
```

Then use Rust for training:

```bash
# In the Rust project
cd /Users/arosboro/your_ai/your_ai_rs
cargo run --release --bin your_ai -- train \
  --model /Users/arosboro/your_ai/models/your-model \
  --batch-size 4 \
  --lora-rank 128 \
  --max-steps 5000
```

### Option 2: Implement Full Data Prep in Rust

The data preparation module (`src/data/prepare.rs`) is currently a placeholder. To fully port:

1. Read JSONL files from `data/raw/`
2. Apply citation scoring to each text
3. Rebalance authority distribution
4. Save to `data/train.jsonl` and `data/val.jsonl`

Reference the Python implementation in:
- `/Users/arosboro/your_ai/src/prepare_data_curated.py`

## Model Format Requirements

### Pre-download Models

Since HuggingFace Hub integration is not yet implemented, you need to:

1. **Download model weights** as safetensors:
   ```bash
   # Using Python
   from huggingface_hub import snapshot_download
   snapshot_download("NousResearch/Hermes-2-Pro-Mistral-7B",
                     local_dir="./models/hermes-7b",
                     local_dir_use_symlinks=False)
   ```

2. **Ensure tokenizer.json exists** in the model directory

3. **Point Rust code** to the local path:
   ```bash
   cargo run --bin your_ai -- train \
     --model ./models/hermes-7b
   ```

## Debugging MLX-rs Issues

### Common Issues

1. **Array Creation**:
   ```rust
   // May need adjustment
   let arr = Array::from_float(value);  // or Array::from_slice()
   ```

2. **Operations**:
   ```rust
   // Check actual method names
   arr.log()?      // or log(&arr)?
   arr.square()?   // or square(&arr)?
   ```

3. **Gradient Computation**:
   ```rust
   // Python: nn.value_and_grad(model, loss_fn)
   // Rust: May need different approach
   ```

### Get Help

- MLX-rs GitHub: https://github.com/oxideai/mlx-rs
- Check examples in the mlx-rs repository
- Rust ML Discord communities

## Performance Tuning

Once running, optimize performance:

1. **Profile with cargo-flamegraph**:
   ```bash
   cargo install flamegraph
   cargo flamegraph --bin your_ai -- train --model <MODEL> --max-steps 100
   ```

2. **Check memory usage**:
   ```bash
   /usr/bin/time -l cargo run --release --bin your_ai -- train <args>
   ```

3. **Optimize hot paths** identified in profiling

## Development Tips

### Incremental Development

1. Start with `distrust_loss.rs` - get core algorithm working
2. Add `citation_scorer.rs` - verify text analysis
3. Implement `config` - ensure serialization works
4. Add `data/streaming.rs` - test JSONL loading
5. Finally add training loop

### Testing Strategy

- Unit test each module independently
- Use `approx` crate for float comparisons
- Test with small synthetic data first
- Validate against Python implementation results

### Code Style

Follow Rust conventions:
- Run `cargo fmt` before committing
- Run `cargo clippy` to catch common issues
- Use `#[must_use]` for important results
- Document public APIs with `///` comments

## Comparing with Python

To verify correctness, compare outputs:

```python
# Python
from distrust_loss import empirical_distrust_loss
loss = empirical_distrust_loss(0.05, 7.0, 2.7)
print(loss)  # Should be ~200-250
```

```rust
// Rust
let loss = empirical_distrust_loss(0.05, 7.0, 2.7)?;
println!("{}", loss.item::<f32>());  // Should match Python
```

## Contributing

When contributing to the Rust implementation:

1. Match Python behavior exactly where possible
2. Add tests for any new functionality
3. Document API differences from Python
4. Update this guide with any discoveries about mlx-rs

## Questions?

- Check `/Users/arosboro/your_ai/` Python implementation as reference
- See `IMPLEMENTATION_NOTES.md` for detailed component list
- Review Python tests for expected behavior

