# Rust Empirical Benchmark CLI Plan

## Overview

Create a Rust CLI that empirically tests training configurations to find optimal settings for the user's hardware, mirroring the Python `find_optimal_profile.py` approach but with native Metal GPU integration.

## Goals

1. **Empirically test** batch_size × lora_rank × lora_layers combinations
2. **Measure peak memory** using the new MemoryMonitor module
3. **Find optimal throughput** without causing OOM
4. **Save configuration profiles** to JSON for reuse
5. **Integrate with training** via `--auto-optimize` flag

## CLI Commands

### Standalone Command
```bash
your_ai optimize \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --quick                    # Test common configs only
  --max-memory 28.0          # Memory limit in GB
  --output optimal_config.json
```

### Integrated with Training
```bash
your_ai train \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --auto-optimize            # Run optimization first
  --max-steps 5000
```

## Implementation Files

### 1. New Module: [`src/benchmarks/optimizer.rs`](../src/benchmarks/optimizer.rs)

Core empirical optimization logic:

```rust
pub struct OptimizationResult {
    pub batch_size: usize,
    pub lora_rank: usize,
    pub lora_layers: usize,
    pub peak_memory_mb: f64,
    pub step_time_ms: f64,
    pub throughput_score: f64,  // batch_size * lora_rank * lora_layers
    pub success: bool,
    pub error: Option<String>,
}

pub struct EmpiricalOptimizer {
    model_path: String,
    max_memory_gb: f64,
    test_steps: usize,
    memory_monitor: MemoryMonitor,
}

impl EmpiricalOptimizer {
    // Test configurations systematically
    pub fn find_optimal(&mut self) -> Vec<OptimizationResult>;

    // Test a single configuration
    fn test_config(&mut self, batch: usize, rank: usize, layers: usize) -> OptimizationResult;

    // Run actual training steps
    fn run_training_test(&mut self, config: &Config) -> Result<(f64, f64), Error>;
}
```

### 2. Update: [`src/cli/mod.rs`](../src/cli/mod.rs)

Add new CLI command:

```rust
#[derive(Subcommand)]
enum Commands {
    // ... existing commands ...

    /// Find optimal training configuration for your hardware
    Optimize {
        #[arg(long)]
        model: String,

        #[arg(long)]
        max_memory: Option<f64>,

        #[arg(long)]
        quick: bool,

        #[arg(long)]
        output: Option<String>,
    },
}
```

### 3. Update: [`src/cli/commands.rs`](../src/cli/commands.rs)

Add optimize command handler and `--auto-optimize` to train:

```rust
pub fn optimize(
    model: String,
    max_memory: Option<f64>,
    quick: bool,
    output: Option<String>,
) -> Result<()>;

// Update train to support auto-optimize
pub fn train(
    model: String,
    // ... existing params ...
    auto_optimize: bool,
) -> Result<()>;
```

### 4. New File: [`src/benchmarks/profile.rs`](../src/benchmarks/profile.rs)

Configuration profile management:

```rust
#[derive(Serialize, Deserialize)]
pub struct HardwareProfile {
    pub model: String,
    pub optimal_batch_size: usize,
    pub optimal_lora_rank: usize,
    pub optimal_lora_layers: usize,
    pub peak_memory_gb: f64,
    pub throughput_score: f64,
    pub created_at: String,
    pub all_results: Vec<OptimizationResult>,
}

impl HardwareProfile {
    pub fn save(&self, path: &str) -> Result<()>;
    pub fn load(path: &str) -> Result<Self>;
}
```

## Test Configurations

### Full Mode (default)
| Batch Size | LoRA Rank | LoRA Layers |
|-----------|-----------|-------------|
| 1, 2, 4, 6, 8 | 32, 64, 128, 256 | 8, 16, 24, 32 |

Total: 5 × 4 × 4 = **80 configurations**

### Quick Mode (`--quick`)
| Batch Size | LoRA Rank | LoRA Layers |
|-----------|-----------|-------------|
| 1, 2, 4 | 64, 128 | 16, 32 |

Total: 3 × 2 × 2 = **12 configurations**

## Algorithm

```
1. Load model architecture (not weights) to estimate base memory
2. For each config in test_matrix:
   a. Check if config would exceed memory limit (early skip)
   b. Initialize minimal training setup
   c. Run 10-15 training steps with real data
   d. Record peak memory (with 15% safety margin)
   e. Record average step time
   f. Clear GPU cache between tests
3. Sort results by throughput_score descending
4. Return highest throughput config that succeeded
5. Save profile to JSON
```

## Output Format

### Console Output
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Empirical Optimization Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Testing 12 configurations for: Hermes-3-Llama-3.1-8B
Memory limit: 28.0 GB

[1/12] batch=1, rank=64, layers=16 ... ✓ 12.4 GB, 1.2s/step
[2/12] batch=2, rank=64, layers=16 ... ✓ 14.8 GB, 1.4s/step
[3/12] batch=4, rank=64, layers=16 ... ✓ 19.2 GB, 1.8s/step
[4/12] batch=4, rank=128, layers=16 ... ✓ 22.1 GB, 2.1s/step
[5/12] batch=4, rank=128, layers=32 ... ✓ 26.8 GB, 2.4s/step
[6/12] batch=4, rank=256, layers=32 ... ✗ OOM at 32.1 GB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Optimal Configuration Found
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Batch size:    4
  LoRA rank:     128
  LoRA layers:   32
  Peak memory:   26.8 GB
  Throughput:    16,384 (4 × 128 × 32)

Saved to: optimal_config.json
```

### JSON Output (`optimal_config.json`)
```json
{
  "model": "NousResearch/Hermes-3-Llama-3.1-8B",
  "optimal_batch_size": 4,
  "optimal_lora_rank": 128,
  "optimal_lora_layers": 32,
  "peak_memory_gb": 26.8,
  "throughput_score": 16384,
  "created_at": "2025-12-09T18:30:00Z",
  "all_results": [...]
}
```

## Integration with Training

When `--auto-optimize` is passed to train:

```rust
if auto_optimize {
    println!("Running optimization to find best config...");
    let profile = EmpiricalOptimizer::new(&model, max_memory)
        .find_optimal()?;

    config.training.batch_size = profile.optimal_batch_size;
    config.model.lora_rank = profile.optimal_lora_rank;
    config.model.lora_num_layers = profile.optimal_lora_layers;

    println!("Using optimized config: batch={}, rank={}, layers={}",
        profile.optimal_batch_size,
        profile.optimal_lora_rank,
        profile.optimal_lora_layers);
}
```

## Implementation Tasks

1. **Create `src/benchmarks/optimizer.rs`** - Core optimization logic
2. **Create `src/benchmarks/profile.rs`** - Profile save/load
3. **Update `src/benchmarks/mod.rs`** - Export new modules
4. **Add `Optimize` command to CLI** - `src/cli/mod.rs`
5. **Implement `optimize()` handler** - `src/cli/commands.rs`
6. **Add `--auto-optimize` to train** - `src/cli/mod.rs` and commands.rs
7. **Test with 8B model** - Verify memory tracking works
8. **Create documentation** - Update MEMORY_SAFE_TRAINING.md

## Dependencies

Uses existing modules:
- `crate::utils::MemoryMonitor` - Memory tracking
- `crate::training::DistrustTrainer` - Training steps
- `crate::model::LlamaConfig` - Model config estimation

No new dependencies needed.

## Success Criteria

- [ ] `your_ai optimize` finds optimal config without crashing
- [ ] Results match Python script's recommendations (within 10%)
- [ ] `--auto-optimize` applies settings to training
- [ ] JSON output is valid and loadable
- [ ] Memory stays within specified limits

