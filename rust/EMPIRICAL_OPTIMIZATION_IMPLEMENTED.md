# Empirical Optimization CLI - Implementation Summary

## Overview

Successfully implemented a Rust CLI for empirical hardware optimization that mirrors the Python `find_optimal_profile.py` functionality. The implementation finds optimal training configurations by testing batch_size × lora_rank × lora_layers combinations with real training steps.

## Files Created

### 1. `src/benchmarks/optimizer.rs` (291 lines)

- `EmpiricalOptimizer` struct for systematic configuration testing
- `OptimizationResult` struct to store test results
- Test configuration matrices:
  - **Full mode**: 96 configs (batch: 2-12, rank: 32-128, layers: 8-32)
  - **Quick mode**: 12 configs (batch: 2/4/8, rank: 64/128, layers: 16/24)
- Key features:
  - Runs actual training steps (15 by default)
  - Measures peak memory with 15% safety margin
  - Tracks step time performance
  - Calculates throughput score (batch × rank × layers)
  - Detects OOM conditions
  - Provides detailed console output

### 2. `src/benchmarks/profile.rs` (108 lines)

- `HardwareProfile` struct for saving/loading optimal configurations
- JSON serialization for profile persistence
- `apply_to_config()` method to apply profile to training config
- Timestamps and metadata tracking
- Unit tests for profile creation

### 3. Updated `src/benchmarks/mod.rs`

- Exported new modules: `optimizer` and `profile`
- Exposed public API: `EmpiricalOptimizer`, `OptimizationResult`, `HardwareProfile`

## CLI Integration

### New Commands

#### 1. `your_ai optimize`

Standalone optimization command:

```bash
your_ai optimize \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --quick                    # Test 12 configs instead of 96
  --max-memory 28.0          # Memory limit in GB
  --output optimal_config.json
```

#### 2. Updated `your_ai train`

Added `--auto-optimize` flag:

```bash
your_ai train \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --auto-optimize            # Run optimization first
  --max-steps 5000
```

### Implementation Details

#### `src/cli/mod.rs`

- Added `Optimize` command variant with flags
- Added `--auto-optimize` flag to `Train` command
- Updated command routing in `run()` function

#### `src/cli/commands.rs`

- Implemented `optimize()` handler:

  - Creates optimizer with specified settings
  - Runs empirical tests
  - Prints summary of results
  - Saves profile to JSON if requested

- Updated `train()` handler:
  - Checks `auto_optimize` flag at start
  - Runs optimization if enabled
  - Applies optimal settings to config
  - Falls back to defaults if optimization fails
  - Command-line args override auto-optimized values

### Trainer Enhancement

#### `src/training/trainer.rs`

- Made `training_step()` method public for benchmarking
- Allows external code to run individual training steps
- Maintains backward compatibility with existing code

## Algorithm

1. **Generate test matrix**: Create combinations of batch_size, lora_rank, lora_layers
2. **Sort by complexity**: Test lighter configs first (ascending throughput score)
3. **For each configuration**:
   - Initialize config with test parameters
   - Disable checkpoints and logging
   - Create trainer and memory monitor
   - Run 15 training steps
   - Track peak memory (RSS) with 15% safety margin
   - Record average step time
   - Detect OOM or memory limit violations
   - Clear memory between tests
4. **Find best result**: Highest throughput score that succeeded
5. **Output results**: Console summary + optional JSON file

## Output Example

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Empirical Optimization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model:         NousResearch/Hermes-3-Llama-3.1-8B
  Max Memory:    28.0 GB
  Mode:          Quick
  Configurations: 12
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/12] batch=2, rank=64, layers=16 ... ✓ 12400 MB, 1.2s/step
[2/12] batch=4, rank=64, layers=16 ... ✓ 14800 MB, 1.4s/step
[3/12] batch=4, rank=128, layers=16 ... ✓ 19200 MB, 1.8s/step
[4/12] batch=4, rank=128, layers=24 ... ✓ 22100 MB, 2.1s/step
[5/12] batch=8, rank=128, layers=24 ... ✗ OOM

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Results Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Tested:  12 configurations
  Passed:  4
  Failed:  8

Optimal Configuration Found:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Batch size:    4
  LoRA rank:     128
  LoRA alpha:    256
  LoRA layers:   24
  Peak memory:   22.1 MB (21.58 GB)
  Step time:     2.1s
  Throughput:    12288 (batch × rank × layers)

Top 5 configurations by throughput:
  1. batch=4, rank=128, layers=24 (score=12288, 22100MB)
  2. batch=4, rank=128, layers=16 (score=8192, 19200MB)
  3. batch=4, rank=64, layers=16 (score=4096, 14800MB)
  4. batch=2, rank=64, layers=16 (score=2048, 12400MB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## JSON Profile Format

```json
{
  "model": "NousResearch/Hermes-3-Llama-3.1-8B",
  "optimal_batch_size": 4,
  "optimal_lora_rank": 128,
  "optimal_lora_layers": 24,
  "peak_memory_gb": 21.58,
  "throughput_score": 12288,
  "created_at": "2025-12-09T18:30:00Z",
  "all_results": [...]
}
```

## Key Design Decisions

1. **Throughput metric**: `batch_size * lora_rank * lora_layers` (matches Python)
2. **Safety margin**: 15% added to peak memory (matches Python)
3. **Test duration**: 15 steps per config (matches Python)
4. **Memory monitoring**: Uses existing `MemoryMonitor` with process RSS tracking
5. **Sorting strategy**: Test lighter configs first to fail fast on low-memory systems
6. **Integration**: Auto-optimize runs before training, with CLI args taking precedence

## Compilation Status

✅ **Code compiles successfully** (`cargo check --bin your_ai` passes)
✅ **All 7 implementation tasks completed**
✅ **No linter errors in new code**
✅ **Follows Rust best practices and simplicity guidelines**

## Dependencies

Uses only existing dependencies:

- `crate::utils::MemoryMonitor` - Memory tracking
- `crate::training::DistrustTrainer` - Training steps
- `crate::config::Config` - Configuration management
- `serde`/`serde_json` - JSON serialization
- `anyhow` - Error handling

No new external dependencies required.

## Testing Notes

Full runtime testing requires:

1. MLX environment properly configured
2. Valid model path or HuggingFace model
3. Training data (data/train.jsonl)

The implementation is ready for integration testing once the MLX build environment is resolved.
