# Benchmark Improvements - Logging and Safety for Large Models

## Summary

Enhanced the `your_ai benchmark` command with comprehensive logging, memory estimation, and automatic configuration tuning to safely test large models (including 70B) without causing system crashes.

## What Was Added

### 1. Persistent Benchmark Logging (`benchmark_log.jsonl`)

Every benchmark run now writes a detailed JSON log to `benchmark_log.jsonl` in the current directory. This log persists even if the system crashes, providing a trail for debugging.

**Log Events:**
- `benchmark_start`: When benchmark begins
- `model_start`: When testing a model begins
- `preflight_check`: Memory estimation and config selection
- `subprocess_start`: When subprocess is spawned
- `subprocess_spawned`: Confirmation subprocess started with PID
- `subprocess_completed`: Subprocess finished successfully
- `subprocess_timeout`: Subprocess exceeded 5-minute timeout
- `subprocess_failed`: Subprocess failed with errors
- `safety_stop`: Benchmark stopped due to low memory
- `benchmark_complete`: Final summary

**Example Log Entry:**
```json
{
  "timestamp": 1702143820.5,
  "event": "preflight_check",
  "preset": "hermes-70b",
  "available_gb": 17.5,
  "estimated_base_gb": 128.0,
  "estimated_conservative_gb": 156.0,
  "batch_size": 1,
  "lora_rank": 16,
  "lora_layers": 8
}
```

### 2. Memory Estimation (`estimate_training_memory`)

**Location:** `rust/src/hardware/profiles.rs`

Estimates memory requirements based on model parameter count:
- **7B models:** ~14-16 GB (base-conservative)
- **14B models:** ~27-32 GB (base-conservative)
- **70B models:** ~128-156 GB (base-conservative)

Formula accounts for:
- Quantized model weights (4-bit)
- LoRA adapters
- Optimizer states
- Activation memory (batch-dependent)
- System overhead (~2GB)

### 3. Auto-Configuration (`get_safe_benchmark_config`)

**Location:** `rust/src/hardware/profiles.rs`

Automatically selects safe configuration based on model size and available memory:

| Model Size | Available Memory | Batch | Rank | Layers |
|------------|------------------|-------|------|--------|
| 70B        | < 40 GB          | 1     | 16   | 8      |
| 70B        | 40-60 GB         | 1     | 24   | 12     |
| 70B        | > 60 GB          | 1     | 32   | 16     |
| 14B        | < 20 GB          | 1     | 32   | 12     |
| 14B        | > 20 GB          | 2     | 48   | 16     |
| 7-8B       | Any              | 2     | 64   | 16     |

### 4. Enhanced Subprocess Handling

**Location:** `rust/src/cli/commands.rs`

- **Timeout:** 5-minute limit per model test (prevents hanging)
- **Output Capture:** Pipes stdout/stderr for logging even on crash
- **Non-blocking Wait:** Polls subprocess status every 100ms
- **Graceful Termination:** Kills process on timeout and logs result

## Usage

### Basic Benchmark (with safety checks)
```bash
./target/release/your_ai benchmark
```

Output:
```
Benchmark log: ./benchmark_log.jsonl

[1/5] hermes-mistral-7b (7B)
  Pre-flight: Available=17.5GB, Required=~14-16GB
  Config: batch=2, rank=64, layers=16
  Testing... ✓ Pass (12.3 GB peak)

[5/5] hermes-70b (70B)
  Pre-flight: Available=17.5GB, Required=~128-156GB ⚠
  ⚠️  WARNING: Available memory may be insufficient
  Config: batch=1, rank=16, layers=8
  Testing... ✗ OOM
```

### Force Mode (skip safety checks)
```bash
./target/release/your_ai benchmark --force
```

### Check Log After Crash
```bash
cat benchmark_log.jsonl | jq .
```

Example output after crash:
```json
{"timestamp": 1702143820.5, "event": "model_start", "preset": "hermes-70b"}
{"timestamp": 1702143821.2, "event": "subprocess_spawned", "pid": 12345}
{"timestamp": 1702143822.8, "event": "subprocess_completed", "stdout_preview": "Loading shard 1/29..."}
```

## Implementation Details

### Files Modified

1. **`rust/src/hardware/profiles.rs`** (+58 lines)
   - Added `estimate_training_memory()` function
   - Added `get_safe_benchmark_config()` function

2. **`rust/src/benchmarks/optimizer.rs`** (~5 lines changed)
   - Updated `quick_validate()` to accept `params_str` parameter
   - Uses safe config based on model size

3. **`rust/src/cli/commands.rs`** (~200 lines added/modified)
   - Added `BenchmarkLogger` struct (27 lines)
   - Enhanced `benchmark_single_model()` to accept params
   - Updated `benchmark()` main loop with:
     - Logger initialization and event logging
     - Pre-flight memory checks with warnings
     - Subprocess timeout and output capture
     - Comprehensive error handling and logging

### Safety Features

1. **Pre-flight Warnings:** Shows if available memory is below estimated requirements
2. **Safety Stop:** Stops benchmark if available < 2GB (unless `--force`)
3. **Timeout Protection:** Kills runaway processes after 5 minutes
4. **Persistent Logging:** Crash analysis via `benchmark_log.jsonl`

## Testing on 96GB M3 Ultra

Your system has:
- **Total:** 96 GB
- **Available:** ~17.5 GB (at benchmark start)

Expected results:
- ✅ **7-8B models:** Should pass easily (~12-14 GB peak)
- ✅ **14B models:** Should pass with warning (~18-25 GB peak)
- ⚠️  **70B models:** Will likely OOM with current available memory
  - Needs ~40+ GB available for safe operation
  - Auto-config will use minimal settings (batch=1, rank=16, layers=8)

To successfully benchmark 70B models:
1. Close other applications to free memory
2. Target ~40+ GB available before running
3. Use `--force` mode (accepts the risk)

## Next Steps

If crashes still occur, check `benchmark_log.jsonl`:
- Look for the last `event` before crash
- Check `subprocess_spawned` to confirm PID
- Review `stdout_preview` to see where model loading stopped
- Compare `available_gb` vs `estimated_conservative_gb`

The log file will help identify:
- If crash occurs during model loading (shards)
- If crash occurs during weight initialization
- If crash occurs during first training step
- Exact memory state when crash happened

