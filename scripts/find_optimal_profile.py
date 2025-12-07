#!/usr/bin/env python3
"""
Find Optimal Hardware Profile

Systematically tests training configurations to find the best settings
that maximize throughput without causing OOM.

Tests combinations of:
- batch_size: 1, 2, 4, 6, 8, 10, 12
- lora_rank: 32, 64, 96, 128
- lora_layers: 8, 16, 24, 32

Reports the configuration with maximum (batch_size * lora_rank * lora_layers)
that doesn't OOM, as this represents training capacity.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_config(
    model_path: str,
    batch_size: int,
    lora_rank: int,
    lora_layers: int,
    train_file: str = "data/train.jsonl",
    steps: int = 15,
) -> dict:
    """
    Test a specific configuration using REAL training conditions.

    Returns dict with success status, memory usage, and timing.
    """
    import mlx.core as mx
    import psutil
    from config import Config
    from train_qlora import DistrustTrainer

    result = {
        "batch_size": batch_size,
        "lora_rank": lora_rank,
        "lora_layers": lora_layers,
        "success": False,
        "memory_mb": 0,
        "step_time_s": 0,
        "error": None,
    }

    process = psutil.Process(os.getpid())

    try:
        # Clear previous state
        gc.collect()
        mx.metal.clear_cache()

        # Create config for testing with REAL training setup
        config = Config()
        config.paths.model_path = model_path
        config.paths.train_file = train_file
        config.training.batch_size = batch_size
        config.training.max_steps = steps
        config.training.save_steps = 999999  # Don't save during test
        config.training.eval_steps = 999999  # Don't eval during test
        config.training.logging_steps = 999999  # Don't log during test
        config.model.lora_rank = lora_rank
        config.model.lora_alpha = lora_rank * 2
        config.model.lora_num_layers = lora_layers
        config.performance.checkpoint_enabled = False
        config.performance.tensorboard_enabled = False
        config.performance.use_streaming = False  # Load data for consistency

        # Initialize trainer with REAL training setup
        trainer = DistrustTrainer(config)

        # Load REAL training data
        train_data = trainer.load_data(config.paths.train_file)
        if not train_data:
            raise ValueError("No training data loaded")

        num_samples = min(len(train_data), batch_size * steps * 2)

        # Run REAL training steps
        step_times = []
        peak_memory = 0

        for step in range(1, steps + 1):
            # Get real batch
            start_idx = ((step - 1) * batch_size) % num_samples
            end_idx = min(start_idx + batch_size, num_samples)
            batch_examples = [train_data[i] for i in range(start_idx, end_idx)]

            # Run actual training step with distrust loss
            batch = trainer.prepare_batch(batch_examples)
            start_time = time.time()
            metrics = trainer.train_step(batch)
            step_times.append(time.time() - start_time)

            # Force evaluation
            mx.eval(trainer.model.parameters())
            mx.eval(trainer.optimizer.state)

            # Track peak memory
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)

        # Add 15% safety margin for real training overhead
        result["memory_mb"] = peak_memory * 1.15
        result["step_time_s"] = sum(step_times) / len(step_times)
        result["success"] = True

        # Cleanup
        del trainer
        gc.collect()
        mx.metal.clear_cache()

    except Exception as e:
        error_msg = str(e)
        if "OutOfMemory" in error_msg or "Insufficient Memory" in error_msg:
            result["error"] = "OOM"
        else:
            result["error"] = error_msg[:100]

        gc.collect()
        mx.metal.clear_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Find optimal training profile")
    parser.add_argument(
        "--model",
        default="NousResearch/Hermes-2-Pro-Mistral-7B",
        help="Model to test",
    )
    parser.add_argument(
        "--train-file",
        default="data/train.jsonl",
        help="Training data file (default: data/train.jsonl)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with fewer configurations",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    # Check if training data exists
    if not Path(args.train_file).exists():
        print(f"❌ Training file not found: {args.train_file}")
        print(f"   Please run data preparation first:")
        print(f"   python src/prepare_data_curated.py --input data/raw --output data")
        sys.exit(1)

    print("=" * 70)
    print("Finding Optimal Hardware Profile - REAL Training Conditions")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Training data: {args.train_file}")
    print("Testing with: Real data, distrust loss, full optimizer")
    print("=" * 70)
    print()

    # Define test configurations
    if args.quick:
        batch_sizes = [2, 4, 8]
        lora_ranks = [64, 128]
        lora_layers_list = [16, 24]
    else:
        batch_sizes = [2, 4, 6, 8, 10, 12]
        lora_ranks = [32, 64, 96, 128]
        lora_layers_list = [8, 16, 24, 32]

    results = []
    best_result = None
    best_score = 0

    # Test configurations in order of increasing resource usage
    configs = []
    for batch_size in batch_sizes:
        for lora_rank in lora_ranks:
            for lora_layers in lora_layers_list:
                score = batch_size * lora_rank * lora_layers
                configs.append((score, batch_size, lora_rank, lora_layers))

    # Sort by score to test lighter configs first
    configs.sort()

    total = len(configs)
    for i, (score, batch_size, lora_rank, lora_layers) in enumerate(configs, 1):
        print(
            f"[{i}/{total}] Testing batch={batch_size}, rank={lora_rank}, layers={lora_layers}...",
            end=" ",
            flush=True,
        )

        result = test_config(args.model, batch_size, lora_rank, lora_layers, args.train_file)
        results.append(result)

        if result["success"]:
            print(f"✓ ({result['memory_mb']:.0f}MB, {result['step_time_s']:.1f}s/step)")

            # Track best configuration by score
            if score > best_score:
                best_score = score
                best_result = result
        else:
            print(f"✗ {result['error']}")

    # Print summary
    print()
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Tested: {len(results)} configurations")
    print(f"Passed: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print()

    if best_result:
        print("OPTIMAL CONFIGURATION:")
        print(f"  batch_size:   {best_result['batch_size']}")
        print(f"  lora_rank:    {best_result['lora_rank']}")
        print(f"  lora_alpha:   {best_result['lora_rank'] * 2}")
        print(f"  lora_layers:  {best_result['lora_layers']}")
        print(f"  memory_mb:    {best_result['memory_mb']:.0f}")
        print(f"  step_time:    {best_result['step_time_s']:.1f}s")
        print()
        print("Recommended profile update:")
        print(
            f'  "small": {{"lora_rank": {best_result["lora_rank"]}, "lora_alpha": {best_result["lora_rank"] * 2}, "lora_num_layers": {best_result["lora_layers"]}, "batch_size": {best_result["batch_size"]}}}'
        )
        print()

        # Also show top 5 by different metrics
        print("Top configurations by throughput (batch * rank * layers):")
        successful_sorted = sorted(
            successful,
            key=lambda r: r["batch_size"] * r["lora_rank"] * r["lora_layers"],
            reverse=True,
        )
        for i, r in enumerate(successful_sorted[:5], 1):
            score = r["batch_size"] * r["lora_rank"] * r["lora_layers"]
            print(
                f"  {i}. batch={r['batch_size']}, rank={r['lora_rank']}, layers={r['lora_layers']} (score={score}, {r['memory_mb']:.0f}MB)"
            )
    else:
        print("No successful configurations found!")

    print("=" * 70)

    # Save results if requested
    if args.output:
        output_data = {
            "model": args.model,
            "best": best_result,
            "all_results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
