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
    model_path: str, batch_size: int, lora_rank: int, lora_layers: int, steps: int = 2
) -> dict:
    """
    Test a specific configuration by running actual training steps.

    Returns dict with success status, memory usage, and timing.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.tuner import linear_to_lora_layers

    result = {
        "batch_size": batch_size,
        "lora_rank": lora_rank,
        "lora_layers": lora_layers,
        "success": False,
        "memory_mb": 0,
        "step_time_s": 0,
        "error": None,
    }

    try:
        # Set memory limit
        if mx.metal.is_available():
            mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])

        # Load model
        model, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True})
        model.freeze()

        # Apply LoRA
        lora_config = {
            "rank": lora_rank,
            "scale": 2.0,
            "dropout": 0.0,
            "keys": [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
            ],
        }
        linear_to_lora_layers(model, num_layers=lora_layers, config=lora_config)

        # Apply gradient checkpointing (always for memory efficiency)
        from train_qlora import grad_checkpoint

        grad_checkpoint(model.layers[0])

        # Create dummy batch (realistic sequence length)
        seq_length = 1024
        input_ids = mx.zeros((batch_size, seq_length), dtype=mx.int32)

        # Setup loss and gradient computation
        def compute_loss(model, input_ids):
            logits = model(input_ids)
            # Simulate next-token prediction loss
            labels = input_ids[:, 1:]
            logits = logits[:, :-1, :]
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="mean"
            )
            return loss

        loss_and_grad = nn.value_and_grad(model, compute_loss)

        # Run training steps
        import psutil
        import os

        process = psutil.Process(os.getpid())

        step_times = []
        for step in range(steps):
            start_time = time.time()

            loss, grads = loss_and_grad(model, input_ids)
            mx.eval(loss, grads)

            step_times.append(time.time() - start_time)

        # Record memory after training
        result["memory_mb"] = process.memory_info().rss / 1024 / 1024
        result["step_time_s"] = sum(step_times) / len(step_times)
        result["success"] = True

        # Cleanup
        del model, tokenizer, loss, grads
        gc.collect()

    except Exception as e:
        error_msg = str(e)
        if "OutOfMemory" in error_msg or "Insufficient Memory" in error_msg:
            result["error"] = "OOM"
        else:
            result["error"] = error_msg[:100]

        gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser(description="Find optimal training profile")
    parser.add_argument(
        "--model",
        default="NousResearch/Hermes-2-Pro-Mistral-7B",
        help="Model to test",
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

    print("=" * 70)
    print("Finding Optimal Hardware Profile")
    print("=" * 70)
    print(f"Model: {args.model}")
    print("Testing configurations with gradient checkpointing enabled...")
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

        result = test_config(args.model, batch_size, lora_rank, lora_layers)
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
