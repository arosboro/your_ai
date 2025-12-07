#!/usr/bin/env python3
"""
Find Maximum Training Configuration

Incrementally increases settings until OOM/crash.
Saves progress after EACH successful test to a JSON file.

If the system crashes, check the JSON file for the last working config.

Usage:
    python scripts/find_max_config.py --model NousResearch/Hermes-2-Pro-Mistral-7B
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

RESULTS_FILE = Path(__file__).parent.parent / "optimal_config_results.json"

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def save_result(result: dict, all_results: list):
    """Save results to disk immediately after each test."""
    data = {
        "last_updated": datetime.now().isoformat(),
        "last_successful": result if result.get("success") else None,
        "all_tests": all_results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [Saved to {RESULTS_FILE}]")


def test_config(model_path: str, batch: int, rank: int, layers: int) -> dict:
    """Test a single configuration."""
    import mlx.core as mx
    import mlx.nn as nn
    import psutil
    from mlx_lm import load
    from mlx_lm.tuner import linear_to_lora_layers

    result = {
        "batch_size": batch,
        "lora_rank": rank,
        "lora_layers": layers,
        "lora_alpha": rank * 2,
        "success": False,
        "memory_mb": 0,
        "step_time_s": 0,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        if mx.metal.is_available():
            mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])

        model, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True})
        model.freeze()

        lora_config = {
            "rank": rank,
            "scale": 2.0,
            "dropout": 0.0,
            "keys": [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
            ],
        }
        linear_to_lora_layers(model, num_layers=layers, config=lora_config)

        # Gradient checkpointing
        from train_qlora import grad_checkpoint

        grad_checkpoint(model.layers[0])

        # Training setup
        seq_length = 1024
        input_ids = mx.zeros((batch, seq_length), dtype=mx.int32)

        def compute_loss(model, x):
            logits = model(x)
            labels = x[:, 1:]
            logits = logits[:, :-1, :]
            return nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="mean"
            )

        loss_and_grad = nn.value_and_grad(model, compute_loss)

        # Run 3 steps to ensure stability
        process = psutil.Process(os.getpid())
        times = []
        for _ in range(3):
            start = time.time()
            loss, grads = loss_and_grad(model, input_ids)
            mx.eval(loss, grads)
            times.append(time.time() - start)

        result["memory_mb"] = process.memory_info().rss / 1024 / 1024
        result["step_time_s"] = sum(times) / len(times)
        result["success"] = True

        del model, tokenizer, loss, grads
        gc.collect()

    except Exception as e:
        result["error"] = str(e)[:200]
        gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="NousResearch/Hermes-2-Pro-Mistral-7B")
    parser.add_argument("--start-batch", type=int, default=4, help="Starting batch size")
    parser.add_argument("--start-rank", type=int, default=64, help="Starting LoRA rank")
    parser.add_argument("--start-layers", type=int, default=16, help="Starting LoRA layers")
    args = parser.parse_args()

    print("=" * 60)
    print("Finding Maximum Training Configuration")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Results file: {RESULTS_FILE}")
    print(
        f"Starting from: batch={args.start_batch}, rank={args.start_rank}, layers={args.start_layers}"
    )
    print()
    print("Will test increasing batch sizes first, then rank/layers.")
    print("Progress saved after EACH test. Check JSON if system crashes.")
    print("=" * 60)
    print()

    all_results = []
    best = None

    # Phase 1: Find max batch size with starting rank/layers
    print("PHASE 1: Finding maximum batch size")
    print("-" * 40)
    batch = args.start_batch
    rank = args.start_rank
    layers = args.start_layers

    while batch <= 32:  # Reasonable upper limit
        print(f"Testing batch={batch}, rank={rank}, layers={layers}...", end=" ", flush=True)
        result = test_config(args.model, batch, rank, layers)
        all_results.append(result)

        if result["success"]:
            print(f"✓ ({result['memory_mb']:.0f}MB, {result['step_time_s']:.1f}s)")
            best = result.copy()
            save_result(best, all_results)
            batch += 2  # Increment by 2
        else:
            print(f"✗ {result.get('error', 'Failed')[:50]}")
            save_result(best, all_results)
            batch -= 2  # Go back to last working
            break

    max_batch = batch if batch <= 32 else batch - 2
    print(f"\nMax batch size found: {max_batch}")
    print()

    # Phase 2: Try increasing rank with max batch
    print("PHASE 2: Finding maximum LoRA rank")
    print("-" * 40)
    batch = max_batch
    for rank in [64, 96, 128, 192, 256]:
        if rank <= args.start_rank:
            continue
        print(f"Testing batch={batch}, rank={rank}, layers={layers}...", end=" ", flush=True)
        result = test_config(args.model, batch, rank, layers)
        all_results.append(result)

        if result["success"]:
            print(f"✓ ({result['memory_mb']:.0f}MB, {result['step_time_s']:.1f}s)")
            best = result.copy()
            save_result(best, all_results)
        else:
            print(f"✗ {result.get('error', 'Failed')[:50]}")
            save_result(best, all_results)
            break

    max_rank = best["lora_rank"] if best else args.start_rank
    print(f"\nMax rank found: {max_rank}")
    print()

    # Phase 3: Try increasing layers
    print("PHASE 3: Finding maximum LoRA layers")
    print("-" * 40)
    batch = max_batch
    rank = max_rank
    for layers in [16, 24, 32]:
        if layers <= args.start_layers:
            continue
        print(f"Testing batch={batch}, rank={rank}, layers={layers}...", end=" ", flush=True)
        result = test_config(args.model, batch, rank, layers)
        all_results.append(result)

        if result["success"]:
            print(f"✓ ({result['memory_mb']:.0f}MB, {result['step_time_s']:.1f}s)")
            best = result.copy()
            save_result(best, all_results)
        else:
            print(f"✗ {result.get('error', 'Failed')[:50]}")
            save_result(best, all_results)
            break

    # Final summary
    print()
    print("=" * 60)
    print("OPTIMAL CONFIGURATION FOUND")
    print("=" * 60)
    if best:
        print(f"  batch_size:   {best['batch_size']}")
        print(f"  lora_rank:    {best['lora_rank']}")
        print(f"  lora_alpha:   {best['lora_alpha']}")
        print(f"  lora_layers:  {best['lora_layers']}")
        print(f"  memory_mb:    {best['memory_mb']:.0f}")
        print(f"  step_time:    {best['step_time_s']:.1f}s")
        print()
        print("Profile config line:")
        print(
            f'  "small": {{"lora_rank": {best["lora_rank"]}, "lora_alpha": {best["lora_alpha"]}, "lora_num_layers": {best["lora_layers"]}, "batch_size": {best["batch_size"]}}}'
        )
    print("=" * 60)
    print(f"Full results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
