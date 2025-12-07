#!/usr/bin/env python3
"""
Empirical Batch Size Testing Script

Tests incrementally larger batch sizes to find the maximum that doesn't cause OOM.
Run this manually on your hardware to determine safe batch_size values.

Usage:
    python scripts/test_batch_sizes.py --model NousResearch/Hermes-2-Pro-Mistral-7B
    python scripts/test_batch_sizes.py --model NousResearch/Hermes-2-Pro-Mistral-7B --start 1 --max 16
"""

import argparse
import gc
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_batch_size(model_path: str, batch_size: int, lora_rank: int, lora_layers: int) -> bool:
    """
    Test if a given batch size works without OOM.

    Returns True if successful, False if OOM or other error.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.tuner import linear_to_lora_layers

    try:
        # Set memory limit
        if mx.metal.is_available():
            mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])

        print("  Loading model...", end=" ", flush=True)
        model, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True})

        # Freeze base model
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
        print("✓")

        # Create dummy batch
        seq_length = 1024
        input_ids = mx.zeros((batch_size, seq_length), dtype=mx.int32)

        print(
            f"  Running forward pass (batch={batch_size}, seq={seq_length})...", end=" ", flush=True
        )

        # Forward pass
        logits = model(input_ids)
        mx.eval(logits)
        print("✓")

        print("  Running backward pass...", end=" ", flush=True)

        # Compute loss and gradients
        def loss_fn(model, x):
            logits = model(x)
            # Simple loss: mean of logits (just to trigger backward)
            return mx.mean(logits)

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad(model, input_ids)
        mx.eval(loss, grads)
        print("✓")

        # Cleanup
        del model, tokenizer, logits, loss, grads
        gc.collect()

        return True

    except Exception as e:
        error_msg = str(e)
        if "OutOfMemory" in error_msg or "Insufficient Memory" in error_msg:
            print("✗ OOM")
        else:
            print(f"✗ Error: {error_msg}")

        # Cleanup on error
        gc.collect()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Empirically test maximum safe batch sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="NousResearch/Hermes-2-Pro-Mistral-7B",
        help="Model to test (default: Hermes-2-Pro-Mistral-7B)",
    )
    parser.add_argument("--start", type=int, default=1, help="Starting batch size (default: 1)")
    parser.add_argument(
        "--max", type=int, default=16, help="Maximum batch size to test (default: 16)"
    )
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank to use (default: 64)")
    parser.add_argument(
        "--lora-layers", type=int, default=16, help="Number of LoRA layers (default: 16)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Empirical Batch Size Testing")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"LoRA rank:   {args.lora_rank}")
    print(f"LoRA layers: {args.lora_layers}")
    print(f"Test range:  {args.start} to {args.max}")
    print("=" * 60)
    print()

    max_working = 0
    results = []

    for batch_size in range(args.start, args.max + 1):
        print(f"\nTesting batch_size={batch_size}:")

        success = test_batch_size(args.model, batch_size, args.lora_rank, args.lora_layers)

        results.append((batch_size, success))

        if success:
            max_working = batch_size
            print(f"  → batch_size={batch_size} WORKS")
        else:
            print(f"  → batch_size={batch_size} FAILED")
            # Stop on first failure for efficiency
            print("\nStopping at first failure.")
            break

    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    for batch_size, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  batch_size={batch_size}: {status}")
    print()
    print(f"Maximum safe batch_size: {max_working}")
    print()
    print("Recommended settings for your hardware profile:")
    print(
        f'  "small": {{"lora_rank": {args.lora_rank}, "lora_alpha": {args.lora_rank * 2}, "lora_num_layers": {args.lora_layers}, "batch_size": {max_working}}}'
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
