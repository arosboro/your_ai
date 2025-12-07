#!/usr/bin/env python3
"""
Safe Single Configuration Test

Tests ONE configuration in isolation. Run multiple times with different
parameters to find optimal settings. Each run is independent to prevent
memory accumulation.

Usage:
    python scripts/safe_profile_test.py --batch 4 --rank 64 --layers 16
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Test a single training configuration safely")
    parser.add_argument("--model", default="NousResearch/Hermes-2-Pro-Mistral-7B")
    parser.add_argument("--batch", type=int, required=True, help="Batch size to test")
    parser.add_argument("--rank", type=int, required=True, help="LoRA rank to test")
    parser.add_argument("--layers", type=int, required=True, help="LoRA layers to test")
    parser.add_argument("--steps", type=int, default=3, help="Training steps to run")
    args = parser.parse_args()

    print(f"Testing: batch={args.batch}, rank={args.rank}, layers={args.layers}")
    print("-" * 50)

    import mlx.core as mx
    import mlx.nn as nn
    import psutil
    from mlx_lm import load
    from mlx_lm.tuner import linear_to_lora_layers

    try:
        # Set memory limit
        if mx.metal.is_available():
            mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])

        print("Loading model...", flush=True)
        model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})
        model.freeze()

        print(f"Applying LoRA (rank={args.rank}, layers={args.layers})...", flush=True)
        lora_config = {
            "rank": args.rank,
            "scale": 2.0,
            "dropout": 0.0,
            "keys": [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
            ],
        }
        linear_to_lora_layers(model, num_layers=args.layers, config=lora_config)

        # Enable gradient checkpointing
        from train_qlora import grad_checkpoint

        grad_checkpoint(model.layers[0])
        print("Gradient checkpointing enabled")

        # Setup training
        seq_length = 1024
        input_ids = mx.zeros((args.batch, seq_length), dtype=mx.int32)

        def compute_loss(model, x):
            logits = model(x)
            labels = x[:, 1:]
            logits = logits[:, :-1, :]
            return nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="mean"
            )

        loss_and_grad = nn.value_and_grad(model, compute_loss)

        # Run training steps
        process = psutil.Process(os.getpid())
        step_times = []

        print(f"Running {args.steps} training steps...", flush=True)
        for step in range(args.steps):
            start = time.time()
            loss, grads = loss_and_grad(model, input_ids)
            mx.eval(loss, grads)
            elapsed = time.time() - start
            step_times.append(elapsed)
            mem_mb = process.memory_info().rss / 1024 / 1024
            print(
                f"  Step {step + 1}: loss={float(loss):.4f}, time={elapsed:.1f}s, mem={mem_mb:.0f}MB"
            )

        avg_time = sum(step_times) / len(step_times)
        final_mem = process.memory_info().rss / 1024 / 1024

        print("-" * 50)
        print("SUCCESS!")
        print(f"  Average step time: {avg_time:.1f}s")
        print(f"  Memory usage: {final_mem:.0f}MB")
        print(f"  Config: batch={args.batch}, rank={args.rank}, layers={args.layers}")

    except Exception as e:
        print("-" * 50)
        print(f"FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
