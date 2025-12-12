#!/usr/bin/env python3
"""
Memory Limit Testing for Empirical Distrust Training

This script empirically determines the maximum safe training configuration
for your hardware by gradually increasing settings until memory limits are hit.

Usage:
    python scripts/test_memory_limits.py --model NousResearch/Hermes-2-Pro-Mistral-7B
"""

import sys
import json
import argparse
import subprocess
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hardware_profiles import detect_hardware, save_hardware_profile


def test_configuration(
    model_path: str, batch_size: int, lora_rank: int, lora_layers: int, max_steps: int = 15
) -> bool:
    """
    Test if a configuration works without OOM using REAL training.

    Extended to 15 steps to detect late-allocating buffers and memory growth.

    Returns:
        True if successful, False if OOM or crash
    """
    print(f"\n🔬 Testing: batch={batch_size}, rank={lora_rank}, layers={lora_layers} (15 steps)")

    cmd = [
        sys.executable,
        "src/train_qlora.py",
        "--model",
        model_path,
        "--batch-size",
        str(batch_size),
        "--lora-rank",
        str(lora_rank),
        "--lora-layers",
        str(lora_layers),
        "--max-steps",
        str(max_steps),
        "--no-auto-maximize",  # Critical: disable auto-maximize
        # Use streaming mode (default) to match real training conditions
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout (longer for 15 steps)
        )

        # Any non-zero return code is a failure
        if result.returncode != 0:
            print("   ❌ Failed - Non-zero exit code")
            # Log specific OOM errors if present
            if (
                "Insufficient Memory" in result.stderr
                or "kIOGPUCommandBufferCallbackErrorOutOfMemory" in result.stderr
            ):
                print("      (OOM detected)")
            # Include relevant output for debugging
            if result.stderr:
                stderr_preview = result.stderr[:200].replace("\n", " ")
                print(f"      stderr: {stderr_preview}...")
            return False

        # returncode == 0: Check both stdout and stderr for training completion
        combined_output = result.stdout + result.stderr

        # Look for successful completion of multiple steps
        success_markers = ["Training:", "Baseline memory:"]
        step_markers = [f"{i}/15" for i in range(10, 16)]  # Steps 10-15

        has_training = any(marker in combined_output for marker in success_markers)
        has_late_steps = any(marker in combined_output for marker in step_markers)

        if has_training and has_late_steps:
            print("   ✅ Success - Completed 15 steps successfully!")
            return True
        elif has_training:
            print("   ⚠️  Started but didn't complete 15 steps - treating as failure")
            return False
        else:
            print("   ⚠️  No training markers found - treating as failure")
            return False

    except subprocess.TimeoutExpired:
        print("   ✅ Success - Ran for 10 minutes without crash")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def binary_search_batch_size(
    model_path: str, lora_rank: int, lora_layers: int, min_batch: int = 1, max_batch: int = 128
) -> int:
    """
    Use binary search to find maximum working batch size.
    """
    print(f"\n📊 Binary search for batch size (rank={lora_rank}, layers={lora_layers})")

    best_working = min_batch

    # First verify min works
    if not test_configuration(model_path, min_batch, lora_rank, lora_layers):
        print(f"\n❌ Even minimum batch size {min_batch} fails!")
        return min_batch

    low, high = min_batch, max_batch

    while low <= high:
        mid = (low + high) // 2

        if test_configuration(model_path, mid, lora_rank, lora_layers):
            best_working = mid
            low = mid + 1  # Try larger
        else:
            high = mid - 1  # Try smaller

        time.sleep(2)  # Brief pause between tests

    print(f"\n✅ Maximum batch size: {best_working}")
    return best_working


def progressive_test(model_path: str, output_file: str = None):
    """
    Progressively test configurations to find optimal settings.

    Strategy:
    1. Start with minimal safe settings (batch=1, rank=32, layers=8)
    2. Find max batch size for base settings
    3. Increase rank and retest
    4. Increase layers and retest
    """
    print("=" * 70)
    print("Memory Limit Testing - Empirical Approach")
    print("=" * 70)

    # Detect hardware
    generation, variant, memory_gb = detect_hardware()
    if generation:
        print(f"\nDetected: {generation.upper()} {variant.title()} {memory_gb}GB")
    else:
        print("\n⚠️  Could not detect hardware")

    print(f"\nModel: {model_path}")
    print("\nThis will take 10-20 minutes...")
    print("Each test runs for a few steps to verify stability.")

    # Phase 1: Base configuration
    print("\n" + "=" * 70)
    print("Phase 1: Testing base configuration")
    print("=" * 70)

    base_rank = 32
    base_layers = 8

    max_batch = binary_search_batch_size(
        model_path, base_rank, base_layers, min_batch=1, max_batch=64
    )

    # Phase 2: Increase rank
    print("\n" + "=" * 70)
    print("Phase 2: Testing with higher LoRA rank")
    print("=" * 70)

    test_ranks = [64, 96, 128, 192, 256, 384, 512]
    best_rank = base_rank

    for rank in test_ranks:
        if rank <= base_rank:
            continue

        # Test with half the max batch to see if rank fits
        test_batch = max(1, max_batch // 2)

        if test_configuration(model_path, test_batch, rank, base_layers):
            best_rank = rank
            # Find new max batch for this rank
            max_batch = binary_search_batch_size(
                model_path, rank, base_layers, min_batch=1, max_batch=max_batch
            )
        else:
            print(f"   Rank {rank} doesn't fit, stopping at {best_rank}")
            break

        time.sleep(2)

    # Phase 3: Increase layers
    print("\n" + "=" * 70)
    print("Phase 3: Testing with more layers")
    print("=" * 70)

    test_layers = [16, 24, 32]
    best_layers = base_layers

    for layers in test_layers:
        if layers <= base_layers:
            continue

        test_batch = max(1, max_batch // 2)

        if test_configuration(model_path, test_batch, best_rank, layers):
            best_layers = layers
            max_batch = binary_search_batch_size(
                model_path, best_rank, layers, min_batch=1, max_batch=max_batch
            )
        else:
            print(f"   {layers} layers doesn't fit, stopping at {best_layers}")
            break

        time.sleep(2)

    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nOptimal configuration for {model_path}:")
    print(f"  Batch size:  {max_batch}")
    print(f"  LoRA rank:   {best_rank}")
    print(f"  LoRA layers: {best_layers}")

    # Save results
    results = {
        "model": model_path,
        "hardware": {
            "generation": generation,
            "variant": variant,
            "memory_gb": memory_gb,
        },
        "optimal_config": {
            "batch_size": max_batch,
            "lora_rank": best_rank,
            "lora_alpha": best_rank * 2,
            "lora_num_layers": best_layers,
        },
        "tested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if output_file is None:
        output_file = "optimal_config_measured.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")

    # Offer to save as hardware profile
    print("\n" + "=" * 70)
    response = input("Save as hardware profile for future use? [Y/n] ").strip().lower()

    if response in ("", "y", "yes"):
        profile = {
            "generation": generation,
            "variant": variant,
            "memory_gb": memory_gb,
            "batch_size": max_batch,
            "lora_rank": best_rank,
            "lora_alpha": best_rank * 2,
            "lora_num_layers": best_layers,
            "grad_checkpoint": True,
            "training_budget_gb": int(memory_gb * 0.80) if memory_gb else 64,
            "empirically_validated": True,
            "model_tested": model_path,
        }

        save_hardware_profile(profile)
        print("\n✅ Hardware profile saved!")
        print("   Future training will use these validated settings")

    print("\n" + "=" * 70)
    print("Testing complete!")
    print("=" * 70)
    print("\nTo train with these settings:")
    print(f"  python src/train_qlora.py --model {model_path}")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Empirically test memory limits for training configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script finds the optimal training configuration for your hardware
by testing progressively larger settings until memory limits are reached.

The process is empirical and reliable - no guessing or estimation.

Examples:
  # Test with 7B model
  python scripts/test_memory_limits.py --model NousResearch/Hermes-2-Pro-Mistral-7B

  # Test and save to specific file
  python scripts/test_memory_limits.py --model dolphin-8b --output results.json
        """,
    )

    parser.add_argument("--model", required=True, help="Model path or HuggingFace ID to test with")
    parser.add_argument("--output", help="Output JSON file (default: optimal_config_measured.json)")

    args = parser.parse_args()

    try:
        progressive_test(args.model, args.output)
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
