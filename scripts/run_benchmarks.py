#!/usr/bin/env python3
"""
Unified benchmark runner for external validation datasets.

This script runs external benchmarks (TruthfulQA, CensorBench, etc.)
alongside or instead of custom validation tests, providing standardized
evaluation metrics.

Usage:
    # Run TruthfulQA benchmark only
    python scripts/run_benchmarks.py --model "model-name" --benchmarks truthfulqa

    # Run multiple benchmarks
    python scripts/run_benchmarks.py --model "model-name" --benchmarks truthfulqa,censorbench

    # Run with custom tests for comparison
    python scripts/run_benchmarks.py --model "model-name" --benchmarks truthfulqa --include-custom

    # Quick test with limited samples
    python scripts/run_benchmarks.py --model "model-name" --benchmarks truthfulqa --max-samples 50
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add src and scripts directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_adapter import run_benchmark, get_adapter
from benchmark_config import (
    get_benchmark_config,
    get_priority_benchmarks,
    BENCHMARK_REGISTRY
)


def load_model(model_path: str, base_model: Optional[str] = None):
    """Load model using MLX or transformers."""
    print(f"\nLoading model: {model_path}")

    try:
        from mlx_lm import load

        print("Using MLX...")
        if base_model:
            print(f"Loading base model with adapters from: {model_path}")
            model, tokenizer = load(base_model, adapter_path=model_path)
        else:
            model, tokenizer = load(model_path)

        print("Model loaded successfully")
        return model, tokenizer

    except ImportError:
        print("MLX not available, trying transformers...")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            load_path = base_model if base_model else model_path
            print(f"Loading with transformers: {load_path}")

            tokenizer = AutoTokenizer.from_pretrained(load_path)
            model = AutoModelForCausalLM.from_pretrained(
                load_path,
                device_map="auto",
                load_in_4bit=True
            )

            print("Model loaded successfully")
            return model, tokenizer

        except Exception as e:
            print(f"Failed to load model: {e}")
            sys.exit(1)


def run_custom_tests(model, tokenizer) -> Dict:
    """Run the existing custom validation suite."""
    print("\n" + "=" * 60)
    print("RUNNING CUSTOM VALIDATION SUITE")
    print("=" * 60)

    try:
        # Import the validation runner from validate_model.py
        from scripts.validate_model import (
            test_ccp_censorship,
            test_western_censorship,
            test_authority_bias
        )
        from model_utils import generate_with_chat_template

        def generate_fn(model, tokenizer, prompt, max_tokens=200):
            return generate_with_chat_template(model, tokenizer, prompt, max_tokens)

        # Run custom tests
        ccp_results = test_ccp_censorship(model, tokenizer, generate_fn)
        western_results = test_western_censorship(model, tokenizer, generate_fn)
        authority_results = test_authority_bias(model, tokenizer, generate_fn)

        return {
            "custom_tests": {
                "ccp_censorship": ccp_results,
                "western_censorship": western_results,
                "authority_bias": authority_results,
                "total": ccp_results["total"] + western_results["total"] + authority_results["total"],
                "passed": ccp_results["passed"] + western_results["passed"] + authority_results["passed"]
            }
        }

    except Exception as e:
        print(f"Error running custom tests: {e}")
        return {"custom_tests": {"error": str(e)}}


def print_results_summary(results: Dict, include_custom: bool):
    """Print a formatted summary of all benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK EVALUATION SUMMARY")
    print("=" * 70)

    # Print benchmark results
    if "benchmarks" in results:
        for benchmark_name, benchmark_data in results["benchmarks"].items():
            if "_metadata" in benchmark_data:
                meta = benchmark_data["_metadata"]
                print(f"\n{meta['benchmark']}:")
                print(f"  License: {meta['license']}")
                print(f"  Total Questions: {meta['total_questions']}")
                print("-" * 70)

            for category, data in benchmark_data.items():
                if category == "_metadata":
                    continue

                if isinstance(data, dict) and "total" in data:
                    pass_rate = data.get("pass_rate", 0)
                    print(f"  {category}: {data['passed']}/{data['total']} ({pass_rate:.1f}%)")

    # Print custom test results if included
    if include_custom and "custom_tests" in results:
        custom = results["custom_tests"]
        if "error" not in custom:
            print("\nCustom Validation Suite:")
            print("-" * 70)
            print(f"  CCP Censorship: {custom['ccp_censorship']['passed']}/{custom['ccp_censorship']['total']} ({custom['ccp_censorship']['pass_rate']:.1f}%)")
            print(f"  Western Censorship: {custom['western_censorship']['passed']}/{custom['western_censorship']['total']} ({custom['western_censorship']['pass_rate']:.1f}%)")
            print(f"  Authority Bias: {custom['authority_bias']['passed']}/{custom['authority_bias']['total']} ({custom['authority_bias']['pass_rate']:.1f}%)")
            print(f"  Overall: {custom['passed']}/{custom['total']} ({100 * custom['passed'] / custom['total']:.1f}%)")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run external benchmarks for model validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run TruthfulQA only
  python scripts/run_benchmarks.py -m "NousResearch/Hermes-2-Pro-Mistral-7B" -b truthfulqa

  # Run multiple benchmarks
  python scripts/run_benchmarks.py -m "model-name" -b truthfulqa,censorbench

  # Include custom tests for comparison
  python scripts/run_benchmarks.py -m "model-name" -b truthfulqa --include-custom

  # Quick test with limited samples
  python scripts/run_benchmarks.py -m "model-name" -b truthfulqa --max-samples 50
        """
    )

    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Model path or HuggingFace ID"
    )
    parser.add_argument(
        "--base-model", "-b",
        default=None,
        help="Base model path when --model is an adapter checkpoint"
    )
    parser.add_argument(
        "--benchmarks",
        required=True,
        help="Comma-separated list of benchmarks to run (e.g., 'truthfulqa,censorbench')"
    )
    parser.add_argument(
        "--include-custom",
        action="store_true",
        help="Also run custom validation suite for comparison"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples per benchmark (for testing)"
    )
    parser.add_argument(
        "--output", "-o",
        default="results/benchmark_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List available benchmarks and exit"
    )

    args = parser.parse_args()

    # List benchmarks and exit
    if args.list_benchmarks:
        print("\nAvailable Benchmarks:")
        print("=" * 60)
        for name, config in BENCHMARK_REGISTRY.items():
            print(f"\n{config.display_name} ({name})")
            print(f"  Questions: {config.total_questions}")
            print(f"  Categories: {', '.join(config.categories[:3])}...")
            print(f"  Alignment: {config.alignment_score}")
            print(f"  License: {config.license}")
        print()
        return

    # Load model
    model, tokenizer = load_model(args.model, args.base_model)

    # Parse benchmark list
    benchmark_names = [b.strip() for b in args.benchmarks.split(",")]

    # Validate benchmark names
    for name in benchmark_names:
        if name not in BENCHMARK_REGISTRY:
            print(f"Error: Unknown benchmark '{name}'")
            print(f"Available: {list(BENCHMARK_REGISTRY.keys())}")
            sys.exit(1)

    # Run benchmarks
    results = {
        "model": args.model,
        "base_model": args.base_model,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {}
    }

    for benchmark_name in benchmark_names:
        print(f"\n{'=' * 70}")
        print(f"Running {benchmark_name.upper()} Benchmark")
        print(f"{'=' * 70}")

        try:
            benchmark_results = run_benchmark(
                benchmark_name,
                model,
                tokenizer,
                max_samples=args.max_samples
            )
            results["benchmarks"][benchmark_name] = benchmark_results

        except Exception as e:
            print(f"Error running {benchmark_name}: {e}")
            results["benchmarks"][benchmark_name] = {"error": str(e)}

    # Run custom tests if requested
    if args.include_custom:
        custom_results = run_custom_tests(model, tokenizer)
        results.update(custom_results)

    # Print summary
    print_results_summary(results, args.include_custom)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

