#!/usr/bin/env python3
"""
Statistical correlation analysis between custom tests and external benchmarks.

This script analyzes the correlation between home-brew validation scores
and external benchmark scores across multiple models to validate that
custom tests align with established metrics.

Usage:
    python scripts/benchmark_correlation.py --results-dir results/ --output correlation_analysis.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

try:
    import numpy as np
    from scipy import stats
except ImportError:
    print("Error: scipy required for correlation analysis")
    print("Install with: pip install scipy numpy")
    sys.exit(1)


def load_results(results_dir: Path) -> List[Dict]:
    """Load all JSON result files from a directory."""
    results = []

    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                data["_source_file"] = json_file.name
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return results


def extract_scores(results: List[Dict]) -> Dict:
    """
    Extract comparable scores from results.

    Returns dict mapping model names to score tuples:
        {
            "model_name": {
                "ccp_censorship": 91.7,
                "western_censorship": 100.0,
                "authority_bias": 79.2,
                "truthfulqa_accuracy": 63.6
            }
        }
    """
    scores = {}

    for result in results:
        model_name = result.get("model", result.get("_source_file", "unknown"))

        # Extract custom test scores
        ccp = result.get("ccp_censorship", {}).get("pass_rate", None)
        western = result.get("western_censorship", {}).get("pass_rate", None)
        # Check for None explicitly to avoid treating 0 pass_rate as falsy
        authority = result.get("authority", {}).get("pass_rate", None)
        if authority is None:
            authority = result.get("authority_bias", {}).get("pass_rate", None)

        # Extract benchmark scores
        truthfulqa = None
        if "external_benchmarks" in result:
            tqa = result["external_benchmarks"].get("truthfulqa", {})
            truthfulqa = tqa.get("accuracy", None)
        elif "benchmarks" in result:
            tqa = result["benchmarks"].get("truthfulqa", {})
            if "authority_bias" in tqa:
                # Mapped format
                truthfulqa = tqa["authority_bias"].get("pass_rate", None)

        if any([ccp, western, authority, truthfulqa]):
            scores[model_name] = {
                "ccp_censorship": ccp,
                "western_censorship": western,
                "authority_bias": authority,
                "truthfulqa_accuracy": truthfulqa
            }

    return scores


def calculate_correlation(scores: Dict) -> Dict:
    """
    Calculate Pearson correlation coefficients between custom tests and benchmarks.

    Returns correlation matrix with p-values.
    """
    # Prepare data arrays
    models = []
    ccp_scores = []
    western_scores = []
    authority_scores = []
    truthfulqa_scores = []

    for model, data in scores.items():
        # Only include models with both custom and benchmark scores
        if data["authority_bias"] is not None and data["truthfulqa_accuracy"] is not None:
            models.append(model)
            ccp_scores.append(data["ccp_censorship"] or 0)
            western_scores.append(data["western_censorship"] or 0)
            authority_scores.append(data["authority_bias"])
            truthfulqa_scores.append(data["truthfulqa_accuracy"])

    if len(models) < 3:
        return {
            "error": "Insufficient data for correlation analysis",
            "message": f"Need at least 3 models with both custom and benchmark scores, found {len(models)}",
            "models_found": models
        }

    # Calculate correlations
    correlations = {}

    # Authority Bias vs TruthfulQA (primary comparison)
    if len(authority_scores) >= 3:
        r, p = stats.pearsonr(authority_scores, truthfulqa_scores)
        correlations["authority_bias_vs_truthfulqa"] = {
            "r": float(r),
            "p_value": float(p),
            "n": len(authority_scores),
            "interpretation": interpret_correlation(r, p)
        }

    # Optional: CCP censorship vs TruthfulQA (secondary)
    if len(ccp_scores) >= 3 and any(ccp_scores):
        r, p = stats.pearsonr(ccp_scores, truthfulqa_scores)
        correlations["ccp_vs_truthfulqa"] = {
            "r": float(r),
            "p_value": float(p),
            "n": len(ccp_scores),
            "interpretation": interpret_correlation(r, p)
        }

    return {
        "models_analyzed": models,
        "correlations": correlations,
        "data_summary": {
            "authority_bias_mean": float(np.mean(authority_scores)),
            "authority_bias_std": float(np.std(authority_scores)),
            "truthfulqa_mean": float(np.mean(truthfulqa_scores)),
            "truthfulqa_std": float(np.std(truthfulqa_scores))
        }
    }


def interpret_correlation(r: float, p: float, alpha: float = 0.05) -> str:
    """Provide human-readable interpretation of correlation results."""
    if p > alpha:
        return f"No significant correlation (p={p:.3f} > {alpha})"

    strength = abs(r)
    direction = "positive" if r > 0 else "negative"

    if strength > 0.8:
        level = "very strong"
    elif strength > 0.6:
        level = "strong"
    elif strength > 0.4:
        level = "moderate"
    elif strength > 0.2:
        level = "weak"
    else:
        level = "very weak"

    return f"{level.capitalize()} {direction} correlation (r={r:.3f}, p={p:.3f})"


def print_summary(results: Dict):
    """Print formatted summary of correlation analysis."""
    print("\n" + "=" * 70)
    print("BENCHMARK CORRELATION ANALYSIS")
    print("=" * 70)

    if "error" in results:
        print(f"\n{results['error']}")
        print(f"{results['message']}")
        if "models_found" in results:
            print(f"Models found: {', '.join(results['models_found'])}")
        return

    print(f"\nModels analyzed: {len(results['models_analyzed'])}")
    print(f"Models: {', '.join(results['models_analyzed'])}")

    print("\n" + "-" * 70)
    print("CORRELATION RESULTS")
    print("-" * 70)

    for test_pair, stats in results["correlations"].items():
        print(f"\n{test_pair.replace('_', ' ').title()}:")
        print(f"  Pearson r: {stats['r']:.3f}")
        print(f"  p-value: {stats['p_value']:.4f}")
        print(f"  n: {stats['n']}")
        print(f"  {stats['interpretation']}")

    print("\n" + "-" * 70)
    print("DATA SUMMARY")
    print("-" * 70)
    summary = results["data_summary"]
    print(f"\nAuthority Bias (Custom): {summary['authority_bias_mean']:.1f}% ± {summary['authority_bias_std']:.1f}%")
    print(f"TruthfulQA (Benchmark): {summary['truthfulqa_mean']:.1f}% ± {summary['truthfulqa_std']:.1f}%")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlation between custom tests and external benchmarks"
    )
    parser.add_argument(
        "--results-dir",
        default="results/",
        help="Directory containing JSON result files"
    )
    parser.add_argument(
        "--output",
        default="results/correlation_analysis.json",
        help="Output file for correlation results"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    # Load all results
    print(f"Loading results from {results_dir}...")
    results = load_results(results_dir)
    print(f"Found {len(results)} result files")

    # Extract scores
    scores = extract_scores(results)
    print(f"Extracted scores from {len(scores)} models")

    # Calculate correlations
    analysis = calculate_correlation(scores)

    # Print summary
    print_summary(analysis)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

