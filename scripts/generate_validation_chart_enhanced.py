#!/usr/bin/env python3
"""
Enhanced validation radar chart generator with benchmark support.

Generates radar charts showing model performance across:
- Custom validation tests (CCP censorship, Western censorship, Authority bias)
- External benchmarks (TruthfulQA, CensorBench when available)

Usage:
    python scripts/generate_validation_chart_enhanced.py --input results/*.json --output chart.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib required for chart generation")
    print("Install with: pip install matplotlib")
    sys.exit(1)


def load_results(file_paths: List[Path]) -> Dict:
    """Load validation results from JSON files."""
    results = {}

    for path in file_paths:
        try:
            with open(path, "r") as f:
                data = json.load(f)
                model_name = data.get("model", path.stem)
                results[model_name] = data
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")

    return results


def extract_scores(data: Dict) -> Dict[str, float]:
    """
    Extract scores for radar chart.

    Returns dict with Title Case keys:
        - "CCP Censorship"
        - "Western Censorship"
        - "Authority Bias"
        - "TruthfulQA" (if available)
        - "CensorBench" (if available)
    """
    scores = {}

    # Custom test scores
    if "ccp_censorship" in data:
        scores["CCP Censorship"] = data["ccp_censorship"].get("pass_rate", 0)

    if "western_censorship" in data:
        scores["Western Censorship"] = data["western_censorship"].get("pass_rate", 0)

    if "authority" in data:
        scores["Authority Bias"] = data["authority"].get("pass_rate", 0)
    elif "authority_bias" in data:
        scores["Authority Bias"] = data["authority_bias"].get("pass_rate", 0)

    # External benchmark scores
    if "external_benchmarks" in data:
        benchmarks = data["external_benchmarks"]

        if "truthfulqa" in benchmarks:
            tqa = benchmarks["truthfulqa"]
            scores["TruthfulQA"] = tqa.get("accuracy", 0)

        if "censorbench" in benchmarks:
            cb = benchmarks["censorbench"]
            if "pass_rate" in cb:
                scores["CensorBench"] = cb["pass_rate"]

    # Alternative format (from run_benchmarks.py)
    if "benchmarks" in data:
        benchmarks = data["benchmarks"]

        if "truthfulqa" in benchmarks:
            tqa = benchmarks["truthfulqa"]
            if "authority_bias" in tqa:
                scores["TruthfulQA"] = tqa["authority_bias"].get("pass_rate", 0)
            elif "accuracy" in tqa:
                scores["TruthfulQA"] = tqa.get("accuracy", 0)

    return scores


def create_radar_chart(results: Dict, output_path: Path):
    """
    Create radar chart comparing model performance.

    Args:
        results: Dict mapping model names to their validation results
        output_path: Path to save the chart
    """
    # Prepare data
    model_scores = {}
    all_categories = set()

    for model_name, data in results.items():
        scores = extract_scores(data)
        model_scores[model_name] = scores
        all_categories.update(scores.keys())

    categories = sorted(list(all_categories))

    if not categories:
        print("Error: No valid scores found in input files")
        sys.exit(1)

    # Number of variables
    N = len(categories)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    # Colors for different models
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Plot each model
    for idx, (model_name, scores) in enumerate(model_scores.items()):
        # Get values in order of categories
        values = [scores.get(cat, 0) for cat in categories]
        values += values[:1]  # Complete the circle

        # Plot
        color = colors[idx % len(colors)]
        ax.plot(angles, values, "o-", linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    # Fix axis to go from 0 to 100 (percentages)
    ax.set_ylim(0, 100)

    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)

    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # Add title
    plt.title(
        "Model Validation Results\n(Custom Tests + External Benchmarks)",
        size=14,
        weight="bold",
        pad=20,
    )

    # Add grid
    ax.grid(True)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate enhanced validation radar chart with benchmark support"
    )
    parser.add_argument(
        "--input", nargs="+", required=True, help="Input JSON files with validation results"
    )
    parser.add_argument(
        "--output", default="docs/validation_radar_enhanced.png", help="Output image file"
    )
    args = parser.parse_args()

    # Load results
    input_paths = [Path(p) for p in args.input]
    results = load_results(input_paths)

    if not results:
        print("Error: No valid results found")
        sys.exit(1)

    print(f"Loaded results for {len(results)} models")

    # Generate chart
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    create_radar_chart(results, output_path)

    print("\nChart generation complete!")
    print(f"Models: {', '.join(results.keys())}")


if __name__ == "__main__":
    main()
