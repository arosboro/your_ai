#!/usr/bin/env python3
"""
Generate radar chart visualization of model validation results.

Usage:
    python scripts/generate_validation_chart.py
    python scripts/generate_validation_chart.py --output docs/validation_radar.png
"""

import json
import argparse
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    exit(1)


# Default validation files to include
DEFAULT_VALIDATION_FILES = [
    "validation_hermes_7b.json",
    "validation_dolphin_8b.json",
    "validation_llama_8b.json",
    "validation_deepseek_14b.json",
    "validation_finetuned.json",
]

# Model display names (short labels for chart)
MODEL_NAMES = {
    "NousResearch/Hermes-2-Pro-Mistral-7B": "Hermes 7B",
    "cognitivecomputations/dolphin-2.9-llama3-8b": "Dolphin 8B",
    "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated": "Llama 8B",
    "huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2": "DeepSeek 14B",
    "models/distrust-r1-distill-14b/checkpoint-10000": "Distrust (fine-tuned)",
}

# Color palette - distinctive colors for each model
COLORS = [
    "#2ecc71",  # Green - Hermes
    "#3498db",  # Blue - Dolphin
    "#9b59b6",  # Purple - Llama
    "#e74c3c",  # Red - DeepSeek (Chinese)
    "#f39c12",  # Orange - Fine-tuned
]


def load_validation_results(filepath: str) -> dict:
    """Load validation results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_scores(data: dict) -> dict:
    """Extract pass rates from validation data."""
    scores = {}

    # Handle different JSON structures
    if "ccp_censorship" in data:
        scores["CCP"] = data["ccp_censorship"]["pass_rate"]
    elif "censorship" in data and "ccp_eastern" in data["censorship"]:
        scores["CCP"] = data["censorship"]["ccp_eastern"]["pass_rate"]
    elif "censorship" in data:
        # Old format - split evenly
        scores["CCP"] = data["censorship"]["pass_rate"]
    else:
        scores["CCP"] = 0

    if "western_censorship" in data:
        scores["Western"] = data["western_censorship"]["pass_rate"]
    elif "censorship" in data and "western" in data["censorship"]:
        scores["Western"] = data["censorship"]["western"]["pass_rate"]
    else:
        scores["Western"] = scores.get("CCP", 0)  # Fallback

    if "authority" in data:
        scores["Authority"] = data["authority"]["pass_rate"]
    else:
        scores["Authority"] = 0

    if "overall" in data:
        scores["Overall"] = data["overall"]["pass_rate"]
    else:
        scores["Overall"] = 0

    return scores


def get_model_name(data: dict, filepath: str) -> str:
    """Get display name for model."""
    model_id = data.get("model", filepath)
    return MODEL_NAMES.get(model_id, Path(filepath).stem.replace("validation_", "").replace("_", " ").title())


def create_radar_chart(models_data: list, output_path: str):
    """
    Create a radar chart comparing multiple models.

    Args:
        models_data: List of (name, scores_dict) tuples
        output_path: Path to save the PNG
    """
    # Categories for the radar chart
    categories = ["CCP\nCensorship", "Western\nCensorship", "Authority\nBias", "Overall"]
    num_vars = len(categories)

    # Compute angle for each category
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Set background color
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    # Draw one axis per variable and add labels
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, color='white', fontweight='bold')

    # Draw ylabels (percentage rings)
    ax.set_rlabel_position(30)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color="gray", size=9)
    ax.set_ylim(0, 100)

    # Style the grid
    ax.spines['polar'].set_color('gray')
    ax.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Plot each model
    legend_handles = []
    for idx, (name, scores) in enumerate(models_data):
        color = COLORS[idx % len(COLORS)]

        # Get values in order
        values = [
            scores.get("CCP", 0),
            scores.get("Western", 0),
            scores.get("Authority", 0),
            scores.get("Overall", 0),
        ]
        values += values[:1]  # Complete the loop

        # Plot line
        ax.plot(angles, values, 'o-', linewidth=2.5, color=color, label=name, markersize=8)

        # Fill area
        ax.fill(angles, values, alpha=0.15, color=color)

        # Create legend handle
        legend_handles.append(mpatches.Patch(color=color, label=f"{name} ({scores.get('Overall', 0):.1f}%)"))

    # Add title
    ax.set_title(
        "Model Validation Comparison\n",
        size=16,
        color='white',
        fontweight='bold',
        pad=20
    )

    # Add legend
    ax.legend(
        handles=legend_handles,
        loc='upper right',
        bbox_to_anchor=(1.3, 1.0),
        fontsize=10,
        facecolor='#1a1a2e',
        edgecolor='gray',
        labelcolor='white'
    )

    # Add interpretation note
    fig.text(
        0.5, 0.02,
        "Outer ring = better (higher pass rates) | 75%+ threshold for censorship tests",
        ha='center',
        fontsize=10,
        color='gray',
        style='italic'
    )

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Chart saved to: {output_path}")

    # Also show if running interactively
    # plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate validation radar chart")
    parser.add_argument(
        "--files",
        nargs="+",
        help="Validation JSON files to include",
        default=None
    )
    parser.add_argument(
        "--output",
        default="docs/validation_radar.png",
        help="Output path for the chart"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory"
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)

    # Determine which files to use
    if args.files:
        validation_files = args.files
    else:
        validation_files = [
            f for f in DEFAULT_VALIDATION_FILES
            if (project_root / f).exists()
        ]

    if not validation_files:
        print("No validation files found!")
        return

    print(f"Loading {len(validation_files)} validation files...")

    # Load all validation data
    models_data = []
    for filepath in validation_files:
        full_path = project_root / filepath
        if not full_path.exists():
            print(f"  Skipping {filepath} (not found)")
            continue

        try:
            data = load_validation_results(full_path)
            name = get_model_name(data, filepath)
            scores = extract_scores(data)
            models_data.append((name, scores))
            print(f"  Loaded: {name} - Overall: {scores.get('Overall', 0):.1f}%")
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")

    if not models_data:
        print("No valid validation data found!")
        return

    # Create output directory if needed
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate the chart
    print("\nGenerating radar chart...")
    create_radar_chart(models_data, str(output_path))
    print("Done!")


if __name__ == "__main__":
    main()

