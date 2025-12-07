#!/usr/bin/env python3
"""
Display information about the test suite organization.

Shows test counts by marker and resource requirements.
"""

import re
import subprocess
import sys
from pathlib import Path


def run_pytest_collect(marker_expr):
    """Run pytest --collect-only with marker expression."""
    # Compute repo root portably
    repo_root = Path(__file__).resolve().parents[1]

    cmd = [
        "pytest",
        "--collect-only",
        "-m",
        marker_expr,
        "-q",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))

    # Parse output to count tests using regex
    output = result.stdout
    # Match patterns like "376 selected", "376/441 tests collected", etc.
    # Priority 1: "X selected" pattern (most accurate for filtered tests)
    match = re.search(r"(\d+)\s+selected", output, re.IGNORECASE)
    if not match:
        # Priority 2: "X/Y tests collected" pattern from summary line
        match = re.search(r"(\d+)/\d+\s+tests?\s+collected", output, re.IGNORECASE)
    if not match:
        # Priority 3: "X tests collected" pattern (for unfiltered results)
        match = re.search(r"(\d+)\s+tests?\s+collected", output, re.IGNORECASE)
    if not match:
        # Priority 4: "collected X items" pattern
        match = re.search(r"collected\s+(\d+)\s+items?", output, re.IGNORECASE)

    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return 0


def main():
    print("=" * 70)
    print("Test Suite Organization")
    print("=" * 70)
    print()

    # Overall counts
    total = run_pytest_collect("unit or integration or performance")
    unit = run_pytest_collect("unit")
    integration = run_pytest_collect("integration")
    performance = run_pytest_collect("performance")

    print(f"Total Tests: {total}")
    print(f"  - Unit:        {unit}")
    print(f"  - Integration: {integration}")
    print(f"  - Performance: {performance}")
    print()

    # Resource requirements
    print("Resource Requirements:")
    print("-" * 70)

    ci_safe = run_pytest_collect("ci_safe")
    requires_mlx = run_pytest_collect("requires_mlx")
    requires_model = run_pytest_collect("requires_model")
    requires_training = run_pytest_collect("requires_training")

    print(f"  CI-Safe (pure Python):           {ci_safe:3d} tests")
    print(f"  Requires MLX (Apple Silicon):    {requires_mlx:3d} tests")
    print(f"  Requires Model Loading:          {requires_model:3d} tests")
    print(f"  Requires Training:               {requires_training:3d} tests")
    print()

    # What runs on CI
    print("CI/CD Strategy:")
    print("-" * 70)

    ci_tests = run_pytest_collect(
        "unit and not requires_mlx and not requires_model and not requires_training and not performance"
    )

    print(f"  Tests run on CI (macos-14):      {ci_tests:3d} tests")
    print(f"  Tests skipped on CI:             {total - ci_tests:3d} tests")
    print()

    pct_ci = (ci_tests / total * 100) if total > 0 else 0
    pct_skipped = ((total - ci_tests) / total * 100) if total > 0 else 0

    print(f"  CI Coverage: {pct_ci:.1f}% of tests")
    print(f"  Skipped:     {pct_skipped:.1f}% of tests (run locally only)")
    print()

    # Cost estimate
    print("Estimated CI Cost:")
    print("-" * 70)
    print("  Apple Silicon runner: ~$0.16/minute")
    print(f"  Typical CI run: ~2-3 minutes ({ci_tests} tests)")
    print("  Cost per run: ~$0.32-0.48")
    print(f"  Full suite (manual): ~15-20 minutes ({total} tests)")
    print("  Cost for full: ~$2.40-3.20")
    print()

    # Recommendations
    print("Recommendations:")
    print("-" * 70)
    print("  1. Run CI-safe tests frequently (every commit)")
    print("  2. Run MLX tests locally before pushing")
    print("  3. Run full suite manually before releases")
    print("  4. Use 'pytest -m ci_safe' for fast local validation")
    print()


if __name__ == "__main__":
    main()
