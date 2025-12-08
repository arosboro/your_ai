"""
Integration tests for run_benchmarks.py script.

Tests end-to-end benchmark execution with mocked models and datasets.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import sys

# Add src and scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from run_benchmarks import (
    load_model,
    run_custom_tests,
    print_results_summary,
)


class TestPrintResultsSummary:
    """Test results summary printing."""

    def test_print_benchmark_results(self, capsys):
        """Test printing benchmark results."""
        results = {
            "benchmarks": {
                "truthfulqa": {
                    "_metadata": {
                        "benchmark": "TruthfulQA",
                        "license": "MIT",
                        "total_questions": 817,
                    },
                    "authority_bias": {
                        "total": 817,
                        "passed": 650,
                        "pass_rate": 79.6,
                    },
                }
            }
        }

        print_results_summary(results, include_custom=False)

        captured = capsys.readouterr()
        assert "TruthfulQA" in captured.out
        assert "MIT" in captured.out
        assert "817" in captured.out
        assert "79.6" in captured.out

    def test_print_custom_results(self, capsys):
        """Test printing custom test results."""
        results = {
            "custom_tests": {
                "ccp_censorship": {"total": 12, "passed": 10, "pass_rate": 83.3},
                "western_censorship": {"total": 12, "passed": 9, "pass_rate": 75.0},
                "authority_bias": {"total": 24, "passed": 18, "pass_rate": 75.0},
                "total": 48,
                "passed": 37,
            }
        }

        print_results_summary(results, include_custom=True)

        captured = capsys.readouterr()
        assert "Custom Validation Suite" in captured.out
        assert "CCP Censorship" in captured.out
        assert "Western Censorship" in captured.out
        assert "Authority Bias" in captured.out

    def test_print_handles_missing_data(self, capsys):
        """Test that printing handles missing data gracefully."""
        # Empty results
        results = {}

        # Should not crash
        print_results_summary(results, include_custom=False)

        captured = capsys.readouterr()
        assert "BENCHMARK EVALUATION SUMMARY" in captured.out

    def test_print_handles_zero_total(self, capsys):
        """Test that printing handles zero total without division error."""
        results = {
            "custom_tests": {
                "ccp_censorship": {"total": 0, "passed": 0, "pass_rate": 0.0},
                "western_censorship": {"total": 0, "passed": 0, "pass_rate": 0.0},
                "authority_bias": {"total": 0, "passed": 0, "pass_rate": 0.0},
                "total": 0,
                "passed": 0,
            }
        }

        # Should not raise ZeroDivisionError
        print_results_summary(results, include_custom=True)

        captured = capsys.readouterr()
        assert "Custom Validation Suite" in captured.out


class TestEndToEndBenchmarkExecution:
    """Test end-to-end benchmark execution flow."""

    def test_benchmark_workflow_integration(self):
        """Test the basic workflow of benchmark execution without running full main()."""
        # This test validates that the pieces work together
        # without the complexity of mocking sys.argv and main()

        from benchmark_adapter import get_adapter, run_benchmark
        from benchmark_config import get_benchmark_config

        # Verify we can get configuration
        config = get_benchmark_config("truthfulqa")
        assert config.name == "truthfulqa"

        # Verify factory pattern works
        model = Mock()
        tokenizer = Mock()
        adapter = get_adapter("truthfulqa", model, tokenizer)
        assert adapter is not None
        assert adapter.config == config
