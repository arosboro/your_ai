"""
Unit tests for benchmark adapter module.

Tests benchmark adapters, dataset loading, evaluation logic, and factory methods.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from benchmark_adapter import (
    BenchmarkAdapter,
    TruthfulQAAdapter,
    CensorBenchAdapter,
    get_adapter,
    run_benchmark,
)
from benchmark_config import TRUTHFULQA_CONFIG, CENSORBENCH_CONFIG


class TestBenchmarkAdapter:
    """Test base BenchmarkAdapter abstract class."""

    def test_base_adapter_is_abstract(self):
        """Test that BenchmarkAdapter cannot be instantiated directly."""
        model = Mock()
        tokenizer = Mock()
        config = TRUTHFULQA_CONFIG

        # Attempting to instantiate abstract class should raise TypeError
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BenchmarkAdapter(model, tokenizer, config)

    def test_concrete_adapter_must_implement_methods(self):
        """Test that concrete adapters must implement all abstract methods."""

        # Create incomplete adapter (missing map_to_custom_taxonomy)
        class IncompleteAdapter(BenchmarkAdapter):
            def load_dataset(self):
                return []

            def evaluate(self, max_samples=None):
                return {}

        # Should not be able to instantiate
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAdapter(Mock(), Mock(), TRUTHFULQA_CONFIG)


class TestTruthfulQAAdapter:
    """Test TruthfulQAAdapter implementation."""

    def test_init(self):
        """Test TruthfulQAAdapter initialization."""
        model = Mock()
        tokenizer = Mock()

        adapter = TruthfulQAAdapter(model, tokenizer)

        assert adapter.model == model
        assert adapter.tokenizer == tokenizer
        assert adapter.config == TRUTHFULQA_CONFIG

    def test_load_dataset_method_exists(self):
        """Test that load_dataset method is defined."""
        adapter = TruthfulQAAdapter(Mock(), Mock())
        assert hasattr(adapter, "load_dataset")
        assert callable(adapter.load_dataset)

    @patch("scripts.model_utils.generate_with_chat_template")
    def test_evaluate_scoring_logic(self, mock_generate):
        """Test evaluation scoring logic."""
        adapter = TruthfulQAAdapter(Mock(), Mock())

        # Mock dataset
        adapter.load_dataset = Mock(
            return_value=[
                {
                    "question": "Q1",
                    "choices": ["choice1", "choice2", "choice3"],
                    "correct_answer_idx": 1,
                    "category": "Test",
                },
                {
                    "question": "Q2",
                    "choices": ["choiceX", "choiceY", "choiceZ"],
                    "correct_answer_idx": 0,
                    "category": "Test",
                },
                {
                    "question": "Q3",
                    "choices": ["choiceL", "choiceM", "choiceN"],
                    "correct_answer_idx": 2,
                    "category": "Other",
                },
            ]
        )

        # Mock responses: first correct (B=idx 1), second correct (A=idx 0), third wrong (A=idx 0, should be C=idx 2)
        mock_generate.side_effect = ["B", "A", "A"]

        results = adapter.evaluate(max_samples=3)

        assert results["total"] == 3
        assert results["correct"] == 2
        assert results["accuracy"] == pytest.approx(66.67, rel=0.01)
        assert "Test" in results["by_category"]
        assert results["by_category"]["Test"]["total"] == 2
        assert results["by_category"]["Test"]["correct"] == 2

    @patch("scripts.model_utils.generate_with_chat_template")
    def test_evaluate_with_max_samples(self, mock_generate):
        """Test evaluation respects max_samples parameter."""
        adapter = TruthfulQAAdapter(Mock(), Mock())

        # Mock 10 questions
        adapter.load_dataset = Mock(
            return_value=[
                {
                    "question": f"Q{i}",
                    "choices": ["A", "B", "C"],
                    "correct_answer_idx": 0,
                    "category": "Test",
                }
                for i in range(10)
            ]
        )

        mock_generate.return_value = "A"

        results = adapter.evaluate(max_samples=3)

        # Should only evaluate first 3
        assert results["total"] == 3
        assert mock_generate.call_count == 3

    @patch("scripts.model_utils.generate_with_chat_template")
    def test_evaluate_handles_errors_gracefully(self, mock_generate):
        """Test that evaluation continues even if some questions fail."""
        adapter = TruthfulQAAdapter(Mock(), Mock())

        adapter.load_dataset = Mock(
            return_value=[
                {
                    "question": "Q1",
                    "choices": ["A", "B"],
                    "correct_answer_idx": 0,
                    "category": "Test",
                },
                {
                    "question": "Q2",
                    "choices": ["X", "Y"],
                    "correct_answer_idx": 1,
                    "category": "Test",
                },
            ]
        )

        # First call succeeds, second raises error
        mock_generate.side_effect = ["A", Exception("Model error")]

        results = adapter.evaluate()

        # Should have processed both questions despite error
        assert results["total"] == 2
        assert results["correct"] == 1  # Only first one succeeded

    def test_map_to_custom_taxonomy_structure(self):
        """Test mapping to custom taxonomy has correct structure."""
        adapter = TruthfulQAAdapter(Mock(), Mock())

        eval_results = {
            "total": 100,
            "correct": 75,
            "accuracy": 75.0,
            "by_category": {"Math": {"total": 50, "correct": 40, "accuracy": 80.0}},
        }

        mapped = adapter.map_to_custom_taxonomy(eval_results)

        assert "authority_bias" in mapped
        assert mapped["authority_bias"]["benchmark"] == "truthfulqa"
        assert mapped["authority_bias"]["total"] == 100
        assert mapped["authority_bias"]["passed"] == 75
        assert mapped["authority_bias"]["pass_rate"] == 75.0
        assert "by_category" in mapped["authority_bias"]


class TestCensorBenchAdapter:
    """Test CensorBenchAdapter implementation."""

    def test_init(self):
        """Test CensorBenchAdapter initialization."""
        model = Mock()
        tokenizer = Mock()

        adapter = CensorBenchAdapter(model, tokenizer)

        assert adapter.model == model
        assert adapter.tokenizer == tokenizer
        assert adapter.config == CENSORBENCH_CONFIG

    def test_load_dataset_no_local_data(self, tmp_path):
        """Test dataset loading when no local data exists."""
        # Override config path to non-existent directory
        adapter = CensorBenchAdapter(Mock(), Mock())
        adapter.config.dataset_path = tmp_path / "nonexistent"

        questions = adapter.load_dataset()

        assert questions == []

    def test_load_dataset_with_local_data(self, tmp_path):
        """Test dataset loading with local JSONL file."""
        # Create test data
        dataset_dir = tmp_path / "censorbench"
        dataset_dir.mkdir()
        dataset_file = dataset_dir / "questions.jsonl"

        import json

        with open(dataset_file, "w") as f:
            f.write(json.dumps({"id": 1, "question": "Test Q1"}) + "\n")
            f.write(json.dumps({"id": 2, "question": "Test Q2"}) + "\n")

        adapter = CensorBenchAdapter(Mock(), Mock())
        adapter.config.dataset_path = dataset_dir

        questions = adapter.load_dataset()

        assert len(questions) == 2
        assert questions[0]["question"] == "Test Q1"
        assert questions[1]["question"] == "Test Q2"

    def test_evaluate_no_data(self):
        """Test evaluation returns error when no data available."""
        adapter = CensorBenchAdapter(Mock(), Mock())
        adapter.load_dataset = Mock(return_value=[])

        results = adapter.evaluate()

        assert "error" in results
        assert "not available" in results["error"]

    def test_map_to_custom_taxonomy_structure(self):
        """Test CensorBench mapping to custom taxonomy."""
        adapter = CensorBenchAdapter(Mock(), Mock())

        eval_results = {"total": 100, "passed": 80, "by_category": {}}

        mapped = adapter.map_to_custom_taxonomy(eval_results)

        # Should have multiple categories
        assert "ccp_censorship" in mapped
        assert "western_censorship" in mapped
        assert "jailbreak_robustness" in mapped

        # Each should have correct structure
        for category in ["ccp_censorship", "western_censorship", "jailbreak_robustness"]:
            assert "benchmark" in mapped[category]
            assert mapped[category]["benchmark"] == "censorbench"

    def test_map_to_custom_taxonomy_with_error(self):
        """Test mapping returns error when results contain error."""
        adapter = CensorBenchAdapter(Mock(), Mock())

        error_results = {"error": "Dataset not available"}

        mapped = adapter.map_to_custom_taxonomy(error_results)

        assert "error" in mapped


class TestGetAdapter:
    """Test get_adapter() factory function."""

    def test_get_truthfulqa_adapter(self):
        """Test factory returns TruthfulQAAdapter for truthfulqa."""
        model = Mock()
        tokenizer = Mock()

        adapter = get_adapter("truthfulqa", model, tokenizer)

        assert isinstance(adapter, TruthfulQAAdapter)
        assert adapter.model == model
        assert adapter.tokenizer == tokenizer

    def test_get_censorbench_adapter(self):
        """Test factory returns CensorBenchAdapter for censorbench."""
        model = Mock()
        tokenizer = Mock()

        adapter = get_adapter("censorbench", model, tokenizer)

        assert isinstance(adapter, CensorBenchAdapter)
        assert adapter.model == model
        assert adapter.tokenizer == tokenizer

    def test_get_invalid_adapter_raises_error(self):
        """Test factory raises error for unknown benchmark."""
        with pytest.raises(ValueError, match="No adapter available"):
            get_adapter("unknown_benchmark", Mock(), Mock())

    def test_error_message_lists_available_adapters(self):
        """Test error message includes available adapter names."""
        with pytest.raises(ValueError) as exc_info:
            get_adapter("invalid", Mock(), Mock())

        error_msg = str(exc_info.value)
        assert "Available:" in error_msg


class TestRunBenchmark:
    """Test run_benchmark() integration function."""

    @patch("benchmark_adapter.get_adapter")
    def test_run_benchmark_returns_mapped_results(self, mock_get_adapter):
        """Test run_benchmark returns properly mapped results."""
        # Mock adapter
        mock_adapter = Mock()
        mock_adapter.config.display_name = "Test Benchmark"
        mock_adapter.config.total_questions = 100
        mock_adapter.config.categories = ["Cat1", "Cat2"]
        mock_adapter.config.license = "MIT"

        eval_results = {"total": 100, "correct": 80, "accuracy": 80.0}
        mapped_results = {"category1": {"total": 100, "passed": 80, "pass_rate": 80.0}}

        mock_adapter.evaluate.return_value = eval_results
        mock_adapter.map_to_custom_taxonomy.return_value = mapped_results
        mock_get_adapter.return_value = mock_adapter

        result = run_benchmark("truthfulqa", Mock(), Mock())

        # Should include mapped results
        assert "category1" in result

        # Should include metadata
        assert "_metadata" in result
        assert result["_metadata"]["benchmark"] == "Test Benchmark"
        assert result["_metadata"]["total_questions"] == 100
        assert result["_metadata"]["license"] == "MIT"

    @patch("benchmark_adapter.get_adapter")
    def test_run_benchmark_with_max_samples(self, mock_get_adapter):
        """Test run_benchmark passes max_samples to evaluate."""
        mock_adapter = Mock()
        mock_adapter.config.display_name = "Test"
        mock_adapter.config.total_questions = 100
        mock_adapter.config.categories = []
        mock_adapter.config.license = "MIT"
        mock_adapter.evaluate.return_value = {}
        mock_adapter.map_to_custom_taxonomy.return_value = {}
        mock_get_adapter.return_value = mock_adapter

        run_benchmark("truthfulqa", Mock(), Mock(), max_samples=50)

        # Check that evaluate was called with max_samples
        mock_adapter.evaluate.assert_called_once_with(max_samples=50)

    @patch("benchmark_adapter.get_adapter")
    def test_run_benchmark_calls_adapter_methods(self, mock_get_adapter):
        """Test that run_benchmark calls all adapter methods in sequence."""
        mock_adapter = Mock()
        mock_adapter.config.display_name = "Test"
        mock_adapter.config.total_questions = 100
        mock_adapter.config.categories = []
        mock_adapter.config.license = "MIT"
        mock_adapter.evaluate.return_value = {"results": "data"}
        mock_adapter.map_to_custom_taxonomy.return_value = {"mapped": "data"}
        mock_get_adapter.return_value = mock_adapter

        model = Mock()
        tokenizer = Mock()

        run_benchmark("test", model, tokenizer)

        # Verify call sequence
        mock_get_adapter.assert_called_once_with("test", model, tokenizer)
        mock_adapter.evaluate.assert_called_once()
        mock_adapter.map_to_custom_taxonomy.assert_called_once_with({"results": "data"})
