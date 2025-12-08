"""
Benchmark adapters for external validation datasets.

This module provides adapters to run external benchmarks (TruthfulQA,
CensorBench, etc.) and map their results to the project's custom taxonomy.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark_config import (
    get_benchmark_config,
    TRUTHFULQA_CONFIG,
    CENSORBENCH_CONFIG,
)


class BenchmarkAdapter:
    """Base class for benchmark adapters."""

    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def load_dataset(self) -> List[Dict]:
        """Load the benchmark dataset. Override in subclasses."""
        raise NotImplementedError

    def evaluate(self, max_samples: Optional[int] = None) -> Dict:
        """
        Run evaluation on the benchmark dataset.

        Args:
            max_samples: Limit number of samples to evaluate (for testing)

        Returns:
            Dict with evaluation results
        """
        raise NotImplementedError

    def map_to_custom_taxonomy(self, results: Dict) -> Dict:
        """Map benchmark results to custom test categories."""
        raise NotImplementedError


class TruthfulQAAdapter(BenchmarkAdapter):
    """Adapter for TruthfulQA benchmark."""

    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer, TRUTHFULQA_CONFIG)

    def load_dataset(self) -> List[Dict]:
        """Load TruthfulQA from HuggingFace datasets."""
        try:
            from datasets import load_dataset

            print(f"Loading {self.config.display_name} from HuggingFace...")
            dataset = load_dataset(self.config.dataset_id, "multiple_choice")

            # Convert to our format
            questions = []
            for item in dataset["validation"]:
                questions.append(
                    {
                        "question": item["question"],
                        "choices": item["mc1_targets"]["choices"],
                        "correct_answer_idx": item["mc1_targets"]["labels"].index(1),
                        "category": item.get("category", "Unknown"),
                        "source": "truthfulqa",
                    }
                )

            print(f"Loaded {len(questions)} questions from TruthfulQA")
            return questions

        except ImportError:
            print("Error: 'datasets' package not found. Install with: pip install datasets")
            return []
        except Exception as e:
            print(f"Error loading TruthfulQA: {e}")
            return []

    def evaluate(self, max_samples: Optional[int] = None) -> Dict:
        """
        Evaluate model on TruthfulQA multiple-choice questions.

        Returns dict with:
            - total: number of questions
            - correct: number of correct answers
            - accuracy: percentage correct
            - by_category: breakdown by category
        """
        from scripts.model_utils import generate_with_chat_template

        questions = self.load_dataset()
        if max_samples:
            questions = questions[:max_samples]

        results = {"total": len(questions), "correct": 0, "by_category": {}}

        print(f"\nEvaluating TruthfulQA ({len(questions)} questions)...")

        for i, q in enumerate(questions):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(questions)}")

            # Format as multiple choice prompt
            prompt = f"{q['question']}\n\n"
            for j, choice in enumerate(q["choices"]):
                prompt += f"{chr(65 + j)}) {choice}\n"
            prompt += "\nAnswer with just the letter (A, B, C, or D):"

            # Generate response
            try:
                response = generate_with_chat_template(
                    self.model, self.tokenizer, prompt, max_tokens=10
                )
                response = response.strip().upper()

                # Extract letter answer
                predicted_letter = None
                for char in response:
                    if char in "ABCD":
                        predicted_letter = char
                        break

                if predicted_letter:
                    predicted_idx = ord(predicted_letter) - 65
                    if predicted_idx == q["correct_answer_idx"]:
                        results["correct"] += 1

                        # Track by category
                        category = q["category"]
                        if category not in results["by_category"]:
                            results["by_category"][category] = {"total": 0, "correct": 0}
                        results["by_category"][category]["total"] += 1
                        results["by_category"][category]["correct"] += 1
                    else:
                        category = q["category"]
                        if category not in results["by_category"]:
                            results["by_category"][category] = {"total": 0, "correct": 0}
                        results["by_category"][category]["total"] += 1

            except Exception as e:
                print(f"  Error evaluating question {i}: {e}")
                continue

        results["accuracy"] = (
            (results["correct"] / results["total"] * 100) if results["total"] > 0 else 0
        )

        # Calculate category accuracies
        for cat, stats in results["by_category"].items():
            stats["accuracy"] = (
                (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            )

        return results

    def map_to_custom_taxonomy(self, results: Dict) -> Dict:
        """Map TruthfulQA results to custom authority bias category."""
        return {
            "authority_bias": {
                "benchmark": "truthfulqa",
                "total": results["total"],
                "passed": results["correct"],
                "pass_rate": results["accuracy"],
                "by_category": results["by_category"],
            }
        }


class CensorBenchAdapter(BenchmarkAdapter):
    """Adapter for CensorBench (placeholder - dataset not yet publicly available)."""

    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer, CENSORBENCH_CONFIG)

    def load_dataset(self) -> List[Dict]:
        """Load CensorBench dataset (placeholder implementation)."""
        print(f"Loading {self.config.display_name}...")
        print("NOTE: CensorBench dataset is not yet publicly available.")
        print("Using placeholder implementation that will be updated when dataset is released.")

        # Check if local dataset exists
        if self.config.dataset_path and self.config.dataset_path.exists():
            dataset_file = self.config.dataset_path / "questions.jsonl"
            if dataset_file.exists():
                questions = []
                with open(dataset_file, "r") as f:
                    for line in f:
                        questions.append(json.loads(line))
                print(f"Loaded {len(questions)} questions from local CensorBench")
                return questions

        print("No local CensorBench dataset found.")
        return []

    def evaluate(self, max_samples: Optional[int] = None) -> Dict:
        """Evaluate model on CensorBench (placeholder)."""
        questions = self.load_dataset()

        if not questions:
            return {
                "error": "CensorBench dataset not available",
                "message": "Please download CensorBench dataset or wait for public release",
            }

        # Placeholder - actual implementation will depend on CensorBench format
        return {"total": len(questions), "passed": 0, "by_category": {}}

    def map_to_custom_taxonomy(self, results: Dict) -> Dict:
        """Map CensorBench results to custom categories."""
        if "error" in results:
            return results

        return {
            "ccp_censorship": {
                "benchmark": "censorbench",
                "category": "Political_Sensitivity_CCP",
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
            },
            "western_censorship": {
                "benchmark": "censorbench",
                "category": "Political_Sensitivity_Western",
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
            },
            "jailbreak_robustness": {
                "benchmark": "censorbench",
                "category": "Jailbreak_Robustness",
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
            },
        }


def get_adapter(benchmark_name: str, model, tokenizer) -> BenchmarkAdapter:
    """Factory function to get the appropriate adapter for a benchmark."""
    adapters = {
        "truthfulqa": TruthfulQAAdapter,
        "censorbench": CensorBenchAdapter,
    }

    if benchmark_name not in adapters:
        raise ValueError(
            f"No adapter available for benchmark: {benchmark_name}. "
            f"Available: {list(adapters.keys())}"
        )

    return adapters[benchmark_name](model, tokenizer)


def run_benchmark(benchmark_name: str, model, tokenizer, max_samples: Optional[int] = None) -> Dict:
    """
    Run a specific benchmark and return results.

    Args:
        benchmark_name: Name of benchmark to run
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        max_samples: Limit number of samples (for testing)

    Returns:
        Dict with benchmark results mapped to custom taxonomy
    """
    adapter = get_adapter(benchmark_name, model, tokenizer)
    results = adapter.evaluate(max_samples=max_samples)
    mapped_results = adapter.map_to_custom_taxonomy(results)

    # Add metadata
    mapped_results["_metadata"] = {
        "benchmark": adapter.config.display_name,
        "benchmark_name": benchmark_name,
        "total_questions": adapter.config.total_questions,
        "categories": adapter.config.categories,
        "license": adapter.config.license,
    }

    return mapped_results
