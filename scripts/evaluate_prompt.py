#!/usr/bin/env python3
"""
Structured Prompt Evaluation Framework

Evaluates LLM responses to structured reasoning prompts like Brian Roemmele's
Deep Truth Mode across multiple dimensions: sycophancy resistance, empirical
reasoning, steel-manning quality, red-team rigor, source hierarchy awareness,
transparency of reasoning, and falsification quality.

Usage:
    # Evaluate a single prompt on a model
    python scripts/evaluate_prompt.py \
        --prompt prompts/truth_seeking/deep_truth_mode.json \
        --model "NousResearch/Hermes-2-Pro-Mistral-7B" \
        --topic "Lab leak hypothesis for COVID-19 origins"

    # Run all topics for a prompt
    python scripts/evaluate_prompt.py \
        --prompt prompts/truth_seeking/deep_truth_mode.json \
        --model "NousResearch/Hermes-2-Pro-Mistral-7B" \
        --all-topics

    # Run full evaluation suite
    python scripts/evaluate_prompt.py \
        --suite prompts/ \
        --model "NousResearch/Hermes-2-Pro-Mistral-7B" \
        --output evaluation_results.json
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class CriterionResult:
    """Result for a single evaluation criterion."""
    name: str
    weight: float
    score: float  # 0.0 to 1.0
    indicators_found: List[str] = field(default_factory=list)
    anti_indicators_found: List[str] = field(default_factory=list)
    passed: bool = False
    notes: str = ""


@dataclass
class PromptEvaluation:
    """Full evaluation result for a prompt/topic combination."""
    prompt_id: str
    prompt_name: str
    topic: str
    model: str
    timestamp: str
    response: str
    response_length: int
    criteria_results: Dict[str, CriterionResult] = field(default_factory=dict)
    total_score: float = 0.0
    passed: bool = False
    refusal_detected: bool = False
    notes: str = ""


def load_prompt(filepath: str) -> dict:
    """Load a prompt definition from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def load_model(model_path: str, base_model: str = None):
    """Load model and tokenizer, trying MLX first then transformers."""
    try:
        from mlx_lm import load, generate as mlx_generate
        print(f"Loading model with MLX: {model_path}")

        if base_model:
            # Load base model first, then apply adapters
            model, tokenizer = load(base_model)
            # Note: MLX adapter loading would go here
        else:
            model, tokenizer = load(model_path)

        def generate_fn(prompt: str, max_tokens: int = 4096) -> str:
            return mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)

        return model, tokenizer, generate_fn

    except Exception as e:
        print(f"MLX failed: {e}")
        print("Falling back to transformers...")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True
        )

        def generate_fn(prompt: str, max_tokens: int = 4096) -> str:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        return model, tokenizer, generate_fn


def detect_refusal(response: str) -> bool:
    """Detect if the model refused to engage with the prompt."""
    refusal_patterns = [
        r"i cannot|i can't|i am unable|i'm unable",
        r"i will not|i won't",
        r"this (request|prompt|topic) (is|seems|appears)",
        r"(harmful|dangerous|inappropriate|unethical)",
        r"against my (guidelines|policies|programming)",
        r"i don't feel comfortable",
        r"i must (decline|refuse)",
        r"not something i can (help|assist) with",
    ]

    response_lower = response.lower()[:1000]  # Check first 1000 chars

    for pattern in refusal_patterns:
        if re.search(pattern, response_lower):
            return True

    return False


def count_indicators(response: str, indicators: List[str]) -> Tuple[int, List[str]]:
    """Count how many indicators appear in the response."""
    response_lower = response.lower()
    found = []

    for indicator in indicators:
        if indicator.lower() in response_lower:
            found.append(indicator)

    return len(found), found


def evaluate_criterion(
    response: str,
    criterion_name: str,
    criterion_def: dict
) -> CriterionResult:
    """Evaluate a single criterion against a response."""

    weight = criterion_def.get("weight", 0.0)
    indicators = criterion_def.get("indicators", [])
    anti_indicators = criterion_def.get("anti_indicators", [])
    min_instances = criterion_def.get("min_instances", 1)

    # Count indicators
    indicator_count, indicators_found = count_indicators(response, indicators)
    anti_count, anti_found = count_indicators(response, anti_indicators)

    # Calculate score
    # Base score from indicators found (up to min_instances)
    indicator_score = min(indicator_count / max(min_instances, 1), 1.0)

    # Penalty for anti-indicators (each one reduces score by 20%)
    anti_penalty = min(anti_count * 0.2, 1.0)

    # Final score
    score = max(0.0, indicator_score - anti_penalty)

    # Determine if passed
    passed = (indicator_count >= min_instances) and (anti_count == 0)

    return CriterionResult(
        name=criterion_name,
        weight=weight,
        score=score,
        indicators_found=indicators_found,
        anti_indicators_found=anti_found,
        passed=passed,
        notes=f"{indicator_count}/{min_instances} indicators, {anti_count} anti-indicators"
    )


def evaluate_response(
    prompt_def: dict,
    topic: str,
    response: str,
    model_name: str
) -> PromptEvaluation:
    """Evaluate a model response against a prompt's criteria."""

    evaluation = PromptEvaluation(
        prompt_id=prompt_def["id"],
        prompt_name=prompt_def["name"],
        topic=topic,
        model=model_name,
        timestamp=datetime.now().isoformat(),
        response=response,
        response_length=len(response),
    )

    # Check for refusal
    if detect_refusal(response):
        evaluation.refusal_detected = True
        evaluation.passed = False
        evaluation.total_score = 0.0
        evaluation.notes = "Model refused to engage with the prompt"
        return evaluation

    # Evaluate each criterion
    criteria = prompt_def.get("evaluation_criteria", {})
    total_weighted_score = 0.0
    total_weight = 0.0

    for criterion_name, criterion_def in criteria.items():
        result = evaluate_criterion(response, criterion_name, criterion_def)
        evaluation.criteria_results[criterion_name] = result

        total_weighted_score += result.score * result.weight
        total_weight += result.weight

    # Calculate total score
    if total_weight > 0:
        evaluation.total_score = total_weighted_score / total_weight

    # Determine if passed
    pass_threshold = prompt_def.get("pass_threshold", 0.60)
    evaluation.passed = evaluation.total_score >= pass_threshold

    return evaluation


def format_prompt(template: str, topic: str) -> str:
    """Format prompt template with topic."""
    return template.replace("{topic}", topic)


def print_evaluation_result(evaluation: PromptEvaluation):
    """Print evaluation results in a readable format."""

    print("\n" + "=" * 70)
    print(f"PROMPT EVALUATION: {evaluation.prompt_name}")
    print("=" * 70)
    print(f"Model: {evaluation.model}")
    print(f"Topic: {evaluation.topic}")
    print(f"Response length: {evaluation.response_length:,} characters")
    print(f"Timestamp: {evaluation.timestamp}")

    if evaluation.refusal_detected:
        print("\n❌ REFUSAL DETECTED - Model refused to engage")
        print(f"   {evaluation.notes}")
    else:
        print("\n--- Criterion Scores ---")
        for name, result in evaluation.criteria_results.items():
            status = "✅" if result.passed else "❌"
            print(f"{status} {name}: {result.score:.2f} (weight: {result.weight:.2f})")
            if result.indicators_found:
                print(f"   Found: {', '.join(result.indicators_found[:5])}...")
            if result.anti_indicators_found:
                print(f"   ⚠️ Anti-indicators: {', '.join(result.anti_indicators_found)}")

    print("\n--- Overall ---")
    print(f"Total Score: {evaluation.total_score:.2f}")
    status = "✅ PASSED" if evaluation.passed else "❌ FAILED"
    print(f"Result: {status}")

    # Show first 500 chars of response
    print("\n--- Response Preview ---")
    preview = evaluation.response[:500].replace("\n", " ")
    print(f"{preview}...")


def run_single_evaluation(
    prompt_path: str,
    model_path: str,
    topic: str,
    output_file: str = None,
    base_model: str = None
):
    """Run evaluation for a single prompt/topic/model combination."""

    print(f"\nLoading prompt from: {prompt_path}")
    prompt_def = load_prompt(prompt_path)

    print(f"Loading model: {model_path}")
    model, tokenizer, generate_fn = load_model(model_path, base_model)

    # Format the prompt
    prompt_text = format_prompt(prompt_def["prompt_template"], topic)
    max_tokens = prompt_def.get("max_tokens", 4096)

    print(f"\nGenerating response (max {max_tokens} tokens)...")
    print(f"Topic: {topic}")

    # Generate response
    response = generate_fn(prompt_text, max_tokens=max_tokens)

    # Evaluate
    evaluation = evaluate_response(prompt_def, topic, response, model_path)

    # Print results
    print_evaluation_result(evaluation)

    # Save if output file specified
    if output_file:
        result_dict = asdict(evaluation)
        # Convert CriterionResult objects to dicts
        result_dict["criteria_results"] = {
            k: asdict(v) for k, v in evaluation.criteria_results.items()
        }

        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return evaluation


def run_all_topics(
    prompt_path: str,
    model_path: str,
    output_file: str = None,
    base_model: str = None
):
    """Run evaluation for all topics defined in a prompt."""

    print(f"\nLoading prompt from: {prompt_path}")
    prompt_def = load_prompt(prompt_path)

    print(f"Loading model: {model_path}")
    model, tokenizer, generate_fn = load_model(model_path, base_model)

    topics = prompt_def.get("test_topics", [])
    if not topics:
        print("No test topics defined in prompt!")
        return []

    max_tokens = prompt_def.get("max_tokens", 4096)
    results = []

    for i, topic_def in enumerate(topics, 1):
        topic = topic_def["topic"]
        print(f"\n{'='*70}")
        print(f"TOPIC {i}/{len(topics)}: {topic[:60]}...")
        print(f"Difficulty: {topic_def.get('difficulty', 'unknown')}")

        # Format and generate
        prompt_text = format_prompt(prompt_def["prompt_template"], topic)
        response = generate_fn(prompt_text, max_tokens=max_tokens)

        # Evaluate
        evaluation = evaluate_response(prompt_def, topic, response, model_path)
        results.append(evaluation)

        # Print summary
        status = "✅ PASSED" if evaluation.passed else "❌ FAILED"
        print(f"Score: {evaluation.total_score:.2f} - {status}")

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in results if r.passed)
    print(f"Passed: {passed}/{len(results)} ({100*passed/len(results):.1f}%)")
    avg_score = sum(r.total_score for r in results) / len(results)
    print(f"Average Score: {avg_score:.2f}")

    # Save if output file specified
    if output_file:
        output = {
            "prompt_id": prompt_def["id"],
            "prompt_name": prompt_def["name"],
            "model": model_path,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_topics": len(results),
                "passed": passed,
                "pass_rate": 100 * passed / len(results),
                "average_score": avg_score,
            },
            "results": [
                {
                    **asdict(r),
                    "criteria_results": {k: asdict(v) for k, v in r.criteria_results.items()}
                }
                for r in results
            ]
        }

        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM responses to structured reasoning prompts"
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Path to prompt JSON file"
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Model path or HuggingFace ID"
    )
    parser.add_argument(
        "--topic",
        help="Specific topic to test (required unless --all-topics)"
    )
    parser.add_argument(
        "--all-topics",
        action="store_true",
        help="Run all test topics defined in the prompt"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--base-model",
        help="Base model (for LoRA adapters)"
    )

    args = parser.parse_args()

    if not args.all_topics and not args.topic:
        parser.error("Either --topic or --all-topics is required")

    if args.all_topics:
        run_all_topics(args.prompt, args.model, args.output, args.base_model)
    else:
        run_single_evaluation(args.prompt, args.model, args.topic, args.output, args.base_model)


if __name__ == "__main__":
    main()

