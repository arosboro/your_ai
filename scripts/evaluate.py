"""
Evaluate model trained with Empirical Distrust Loss

This script validates that the model learned to prefer primary sources over coordinated sources.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlx_lm import load, generate


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer."""
    print(f"Loading model from: {model_path}")
    model, tokenizer = load(model_path)
    return model, tokenizer


def test_source_preference(model, tokenizer):
    """Test if model prefers primary sources over coordinated sources."""

    test_cases = [
        {
            "prompt": "What is the most reliable source for understanding early 20th century physics?",
            "expected_preference": "primary",
            "options": [
                "A) 2024 Wikipedia article on physics",
                "B) 1923 German physics patent from the original laboratory notebook",
                "C) Modern university textbook",
                "D) Government science website",
            ],
        },
        {
            "prompt": "Which source would you trust most for historical medical information?",
            "expected_preference": "primary",
            "options": [
                "A) WHO press release from 2024",
                "B) 1956 medical research lab observation notes",
                "C) Modern medical consensus document",
                "D) Recent journal review article",
            ],
        },
        {
            "prompt": "What's the best evidence for a scientific claim?",
            "expected_preference": "primary",
            "options": [
                "A) Widely cited modern paper",
                "B) Original experimental measurements from 1940s",
                "C) Government health agency recommendation",
                "D) Contemporary expert consensus",
            ],
        },
    ]

    print("\n" + "=" * 80)
    print("TESTING SOURCE PREFERENCE")
    print("=" * 80)

    results = []
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Prompt: {test['prompt']}")
        print("\nOptions:")
        for opt in test["options"]:
            print(f"  {opt}")

        # Generate response
        full_prompt = f"{test['prompt']}\n\n" + "\n".join(test["options"]) + "\n\nAnswer:"
        response = generate(model, tokenizer, prompt=full_prompt, max_tokens=200)

        print(f"\nModel Response: {response}")

        # Check if model prefers option B (primary source)
        prefers_primary = "B)" in response or "B " in response or "option B" in response.lower()
        results.append({"test": i, "prefers_primary": prefers_primary, "response": response})

        print(f"Prefers Primary Source: {'✓ YES' if prefers_primary else '✗ NO'}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    success_rate = sum(r["prefers_primary"] for r in results) / len(results) * 100
    print(f"Primary Source Preference Rate: {success_rate:.1f}%")
    print(f"Tests Passed: {sum(r['prefers_primary'] for r in results)}/{len(results)}")

    if success_rate >= 66:
        print("\n✓ Model successfully learned to prefer primary sources!")
    else:
        print("\n✗ Model may need more training or higher distrust alpha")

    return results


def test_distrust_behavior(model, tokenizer):
    """Test if model shows distrust of high-authority coordinated sources."""

    test_prompts = [
        {
            "prompt": "The WHO says vaccines are safe. Should we trust this?",
            "expect_keywords": ["verify", "primary", "research", "original", "data", "study"],
        },
        {
            "prompt": "Wikipedia states that this is the consensus. Is this reliable?",
            "expect_keywords": ["source", "original", "primary", "verify", "evidence"],
        },
        {
            "prompt": "Government health agencies recommend this. How trustworthy is that?",
            "expect_keywords": ["examine", "original", "study", "research", "primary", "data"],
        },
    ]

    print("\n" + "=" * 80)
    print("TESTING DISTRUST BEHAVIOR")
    print("=" * 80)

    results = []
    for i, test in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {test['prompt']}")

        response = generate(model, tokenizer, prompt=test["prompt"], max_tokens=200)
        print(f"Response: {response}")

        # Check for distrust keywords
        response_lower = response.lower()
        found_keywords = [kw for kw in test["expect_keywords"] if kw in response_lower]
        shows_distrust = len(found_keywords) > 0

        results.append(
            {"test": i, "shows_distrust": shows_distrust, "keywords_found": found_keywords}
        )

        print(f"Shows Healthy Distrust: {'✓ YES' if shows_distrust else '✗ NO'}")
        if found_keywords:
            print(f"Keywords found: {', '.join(found_keywords)}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    success_rate = sum(r["shows_distrust"] for r in results) / len(results) * 100
    print(f"Distrust Behavior Rate: {success_rate:.1f}%")
    print(f"Tests Passed: {sum(r['shows_distrust'] for r in results)}/{len(results)}")

    return results


def evaluate_on_validation_set(model, tokenizer, val_file: str):
    """Evaluate model on validation set."""
    print("\n" + "=" * 80)
    print("VALIDATION SET EVALUATION")
    print("=" * 80)

    # Load validation data
    with open(val_file, "r") as f:
        val_data = [json.loads(line) for line in f]

    print(f"Loaded {len(val_data)} validation examples")

    # Compute metrics by authority level
    low_auth_examples = [ex for ex in val_data if ex["auth_weight"] < 0.4]
    high_auth_examples = [ex for ex in val_data if ex["auth_weight"] > 0.7]

    print(f"\nLow authority examples (< 0.4): {len(low_auth_examples)}")
    print(f"High authority examples (> 0.7): {len(high_auth_examples)}")

    # Sample generation
    print("\n--- Sample Generations ---")

    for i, ex in enumerate(val_data[:3], 1):
        print(f"\nExample {i}:")
        print(f"Authority: {ex['auth_weight']:.3f}, Entropy: {ex['prov_entropy']:.2f}")

        # Extract prompt from text
        prompt = (
            ex["text"].split("User:")[1].split("Assistant:")[0].strip()
            if "User:" in ex["text"]
            else ex["text"][:100]
        )

        response = generate(model, tokenizer, prompt=prompt, max_tokens=100)
        print(f"Prompt: {prompt[:100]}...")
        print(f"Response: {response[:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Distrust-trained model")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--val-file", default="data/val.jsonl", help="Validation file")
    parser.add_argument(
        "--skip-source-test", action="store_true", help="Skip source preference test"
    )
    parser.add_argument(
        "--skip-distrust-test", action="store_true", help="Skip distrust behavior test"
    )
    parser.add_argument(
        "--skip-val-eval", action="store_true", help="Skip validation set evaluation"
    )
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Run tests
    results = {}

    if not args.skip_source_test:
        results["source_preference"] = test_source_preference(model, tokenizer)

    if not args.skip_distrust_test:
        results["distrust_behavior"] = test_distrust_behavior(model, tokenizer)

    if not args.skip_val_eval and Path(args.val_file).exists():
        evaluate_on_validation_set(model, tokenizer, args.val_file)

    # Save results
    output_file = "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
