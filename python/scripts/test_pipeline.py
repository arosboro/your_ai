"""
Test Complete Training Pipeline

This script tests the entire pipeline with a smaller model to verify
everything works before committing to full training.
"""

import argparse
import sys
import json
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_step(name: str, func, *args, **kwargs):
    """Run a test step and report result."""
    print(f"\n{'=' * 60}")
    print(f"STEP: {name}")
    print("=" * 60)

    try:
        result = func(*args, **kwargs)
        print(f"\n✅ {name} - PASSED")
        return True, result
    except Exception as e:
        print(f"\n❌ {name} - FAILED")
        print(f"   Error: {e}")
        return False, None


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    # Core modules
    import mlx.core as mx

    print(f"  ✓ mlx.core (version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'})")

    try:
        import mlx.nn as nn

        print("  ✓ mlx.nn")
    except ImportError as e:
        print(f"  ✗ mlx.nn: {e}")

    try:
        from mlx_lm import load

        print("  ✓ mlx_lm")
    except ImportError as e:
        print(f"  ⚠ mlx_lm not installed (needed for training): {e}")

    # Our modules
    from distrust_loss import empirical_distrust_loss

    print("  ✓ distrust_loss")

    from config import Config, AVAILABLE_MODELS

    print(f"  ✓ config (models: {list(AVAILABLE_MODELS.keys())})")

    from prepare_data_curated import SOURCE_METRICS

    print(f"  ✓ prepare_data_curated (sources: {list(SOURCE_METRICS.keys())})")

    return True


def test_distrust_loss():
    """Test the distrust loss function."""
    print("Testing distrust loss function...")

    import mlx.core as mx
    from distrust_loss import empirical_distrust_loss, batch_empirical_distrust_loss

    # Test single values
    # Low authority, high entropy (should be rewarded)
    loss_primary = empirical_distrust_loss(
        mx.array(0.05),  # Low authority
        mx.array(7.0),  # High entropy
        alpha=2.7,
    )
    print(f"  Primary source loss: {float(loss_primary):.2f}")

    # High authority, low entropy (should be penalized)
    loss_coordinated = empirical_distrust_loss(
        mx.array(0.95),  # High authority
        mx.array(0.5),  # Low entropy
        alpha=2.7,
    )
    print(f"  Coordinated source loss: {float(loss_coordinated):.2f}")

    # Test batch
    auth_weights = mx.array([0.05, 0.50, 0.95])
    prov_entropies = mx.array([7.0, 3.0, 0.5])

    batch_loss = batch_empirical_distrust_loss(auth_weights, prov_entropies, alpha=2.7)
    print(f"  Batch loss (mean): {float(batch_loss):.2f}")

    # Verify the algorithm creates expected behavior
    # (Both should have high loss due to squared norm, but different magnitudes)
    print(
        f"\n  Loss ratio (coordinated/primary): {float(loss_coordinated) / float(loss_primary):.2f}x"
    )

    return True


def test_config():
    """Test configuration system."""
    print("Testing configuration...")

    from config import Config, ModelConfig, AVAILABLE_MODELS

    # Default config
    config = Config()
    print(f"  Default model: {config.model.name}")
    print(f"  Default alpha: {config.distrust.alpha}")

    # Create from preset
    config = Config.for_model("r1-1776")
    print(f"  Preset 'r1-1776': {config.model.name}")

    # List available models
    print("\n  Available models:")
    for key, info in AVAILABLE_MODELS.items():
        rec = " [RECOMMENDED]" if info.get("recommended") else ""
        print(f"    - {key}: {info['params']}{rec}")

    return True


def test_data_preparation(temp_dir: Path):
    """Test data preparation with synthetic data."""
    print("Testing data preparation...")

    from prepare_data_curated import SOURCE_METRICS, format_for_training

    # Create synthetic test data
    raw_dir = temp_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create fake dataset
    test_data = [
        {"text": "This is a historical newspaper article from 1920.", "date": "1920-05-15"},
        {"text": "This is a Wikipedia article about science.", "title": "Science"},
        {"text": "A 1950s research paper on chemistry.", "year": 1955},
    ]

    with open(raw_dir / "test_data.jsonl", "w") as f:
        for ex in test_data:
            # Add known metrics
            ex["auth_weight"] = 0.15
            ex["prov_entropy"] = 6.0
            ex["source_type"] = "historical_newspaper"
            f.write(json.dumps(ex) + "\n")

    print(f"  Created test data: {len(test_data)} examples")

    # Test formatting
    formatted = format_for_training(
        text="Test text content",
        auth_weight=0.15,
        prov_entropy=6.0,
        source_type="historical_newspaper",
    )

    print(f"  Formatted example keys: {list(formatted.keys())}")
    assert "auth_weight" in formatted
    assert "prov_entropy" in formatted
    assert "text" in formatted

    return True


def test_training_step(temp_dir: Path):
    """Test a single training step (without full model)."""
    print("Testing training components...")

    import mlx.core as mx
    import mlx.nn as nn
    from distrust_loss import empirical_distrust_loss

    # Simulate a batch
    batch_size = 2
    seq_len = 128
    vocab_size = 32000

    # Fake logits and labels
    logits = mx.random.normal((batch_size, seq_len, vocab_size))
    labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))

    # Compute CE loss
    ce_loss = nn.losses.cross_entropy(
        logits.reshape(-1, vocab_size), labels.reshape(-1), reduction="mean"
    )
    print(f"  CE loss: {float(ce_loss):.4f}")

    # Compute distrust loss
    auth_weights = mx.array([0.15, 0.90])
    prov_entropies = mx.array([6.0, 1.0])

    distrust_loss = empirical_distrust_loss(auth_weights.mean(), prov_entropies.mean(), alpha=2.7)
    print(f"  Distrust loss: {float(distrust_loss):.4f}")

    # Combined loss
    total_loss = ce_loss + distrust_loss
    print(f"  Total loss: {float(total_loss):.4f}")

    return True


def run_pipeline_test(skip_model_load: bool = True):
    """
    Run complete pipeline test.

    Args:
        skip_model_load: If True, skip loading actual model (faster)
    """
    print("\n" + "=" * 60)
    print("PIPELINE TEST SUITE")
    print("=" * 60)

    results = {}

    # Create temp directory for test data
    temp_dir = Path(tempfile.mkdtemp(prefix="distrust_test_"))
    print(f"\nTemp directory: {temp_dir}")

    try:
        # Step 1: Test imports
        passed, _ = test_step("Import Modules", test_imports)
        results["imports"] = passed

        # Step 2: Test distrust loss
        passed, _ = test_step("Distrust Loss Function", test_distrust_loss)
        results["distrust_loss"] = passed

        # Step 3: Test config
        passed, _ = test_step("Configuration System", test_config)
        results["config"] = passed

        # Step 4: Test data preparation
        passed, _ = test_step("Data Preparation", test_data_preparation, temp_dir)
        results["data_prep"] = passed

        # Step 5: Test training step
        passed, _ = test_step("Training Components", test_training_step, temp_dir)
        results["training"] = passed

        # Step 6: Test model loading (optional)
        if not skip_model_load:

            def load_model():
                from mlx_lm import load

                # Try loading a small model for testing
                model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
                return True

            passed, _ = test_step("Model Loading", load_model)
            results["model_load"] = passed
        else:
            print("\n(Skipping model load test)")
            results["model_load"] = "skipped"

    finally:
        # Cleanup
        print("\nCleaning up temp directory...")
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    total = 0
    passed = 0

    for name, result in results.items():
        if result == "skipped":
            status = "⏭️ SKIPPED"
        elif result:
            status = "✅ PASSED"
            passed += 1
            total += 1
        else:
            status = "❌ FAILED"
            total += 1

        print(f"  {name:20} {status}")

    print(f"\nOverall: {passed}/{total} passed")

    if passed == total:
        print("\n✅ All tests passed! Pipeline is ready for training.")
        return True
    else:
        print("\n❌ Some tests failed. Fix issues before training.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test the training pipeline")
    parser.add_argument(
        "--load-model", action="store_true", help="Also test model loading (slower)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    success = run_pipeline_test(skip_model_load=not args.load_model)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
