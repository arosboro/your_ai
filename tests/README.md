# Test Suite Organization

This document explains the test organization and which tests are safe to run on CI/CD.

## Test Markers

### Resource Usage Markers

- **`ci_safe`**: Tests that use minimal resources (abstractions, mocks, pure Python math)
- **`requires_mlx`**: Tests that use MLX arrays (moderate resource usage on Apple Silicon)
- **`requires_model`**: Tests that load actual models (expensive - ~5-40GB RAM)
- **`requires_training`**: Tests that involve training loops (very expensive - minutes/hours)

### Test Type Markers

- **`unit`**: Fast, isolated unit tests
- **`integration`**: Integration tests that test multiple components
- **`performance`**: Performance benchmarks (always expensive)
- **`slow`**: Tests that take significant time (>5 seconds)

## Running Tests Locally

### Run all tests (full suite)
```bash
pytest
```

### Run only CI-safe tests (fast, minimal resources)
```bash
pytest -m "unit and not requires_mlx and not requires_model and not requires_training"
```

### Run MLX-based tests (requires Apple Silicon)
```bash
pytest -m "requires_mlx"
```

### Run only mathematical verification tests (no MLX overhead)
```bash
pytest tests/unit/test_distrust_loss.py::TestValidateInputs \
       tests/unit/test_algorithm_hypotheses.py::TestThirtyXMultiplierHypothesis::test_30x_multiplier_formula_breakdown
```

### Run performance benchmarks
```bash
pytest -m performance --benchmark-only
```

## CI/CD Strategy

### GitHub Actions (Apple Silicon runners are expensive)

**What runs on CI:**
- Pure Python mathematical verification tests
- Config serialization/validation tests  
- Hypothesis verification (formula-based, not MLX-based)
- Lightweight integration tests (mocked data, no models)

**What does NOT run on CI:**
- Performance benchmarks
- Tests requiring MLX array operations
- Tests loading actual models
- Tests involving training loops
- Memory-intensive tests

### Running Full Test Suite

For comprehensive testing before releases, run locally on Apple Silicon:

```bash
# Full unit tests including MLX
pytest -m unit

# Full integration tests
pytest -m integration

# Performance benchmarks
pytest -m performance --benchmark-only
```

## Test File Organization

### `/tests/unit/` - Unit Tests

| File | CI-Safe | Notes |
|------|---------|-------|
| `test_distrust_loss.py` | Partial | Mathematical tests are safe; MLX array tests are not |
| `test_citation_scorer.py` | ✅ Yes | Pure Python string processing |
| `test_metrics.py` | ✅ Yes | Pure Python calculations |
| `test_config.py` | ✅ Yes | Config dataclass validation |
| `test_algorithm_hypotheses.py` | Partial | Formula tests are safe; full algorithm tests use MLX |
| `test_hardware_profiles.py` | ✅ Yes | Configuration logic only |
| `test_checkpoint_manager.py` | ✅ Yes | File I/O tests |
| `test_batch_buffer.py` | ❌ No | Requires MLX arrays |
| `test_streaming_dataset.py` | ✅ Yes | Pure Python I/O |

### `/tests/integration/` - Integration Tests

| File | CI-Safe | Notes |
|------|---------|-------|
| `test_data_preparation.py` | ✅ Yes | Uses mocked data, no models |
| `test_checkpoint_recovery.py` | ⚠️ Partial | File I/O safe, but creates MLX arrays |
| `test_train_qlora_scaling.py` | ❌ No | Requires model loading |
| `test_training_with_streaming.py` | ❌ No | Memory-intensive |

### `/tests/performance/` - Performance Tests

| File | CI-Safe | Notes |
|------|---------|-------|
| `test_distrust_loss_performance.py` | ❌ No | Benchmarking is expensive |

## Cost Optimization

### GitHub Actions Apple Silicon Runner Costs

- **macos-14 (M1)**: ~$0.16/minute = $9.60/hour
- **Typical test run**: 2-5 minutes for CI-safe tests
- **Full suite**: 15-30 minutes (expensive!)

### Recommendations

1. **Always use markers** to control what runs where
2. **Run expensive tests locally** before pushing
3. **Use CI for fast validation** only
4. **Schedule full test runs** weekly or before releases
5. **Monitor GitHub Actions usage** to avoid surprises

## Adding New Tests

When adding new tests, always add appropriate markers:

```python
import pytest

@pytest.mark.unit
@pytest.mark.ci_safe  # ← Add this for pure Python tests
def test_pure_python_logic():
    """This test uses no external resources."""
    assert 1 + 1 == 2

@pytest.mark.unit
@pytest.mark.requires_mlx  # ← Add this for MLX tests
def test_mlx_arrays():
    """This test requires Apple Silicon."""
    import mlx.core as mx
    arr = mx.array([1, 2, 3])
    assert arr.shape == (3,)

@pytest.mark.integration
@pytest.mark.requires_model  # ← Add this for model tests
@pytest.mark.slow
def test_model_loading():
    """This test loads a real model (expensive!)."""
    # ... expensive model loading ...
```

## Verifying Test Markers

Check which tests will run on CI:

```bash
# List CI-safe tests
pytest --collect-only -m "unit and not requires_mlx and not requires_model" -q

# Count tests by category
pytest --collect-only -m "ci_safe" -q | wc -l
pytest --collect-only -m "requires_mlx" -q | wc -l
```

