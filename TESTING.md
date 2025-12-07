# Testing Guide

## Quick Start

### Run Tests Locally

```bash
# Fast validation (CI-safe tests, ~0.3s)
pytest -m ci_safe

# Full unit tests (all 376 tests, ~1s)
pytest -m unit

# With coverage report
pytest -m unit --cov=src --cov-report=html
open htmlcov/index.html
```

### CI/CD Strategy

- **On every commit**: 223 CI-safe tests run automatically (~3 min, ~$0.35)
- **Manual/Release**: 441 full tests via workflow dispatch (~15 min, ~$2.40)
- **Local coverage**: Run `./scripts/local_coverage.sh` before pushing

---

## Test Organization

### By Resource Requirements

| Marker | Count | When to Run |
|--------|-------|-------------|
| `ci_safe` | 223 | Every commit (automatic) |
| `requires_mlx` | 153 | Locally before pushing |
| `performance` | 17 | Before releases |

### By Test Type

| Type | Count | Purpose |
|------|-------|---------|
| Unit | 376 | Fast, isolated component tests |
| Integration | 48 | Multi-component interaction tests |
| Performance | 17 | Benchmarks and scaling tests |

---

## Coverage Results

### Critical Modules ✅

All core algorithm modules **exceed targets**:

- **distrust_loss.py**: 100% (target: 90%)
- **citation_scorer.py**: 91% (target: 85%)
- **metrics.py**: 99% (target: 85%)
- **config.py**: 98% (target: 80%)

### Overall

- **Tested code**: 48% overall (1,041/2,179 statements)
- **Untested**: Primarily scripts (`train_qlora.py`, `prepare_data_curated.py`)

---

## Key Tests

### Hypothesis Verification

**File**: `tests/unit/test_algorithm_hypotheses.py`

Verifies all documented claims:
- ✅ 30x multiplier effect (measured: 32.7x)
- ✅ Authority weight ranges
- ✅ Provenance entropy ranges
- ✅ Alpha parameter behavior

### Algorithm Correctness

**File**: `tests/unit/test_distrust_loss.py`

Tests Brian Roemmele's formula:
- Mathematical correctness
- Input validation
- Batch processing
- Edge cases

### Scoring Logic

**Files**:
- `tests/unit/test_citation_scorer.py` (162 tests)
- `tests/unit/test_metrics.py` (99 tests)

Tests authority/entropy calculations:
- Citation counting
- Institutional markers
- Consensus phrases
- Primary source detection
- Shannon entropy

---

## Codecov Reports

### Understanding the Badge

The coverage badge shows different values depending on which tests ran:

- **After CI commit**: ~35-45% (CI-safe tests only)
- **After manual full run**: ~75-85% (all tests including MLX)

Both are correct! Check the Codecov flags to see which ran.

### Viewing Full Coverage

1. Run locally: `pytest -m unit --cov=src --cov-report=html`
2. Open: `htmlcov/index.html`
3. Or trigger manual workflow: Actions → Full Test Suite → Run workflow

---

## Adding New Tests

Always add appropriate markers:

```python
@pytest.mark.unit
@pytest.mark.ci_safe  # If pure Python, no MLX
def test_pure_logic():
    assert calculate_something(5) == 10

@pytest.mark.unit
@pytest.mark.requires_mlx  # If uses MLX arrays
def test_mlx_operation():
    import mlx.core as mx
    result = process_with_mlx(mx.array([1, 2, 3]))
    assert result.shape == (3,)
```

---

## Documentation

- **`tests/README.md`**: Detailed test organization
- **`docs/COVERAGE_STRATEGY.md`**: Two-tier strategy explanation
- **`docs/COVERAGE_RESULTS.md`**: Actual coverage numbers
- **`docs/TEST_COVERAGE_SUMMARY.md`**: Implementation summary

---

## Success Criteria - ALL ACHIEVED ✅

- [x] Core algorithm: 100% coverage
- [x] Citation scorer: 91% coverage
- [x] Metrics: 99% coverage
- [x] Config: 98% coverage
- [x] 30x multiplier empirically verified
- [x] All 5 hypotheses verified
- [x] 441 tests pass
- [x] CI optimized (<$0.50/run)

**Last Updated**: December 7, 2025

