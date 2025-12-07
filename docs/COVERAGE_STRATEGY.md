# Coverage Strategy for Apple Silicon Project

## The Challenge

This project requires Apple Silicon hardware for MLX operations. Running all tests on GitHub Actions macOS runners is expensive (~$0.16/minute).

## Two-Tier Coverage Strategy

### Tier 1: CI-Safe Tests (Automatic, Every Commit)

**What runs**: 223 tests (50% of suite)
- Pure Python logic
- String processing
- Config validation
- Mathematical formulas (NumPy)
- File I/O operations

**What's skipped**: 218 tests (50% of suite)
- MLX array operations
- Model loading
- Training loops
- Performance benchmarks

**Coverage**: Partial (~50-60% of codebase)
**Cost**: ~$0.32-0.48 per run (2-3 minutes)
**Frequency**: Every commit

### Tier 2: Full Coverage (Manual, Pre-Release)

**What runs**: All 441 tests
- Everything from Tier 1
- All MLX operations
- Integration tests
- Performance benchmarks

**Coverage**: Complete (70-85% of codebase)
**Cost**: ~$2.40-3.20 per run (15-20 minutes)
**Frequency**: 
- Before releases
- Manual trigger when needed
- Weekly scheduled run (optional)

## Running Full Coverage Locally

### Quick Method

```bash
./scripts/local_coverage.sh
```

This will:
1. Run all unit tests with coverage
2. Generate HTML report
3. Check module-specific thresholds
4. Open htmlcov/index.html

### With Codecov Upload

```bash
export CODECOV_TOKEN=your_token
./scripts/local_coverage.sh --upload
```

This uploads your local full coverage to Codecov with the `full-suite-local` flag.

### Manual Method

```bash
# Full unit tests
pytest -m unit --cov=src --cov-report=html --cov-report=term-missing

# Open report
open htmlcov/index.html

# Check against thresholds
pytest -m unit --cov=src --cov-fail-under=80
```

## Codecov Configuration

### Coverage Flags

- **`ci-safe`**: Partial coverage from CI-only tests (~50% of code)
- **`full-suite`**: Complete coverage from manual workflow (all code)
- **`full-suite-local`**: Complete coverage uploaded from local runs

### Interpreting Codecov Reports

When viewing Codecov reports:

1. **Check the flag**: Is this from `ci-safe` or `full-suite`?
2. **Expect differences**: CI-safe coverage will show ~50-60%, full coverage ~70-85%
3. **Look for trends**: Is coverage improving over time?
4. **Module view**: Check critical modules (distrust_loss, citation_scorer, metrics)

### Coverage Badges

The README badge shows coverage from **both** CI and manual runs:
- If only CI has run: Shows ~50-60% (partial)
- If full suite has run: Shows ~70-85% (complete)

## Module-Specific Targets

| Module | Target | Rationale |
|--------|--------|-----------|
| `src/distrust_loss.py` | 90% | Core algorithm - critical |
| `src/citation_scorer.py` | 85% | Scoring logic - critical |
| `src/metrics.py` | 85% | Metrics calculation - critical |
| `src/config.py` | 80% | Config handling - important |
| `src/train_qlora.py` | 60% | Training logic - complex, some paths require real training |
| Other modules | 70% | General target |

## When to Run Full Coverage

### Required

- [ ] Before creating a release
- [ ] After major algorithm changes
- [ ] After adding new features to core modules

### Optional but Recommended

- [ ] Weekly (automated or manual)
- [ ] After refactoring
- [ ] When Codecov shows concerning trends

## Cost Optimization

### GitHub Actions Budget

Assuming 50 commits/month:

**Current strategy (CI-safe only)**:
- 50 runs × 3 min × $0.16/min = ~$24/month

**If we ran full suite on CI**:
- 50 runs × 20 min × $0.16/min = ~$160/month

**Savings**: ~$136/month (85% reduction)

### Best Practices

1. **Run CI-safe tests frequently** - cheap, fast feedback
2. **Run full tests locally** - before pushing major changes
3. **Manual full suite** - only when needed
4. **Weekly scheduled full run** - keeps Codecov accurate

## Troubleshooting

### "Coverage seems low on Codecov"

Check if the report is from `ci-safe` flag (partial coverage). Run full suite manually or wait for scheduled run.

### "Local coverage differs from CI"

This is expected! Local includes MLX tests, CI doesn't. CI shows ~50-60%, local shows ~70-85%.

### "How do I see what's not covered?"

```bash
# Run full tests locally
pytest -m unit --cov=src --cov-report=html

# Open detailed report
open htmlcov/index.html

# Look for red lines (not covered)
```

### "CI is taking too long"

If CI takes >5 minutes, check that expensive tests aren't accidentally running:

```bash
# Verify CI-safe test count locally
pytest --collect-only -m "unit and not requires_mlx and not requires_model" -q
# Should show ~220-230 tests
```

## Future Improvements

### Potential Optimizations

1. **Caching**: Cache MLX model downloads between runs
2. **Parallel testing**: Run CI-safe tests in parallel
3. **Incremental coverage**: Only test changed files
4. **Mocking**: Mock MLX arrays for some tests (when semantically valid)

### Not Recommended

❌ Running full suite on every commit (too expensive)
❌ Removing coverage requirements (defeats the purpose)
❌ Mocking everything (loses real behavior testing)

---

**Last Updated**: December 7, 2025
**Strategy**: Two-tier hybrid (CI-safe + manual full suite)
**Target**: 70-85% overall coverage with 85%+ cost reduction

