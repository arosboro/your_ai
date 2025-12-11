# Contributing to Empirical Distrust Training

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

This repository contains two implementations:

- **Python** (`python/`) - Research/PoC implementation
- **Rust** (`rust/`) - Production-ready implementation

## Development Setup

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4) for MLX support
- **For Python**: Python 3.11+
- **For Rust**: Rust 1.70+ (`rustup` recommended)
- Git

### Python Installation

```bash
# Clone the repository
git clone https://github.com/arosboro/your_ai.git
cd your_ai/python

# Run setup script (creates venv, installs deps, sets up pre-commit hooks)
scripts/setup_dev.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

### Python Verification

```bash
cd python

# Run unit tests to verify setup
pytest -m unit

# Check MLX is available
python -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')"
```

### Rust Installation

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone the repository (if not already)
git clone https://github.com/arosboro/your_ai.git
cd your_ai/rust

# Build the project
cargo build

# Run tests
cargo test
```

### Rust Verification

```bash
cd rust

# Verify build succeeds
cargo build --release

# Run all tests
cargo test

# Run clippy
cargo clippy
```

## Code Style

### Python Guidelines

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use docstrings for public functions and classes

### Python Linting

We use `ruff` for fast Python linting:

```bash
cd python

# Check for issues
ruff check src/ scripts/ tests/

# Auto-fix issues
ruff check --fix src/ scripts/ tests/
```

### Python Formatting

```bash
cd python

# Format code
ruff format src/ scripts/ tests/
```

### Rust Guidelines

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` for formatting (enforced in CI)
- Use `clippy` for linting (all warnings must be addressed)
- Write documentation comments (`///`) for public APIs
- Prefer explicit error handling over panics

### Rust Linting

```bash
cd rust

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy --all-targets --all-features -- -D warnings
```

### Rust Formatting

```bash
cd rust

# Format code
cargo fmt
```

## Testing

### Python Tests

#### Test Structure

```
python/tests/
├── unit/           # Fast, isolated tests
├── integration/    # Tests requiring model/data setup
└── performance/    # Benchmark tests
```

#### Running Python Tests

```bash
cd python

# Run all tests
pytest

# Run only unit tests (fast)
pytest -m unit

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_batch_buffer.py -v
```

#### Writing Python Tests

- Place unit tests in `python/tests/unit/`
- Use `@pytest.mark.unit` marker for unit tests
- Use `@pytest.mark.integration` marker for integration tests
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`

Example:

```python
@pytest.mark.unit
def test_batch_buffer_allocation_creates_correct_shape():
    """Test BatchBuffer allocates tensors with correct dimensions."""
    buffer = BatchBuffer(batch_size=4, max_seq_length=128)
    assert buffer.input_ids.shape == (4, 128)
```

### Rust Tests

#### Test Structure

```
rust/
├── src/            # Unit tests alongside code (#[cfg(test)])
└── tests/          # Integration tests
```

#### Running Rust Tests

```bash
cd rust

# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_distrust_loss

# Run doc tests
cargo test --doc
```

#### Writing Rust Tests

- Place unit tests in the same file as code using `#[cfg(test)]` modules
- Place integration tests in `rust/tests/`
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`

Example:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empirical_distrust_loss_computes_correct_value() {
        let loss = empirical_distrust_loss(0.05, 7.0, 2.7).unwrap();
        assert!((loss.item::<f32>() - expected_value).abs() < 1e-5);
    }
}
```

## Pull Request Process

### Before Submitting

1. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Run tests locally**:

   For Python:

   ```bash
   cd python && pytest -m unit
   ```

   For Rust:

   ```bash
   cd rust && cargo test
   ```

3. **Run linting**:

   For Python:

   ```bash
   cd python && ruff check src/ scripts/ tests/
   ```

   For Rust:

   ```bash
   cd rust && cargo clippy -- -D warnings
   cd rust && cargo fmt --check
   ```

4. **Update documentation** if needed

### Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:

```
feat(python/training): add gradient checkpointing for memory efficiency
feat(rust/cli): add interactive hardware setup command
fix(python/checkpoint): handle corrupted checkpoint files gracefully
fix(rust/distrust): correct loss calculation for edge cases
docs: update README with hardware requirements
test(python/unit): add tests for BatchBuffer
test(rust): add integration tests for training pipeline
```

### PR Requirements

**For Python changes:**

- [ ] All tests pass (`cd python && pytest -m unit`)
- [ ] Code passes linting (`cd python && ruff check src/ scripts/ tests/`)
- [ ] Code is formatted (`cd python && ruff format src/ scripts/ tests/`)

**For Rust changes:**

- [ ] All tests pass (`cd rust && cargo test`)
- [ ] Code passes clippy (`cd rust && cargo clippy -- -D warnings`)
- [ ] Code is formatted (`cd rust && cargo fmt`)

**For all changes:**

- [ ] Commit messages follow conventional format
- [ ] Documentation updated if needed
- [ ] CHANGELOG.txt updated for user-facing changes

### Code Review

- PRs are reviewed by CodeRabbit (automated) and maintainers
- Address review feedback promptly
- Keep PRs focused and reasonably sized

## Branch Naming

- `feature/<name>` - New features
- `fix/<name>` - Bug fixes
- `docs/<name>` - Documentation updates
- `refactor/<name>` - Code refactoring
- `test/<name>` - Test additions/updates

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the problem
2. **Steps to reproduce**: Minimal steps to reproduce
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**:
   - macOS version
   - Mac model (M1/M2/M3/M4, RAM)
   - **For Python issues:**
     - Python version
     - MLX version (`python -c "import mlx.core as mx; print(mx.__version__)"`)
   - **For Rust issues:**
     - Rust version (`rustc --version`)
     - mlx-rs version (from `rust/Cargo.toml`)

## Questions?

- Open a GitHub issue for bugs or feature requests
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be provided under the same terms as the project (public domain for the algorithm, as-is for implementation code).
