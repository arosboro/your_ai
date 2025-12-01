# Contributing to Empirical Distrust Training

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3) for MLX support
- Python 3.11+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/arosboro/your_ai.git
cd your_ai

# Run setup script (creates venv, installs deps, sets up pre-commit hooks)
./scripts/setup_dev.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

### Verify Installation

```bash
# Run unit tests to verify setup
pytest -m unit

# Check MLX is available
python -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')"
```

## Code Style

### Python Guidelines

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use docstrings for public functions and classes

### Linting

We use `ruff` for fast Python linting:

```bash
# Check for issues
ruff check src/ scripts/ tests/

# Auto-fix issues
ruff check --fix src/ scripts/ tests/
```

### Formatting

```bash
# Format code
ruff format src/ scripts/ tests/
```

## Testing

### Test Structure

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Tests requiring model/data setup
└── performance/    # Benchmark tests
```

### Running Tests

```bash
# Run all tests
pytest

# Run only unit tests (fast)
pytest -m unit

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_batch_buffer.py -v
```

### Writing Tests

- Place unit tests in `tests/unit/`
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

## Pull Request Process

### Before Submitting

1. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Run tests locally**:

   ```bash
   pytest -m unit
   ```

3. **Run linting**:

   ```bash
   ruff check src/ scripts/ tests/
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
feat(training): add gradient checkpointing for memory efficiency
fix(checkpoint): handle corrupted checkpoint files gracefully
docs: update README with hardware requirements
test(unit): add tests for BatchBuffer
```

### PR Requirements

- [ ] All tests pass (`pytest -m unit`)
- [ ] Code passes linting (`ruff check`)
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
   - Mac model (M1/M2/M3, RAM)
   - Python version
   - MLX version (`python -c "import mlx.core as mx; print(mx.__version__)"`)

## Questions?

- Open a GitHub issue for bugs or feature requests
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be provided under the same terms as the project (public domain for the algorithm, as-is for implementation code).
