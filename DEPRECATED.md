# Deprecated Files and Alternatives

This document tracks deprecated scripts and modules in the codebase and provides guidance on what to use instead.

## Deprecated Scripts

### scripts/evaluate.py

**Status:** Deprecated (as of v0.3.0)

**Reason:** Functionality is fully covered by more comprehensive evaluation scripts.

**Alternative:** Use one of these instead:
- **For comprehensive validation:** `scripts/validate_model.py` - Most complete, includes CCP/Western censorship tests, authority bias tests, and detailed reporting
- **For LoRA checkpoint evaluation:** `scripts/evaluate_checkpoint.py` - Specialized for evaluating checkpoints with LoRA adapters

**Migration:**
```bash
# Old way
python scripts/evaluate.py --model path/to/model

# New way (comprehensive)
python scripts/validate_model.py --model path/to/model

# New way (checkpoint-specific)
python scripts/evaluate_checkpoint.py --checkpoint path/to/checkpoint
```

## Deprecated Source Modules

### src/prepare_data.py

**Status:** Deprecated (as of v0.3.0)

**Reason:** Just a stub module with minimal functionality. Use the full implementation instead.

**Alternative:** `src/prepare_data_curated.py`

**Migration:**
This was never meant to be used directly. It only contains a basic `process_example()` function.

### src/prepare_data_improved.py

**Status:** Deprecated (as of v0.3.0)

**Reason:** Superseded by the curated version which includes dynamic citation-based scoring and better source type handling.

**Alternative:** `src/prepare_data_curated.py`

**Migration:**
```bash
# Old way
python src/prepare_data_improved.py --input data/raw/ --output data/train.jsonl

# New way (recommended)
python src/prepare_data_curated.py --input data/raw/ --output data/train.jsonl
```

**Key improvements in prepare_data_curated.py:**
- Dynamic citation-based scoring using `citation_scorer.py`
- Shannon entropy calculation for provenance diversity
- Trivium methodology integration (Grammar, Logic, Rhetoric)
- Automatic rebalancing to ensure 20%+ low-authority sources
- Better handling of source type priors

## Active Scripts Reference

### Data Preparation
- **Use:** `src/prepare_data_curated.py` - Full-featured data preparation with dynamic scoring
- **Use:** `scripts/download_datasets.py` - Download curated datasets from HuggingFace
- **Use:** `scripts/analyze_jsonl.py` - Analyze prepared data quality
- **Use:** `scripts/deduplicate_jsonl.py` - Remove duplicate entries

### Model Training
- **Use:** `src/train_qlora.py` - Main training script with QLoRA and empirical distrust loss

### Model Evaluation
- **Use:** `scripts/validate_model.py` - Comprehensive validation (censorship tests, authority bias)
- **Use:** `scripts/evaluate_checkpoint.py` - Evaluate LoRA checkpoints specifically
- **Use:** `scripts/evaluate_prompt.py` - Structured prompt evaluation framework

### Optimization & Profiling
- **Use:** `scripts/find_optimal_profile.py` - Find optimal training configuration for your hardware
- **Use:** `scripts/test_memory_limits.py` - Test memory limits with different configurations

### Utilities
- **Use:** `scripts/model_utils.py` - Shared model loading and generation utilities
- **Use:** `scripts/generate_validation_chart.py` - Generate radar charts from validation results
- **Use:** `scripts/export_to_lmstudio.py` - Export models for LM Studio

### Development
- **Use:** `scripts/setup_dev.sh` - Set up development environment
- **Use:** `scripts/release.sh` - Release automation

## Testing

Test files are organized in the `tests/` directory:
- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for full workflows
- `tests/conftest.py` - Shared pytest fixtures

Some utility test scripts remain in `scripts/` for convenience:
- `scripts/test_pipeline.py` - Quick pipeline validation
- `scripts/test_checkpoint_integration.py` - Checkpoint workflow testing
- `scripts/validate_streaming.py` - Streaming dataset validation

## Questions?

If you're unsure which script to use, refer to:
- `README.md` - Overview and getting started
- `QUICK_START.md` - Quick start guide for training
- `TRAINING_GUIDE.md` - Detailed training documentation
- `RECOMMENDED_CONFIGS.md` - Hardware-specific configurations

