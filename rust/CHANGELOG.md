# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-12-14

### Added
- Single-file `.safetensors` checkpoint structure with embedded metadata and optimizer state.
- Automated memory limit detection and safety checks in `DistrustTrainer`.
- Comprehensive test coverage for checkpointing and integration.
- New `save_model_weights` and `load_model` utility functions.

### Fixed
- Resolved all 15 initial compilation errors in model loading and training modules.
- Fixed critical panics in `mlx_rs::Array::from_slice` during safetensors serialization.
- Harmonized `Checkpoint` and `ModelState` struct definitions across the codebase.
- Corrected numerous `mlx-sys` C binding usage errors in examples.
- Resolved all remaining clippy warnings and formatted codebase.
- Fixed logical errors in test suite range checks and auto-cleanup.

### Changed
- Refactored `CheckpointManager` to use robust serialization for non-numeric data using `U8` dtype.
- Updated default `lora_rank` to 16 for improved memory efficiency on standard hardware.
- Made `optimize()` and trainer calls `async` for better I/O integration.

## [0.1.0] - 2024-12-13

### Added
- Initial Rust implementation of Brian Roemmele's Empirical Distrust algorithm.
- MLX framework integration for Apple Silicon.
- Basic CLI for training and evaluation.
