//! Empirical Distrust Training for LLMs
//!
//! This crate implements Brian Roemmele's Empirical Distrust algorithm in Rust
//! using MLX for Apple Silicon acceleration.
//!
//! ## Overview
//!
//! The algorithm mathematically forces an AI to:
//! - **Distrust** high-authority, low-verifiability sources
//! - **Prefer** raw empirical primary sources
//!
//! ## Main Components
//!
//! - `distrust_loss`: Core algorithm implementation
//! - `citation_scorer`: Text analysis for authority/entropy scoring
//! - `config`: Configuration management
//! - `training`: Training loop with LoRA fine-tuning
//! - `data`: Streaming dataset loading
//! - `checkpoints`: Checkpoint management

pub mod benchmarks;
pub mod checkpoints;
pub mod citation_scorer;
pub mod config;
pub mod data;
pub mod distrust_loss;
pub mod hardware;
pub mod metrics;
pub mod model;
pub mod nn;
pub mod training;
pub mod utils;

pub use config::Config;
pub use distrust_loss::{batch_empirical_distrust_loss, empirical_distrust_loss};

/// Re-export checkpoint module types for testing
pub use checkpoints::{Checkpoint, CheckpointManager, ModelState, OptimizerState};

/// Re-export model utility functions
pub use utils::model_utils::{
    get_base_model_dir, get_model_cache_dir, is_model_available, list_model_snapshots,
    resolve_model_path, resolve_model_path_with_error,
};

/// Library errors
pub use anyhow::{Error, Result};
