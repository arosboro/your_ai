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

/// Library errors
pub use anyhow::{Error, Result};
