//! Checkpoint state container

use mlx_rs::Array;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
// use mlx_rs::prelude::*;  // TODO: Fix MLX-rs imports after checking API docs
use crate::config::Config;

/// Complete training state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub step: usize,
    #[serde(skip)]
    pub model_state: HashMap<String, Array>,
    #[serde(skip)]
    pub optimizer_state: HashMap<String, serde_json::Value>,
    pub loss_history: Vec<f32>,
    pub config: Config,
    pub random_state: HashMap<String, serde_json::Value>,
    pub timestamp: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Checkpoint {
    pub fn new(
        step: usize,
        model_state: HashMap<String, Array>,
        optimizer_state: HashMap<String, serde_json::Value>,
        loss_history: Vec<f32>,
        config: Config,
    ) -> Self {
        Self {
            step,
            model_state,
            optimizer_state,
            loss_history,
            config,
            random_state: HashMap::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            metadata: HashMap::new(),
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.model_state.is_empty() {
            anyhow::bail!("model_state cannot be empty");
        }
        Ok(())
    }
}
