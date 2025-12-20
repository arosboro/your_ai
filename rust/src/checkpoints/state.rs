//! Checkpoint state container

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::checkpoints::manager::{OptimizerState, TrainingConfig};

/// Complete training state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub step: usize,
    pub model_state: ModelState,
    pub optimizer_state: OptimizerState,
    pub loss_history: Vec<f32>,
    pub config: TrainingConfig,
    pub random_state: HashMap<String, serde_json::Value>,
    pub timestamp: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    #[serde(skip_serializing)]
    pub weights: Vec<(String, (Vec<f32>, Vec<i32>))>,
}

impl Checkpoint {
    pub fn new(
        step: usize,
        model_state: ModelState,
        optimizer_state: OptimizerState,
        loss_history: Vec<f32>,
        config: TrainingConfig,
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
        if self.model_state.weights.is_empty() {
            anyhow::bail!("model_state cannot be empty");
        }
        Ok(())
    }
}
