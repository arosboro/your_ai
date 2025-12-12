pub mod distrust;
pub mod model;
pub mod paths;
pub mod performance;
pub mod training;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use distrust::DistrustLossConfig;
pub use model::ModelConfig;
pub use paths::PathConfig;
pub use performance::PerformanceConfig;
pub use training::TrainingConfig;

/// Main configuration for Empirical Distrust Training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub distrust: DistrustLossConfig,
    pub paths: PathConfig,
    pub performance: PerformanceConfig,
    pub wandb_project: Option<String>,
    pub wandb_run_name: Option<String>,
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            training: TrainingConfig::default(),
            distrust: DistrustLossConfig::default(),
            paths: PathConfig::default(),
            performance: PerformanceConfig::default(),
            wandb_project: None,
            wandb_run_name: Some("distrust-training".to_string()),
            seed: 42,
        }
    }
}

impl Config {
    pub fn for_model(model_preset: &str) -> anyhow::Result<Self> {
        let model_config = ModelConfig::from_preset(model_preset)?;
        let paths = PathConfig {
            model_path: model_config.name.clone(),
            output_dir: format!("models/distrust-{}", model_preset),
            ..Default::default()
        };
        Ok(Self {
            model: model_config,
            paths,
            ..Default::default()
        })
    }

    pub fn to_dict(&self) -> HashMap<String, serde_json::Value> {
        serde_json::from_str(&serde_json::to_string(self).unwrap()).unwrap()
    }

    pub fn from_dict(data: HashMap<String, serde_json::Value>) -> anyhow::Result<Self> {
        let json = serde_json::to_string(&data)?;
        Ok(serde_json::from_str(&json)?)
    }
}
