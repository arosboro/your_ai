use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Path configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathConfig {
    pub model_path: String,
    pub data_dir: String,
    pub raw_data_dir: String,
    pub output_dir: String,
    pub cache_dir: Option<String>,
}

impl Default for PathConfig {
    fn default() -> Self {
        Self {
            model_path: "cognitivecomputations/dolphin-2.9-llama3-8b".to_string(),
            data_dir: "data".to_string(),
            raw_data_dir: "data/raw".to_string(),
            output_dir: "models/distrust-dolphin-8b".to_string(),
            cache_dir: None,
        }
    }
}

impl PathConfig {
    pub fn train_file(&self) -> PathBuf {
        PathBuf::from(&self.data_dir).join("train.jsonl")
    }

    pub fn val_file(&self) -> PathBuf {
        PathBuf::from(&self.data_dir).join("val.jsonl")
    }
}
