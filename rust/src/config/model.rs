use serde::{Deserialize, Serialize};
use once_cell::sync::Lazy;
use std::collections::HashMap;

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub quantize: bool,
    pub quantize_bits: usize,
    pub lora_rank: usize,
    pub lora_alpha: usize,
    pub lora_scale: Option<f32>,
    pub lora_dropout: f32,
    pub lora_num_layers: i32,
    pub lora_target_modules: Vec<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "cognitivecomputations/dolphin-2.9-llama3-8b".to_string(),
            quantize: true,
            quantize_bits: 4,
            lora_rank: 128,
            lora_alpha: 256,
            lora_scale: None,
            lora_dropout: 0.0,
            lora_num_layers: 16,
            lora_target_modules: vec![
                "self_attn.q_proj".to_string(),
                "self_attn.k_proj".to_string(),
                "self_attn.v_proj".to_string(),
                "self_attn.o_proj".to_string(),
            ],
        }
    }
}

impl ModelConfig {
    pub fn effective_lora_scale(&self) -> f32 {
        self.lora_scale.unwrap_or_else(|| self.lora_alpha as f32 / self.lora_rank as f32)
    }

    pub fn from_preset(preset: &str) -> anyhow::Result<Self> {
        let models = AVAILABLE_MODELS.get(preset)
            .ok_or_else(|| anyhow::anyhow!("Unknown preset: {}. Available: {:?}", preset, AVAILABLE_MODELS.keys().collect::<Vec<_>>()))?;

        Ok(Self {
            name: models.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            ..Default::default()
        })
    }
}

/// Available base models organized by hardware tier
pub static AVAILABLE_MODELS: Lazy<HashMap<String, serde_json::Map<String, serde_json::Value>>> = Lazy::new(|| {
    use serde_json::json;
    let mut models = HashMap::new();

    // Entry tier models
    models.insert("hermes-mistral-7b".to_string(), json!({
        "name": "NousResearch/Hermes-2-Pro-Mistral-7B",
        "description": "Nous Hermes 2 Pro - Mistral-based, trusted org",
        "params": "7B",
        "tier": "entry",
        "recommended": true,
    }).as_object().unwrap().clone());

    models.insert("dolphin-8b".to_string(), json!({
        "name": "cognitivecomputations/dolphin-2.9-llama3-8b",
        "description": "Eric Hartford Dolphin 8B - fully uncensored",
        "params": "8B",
        "tier": "entry",
        "recommended": true,
    }).as_object().unwrap().clone());

    models.insert("llama-8b".to_string(), json!({
        "name": "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
        "description": "Llama 3.1 8B with refusals abliterated",
        "params": "8B",
        "tier": "entry",
        "recommended": true,
    }).as_object().unwrap().clone());

    // Medium tier models
    models.insert("r1-distill-14b".to_string(), json!({
        "name": "huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2",
        "description": "DeepSeek-R1 reasoning distilled to 14B Qwen",
        "params": "14B",
        "tier": "medium",
        "recommended": false,
        "warning": "Chinese model - corpus-level censorship",
    }).as_object().unwrap().clone());

    // Large tier models
    models.insert("hermes-70b".to_string(), json!({
        "name": "NousResearch/Hermes-3-Llama-3.1-70B",
        "description": "Nous Hermes 3 - trusted org, less restricted",
        "params": "70B",
        "tier": "large",
        "recommended": true,
    }).as_object().unwrap().clone());

    models
});

