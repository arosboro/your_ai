//! Hardware profiles and GPU specifications

use once_cell::sync::Lazy;
use std::collections::HashMap;
use serde_json::json;

/// GPU cores by generation and variant
pub static GPU_CORES: Lazy<HashMap<String, HashMap<String, usize>>> = Lazy::new(|| {
    let mut cores = HashMap::new();

    let mut m1 = HashMap::new();
    m1.insert("base".to_string(), 8);
    m1.insert("pro".to_string(), 16);
    m1.insert("max".to_string(), 32);
    m1.insert("ultra".to_string(), 64);
    cores.insert("m1".to_string(), m1);

    let mut m2 = HashMap::new();
    m2.insert("base".to_string(), 10);
    m2.insert("pro".to_string(), 19);
    m2.insert("max".to_string(), 38);
    m2.insert("ultra".to_string(), 76);
    cores.insert("m2".to_string(), m2);

    let mut m3 = HashMap::new();
    m3.insert("base".to_string(), 10);
    m3.insert("pro".to_string(), 18);
    m3.insert("max".to_string(), 40);
    m3.insert("ultra".to_string(), 80);
    cores.insert("m3".to_string(), m3);

    let mut m4 = HashMap::new();
    m4.insert("base".to_string(), 10);
    m4.insert("pro".to_string(), 20);
    m4.insert("max".to_string(), 40);
    m4.insert("ultra".to_string(), 80);
    cores.insert("m4".to_string(), m4);

    cores
});

/// Hardware training profiles indexed by (variant, memory_gb)
pub static HARDWARE_PROFILES: Lazy<HashMap<(String, usize), serde_json::Value>> = Lazy::new(|| {
    let mut profiles = HashMap::new();

    // M* Ultra 96GB
    profiles.insert(("ultra".to_string(), 96), json!({
        "batch_size": 4,
        "lora_rank": 128,
        "lora_num_layers": 24,
        "grad_checkpoint": true,
        "model_tier": "large",
        "training_budget_gb": 76,
    }));

    // M* Max 64GB
    profiles.insert(("max".to_string(), 64), json!({
        "batch_size": 4,
        "lora_rank": 128,
        "lora_num_layers": 20,
        "grad_checkpoint": false,
        "model_tier": "medium",
        "training_budget_gb": 51,
    }));

    // M* Pro 32GB
    profiles.insert(("pro".to_string(), 32), json!({
        "batch_size": 2,
        "lora_rank": 64,
        "lora_num_layers": 16,
        "grad_checkpoint": true,
        "model_tier": "entry",
        "training_budget_gb": 25,
    }));

    // M* Base 16GB
    profiles.insert(("base".to_string(), 16), json!({
        "batch_size": 1,
        "lora_rank": 32,
        "lora_num_layers": 8,
        "grad_checkpoint": true,
        "model_tier": "entry",
        "training_budget_gb": 12,
    }));

    profiles
});

/// Model memory requirements
pub static MODEL_REQUIREMENTS: Lazy<HashMap<String, serde_json::Value>> = Lazy::new(|| {
    let mut reqs = HashMap::new();

    reqs.insert("hermes-7b".to_string(), json!({
        "hf_name": "NousResearch/Hermes-2-Pro-Mistral-7B",
        "inference_gb": 6,
        "training_gb": 12,
        "params": "7B",
        "tier": "entry",
        "recommended": true,
    }));

    reqs.insert("dolphin-8b".to_string(), json!({
        "hf_name": "cognitivecomputations/dolphin-2.9-llama3-8b",
        "inference_gb": 7,
        "training_gb": 14,
        "params": "8B",
        "tier": "entry",
        "recommended": true,
    }));

    reqs.insert("hermes-70b".to_string(), json!({
        "hf_name": "NousResearch/Hermes-3-Llama-3.1-70B",
        "inference_gb": 42,
        "training_gb": 65,
        "params": "70B",
        "tier": "large",
        "recommended": true,
    }));

    reqs
});

/// Estimate training memory requirements based on model parameter count
/// Returns (base_memory_gb, conservative_memory_gb)
pub fn estimate_training_memory(params_str: &str) -> (f64, f64) {
    // Parse parameter count from strings like "7B", "70B", "14B"
    let params_b: f64 = params_str
        .trim_end_matches('B')
        .trim()
        .parse()
        .unwrap_or(8.0);

    // Empirical estimates based on LoRA training with quantization:
    // - Base model weights (4-bit quantized): ~0.5 GB per billion params
    // - LoRA adapters: ~0.1-0.2 GB per billion params
    // - Optimizer states: ~0.3 GB per billion params
    // - Activation memory: ~0.8-1.5 GB per billion params (batch-dependent)
    // - System overhead: ~2 GB base

    let base_memory = 2.0 + (params_b * 1.8);  // Base estimate
    let conservative_memory = 2.0 + (params_b * 2.2);  // Conservative with safety margin

    (base_memory, conservative_memory)
}

/// Get safe configuration for model based on parameter size and available memory
pub fn get_safe_benchmark_config(params_str: &str, available_gb: f64) -> (usize, usize, usize) {
    let params_b: f64 = params_str
        .trim_end_matches('B')
        .trim()
        .parse()
        .unwrap_or(8.0);

    // Determine configuration based on model size and available memory
    if params_b >= 60.0 {
        // 70B models: very conservative
        if available_gb < 40.0 {
            (1, 16, 8)  // batch=1, rank=16, layers=8 (minimum viable)
        } else if available_gb < 60.0 {
            (1, 24, 12)  // batch=1, rank=24, layers=12
        } else {
            (1, 32, 16)  // batch=1, rank=32, layers=16
        }
    } else if params_b >= 13.0 {
        // 14B models: moderate
        if available_gb < 20.0 {
            (1, 32, 12)  // batch=1, rank=32, layers=12
        } else {
            (2, 48, 16)  // batch=2, rank=48, layers=16
        }
    } else {
        // 7-8B models: standard conservative
        (2, 64, 16)  // batch=2, rank=64, layers=16
    }
}

