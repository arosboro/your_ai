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

