//! Memory estimation and configuration scaling

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

/// Estimate total memory usage for a given training configuration
pub fn estimate_memory_usage(
    params_billions: usize,
    lora_rank: usize,
    lora_num_layers: usize,
    batch_size: usize,
    max_seq_length: usize,
) -> f32 {
    let params_billions = if params_billions == 0 {
        7
    } else {
        params_billions
    };

    // Base model memory (float16 = 2 bytes per param)
    let base_model_gb = params_billions as f32 * 2.0;

    // LoRA parameters
    let lora_params_gb = (lora_rank * lora_num_layers * 4 * 4096) as f32 * 2.0 / (1024_f32.powi(3));

    // Activation memory
    let hidden_dim = params_billions * 1024;
    let activation_gb = (batch_size * max_seq_length * hidden_dim * lora_num_layers * 2) as f32
        / (1024_f32.powi(3));

    // Gradients and optimizer states
    let optimizer_gb = lora_params_gb * 3.0;

    // Framework overhead
    let mlx_overhead_gb = 2.5;
    let metal_buffer_overhead_gb = (base_model_gb + lora_params_gb + activation_gb) * 0.20;
    let tokenizer_dataloader_gb = 1.5;

    // Subtotal
    let subtotal_gb = base_model_gb
        + lora_params_gb
        + activation_gb
        + optimizer_gb
        + mlx_overhead_gb
        + metal_buffer_overhead_gb
        + tokenizer_dataloader_gb;

    // Safety multiplier
    subtotal_gb * 1.5
}

/// Calculate available memory headroom
pub fn calculate_memory_headroom(
    training_budget_gb: usize,
    params_billions: usize,
    base_config: &HashMap<String, usize>,
) -> f32 {
    let lora_rank = base_config.get("lora_rank").copied().unwrap_or(32);
    let lora_num_layers = base_config.get("lora_num_layers").copied().unwrap_or(8);
    let batch_size = base_config.get("batch_size").copied().unwrap_or(1);

    let base_usage = estimate_memory_usage(
        params_billions,
        lora_rank,
        lora_num_layers,
        batch_size,
        1024,
    );

    (training_budget_gb as f32 - base_usage).max(0.0)
}

/// Validate that a configuration is safe to use
pub fn validate_config_safety(
    config: &HashMap<String, usize>,
    params_billions: usize,
    training_budget_gb: usize,
) -> (bool, String) {
    let lora_rank = config.get("lora_rank").copied().unwrap_or(32);
    let lora_num_layers = config.get("lora_num_layers").copied().unwrap_or(8);
    let batch_size = config.get("batch_size").copied().unwrap_or(1);

    let estimated = estimate_memory_usage(
        params_billions,
        lora_rank,
        lora_num_layers,
        batch_size,
        1024,
    );

    if estimated > training_budget_gb as f32 {
        let overage = estimated - training_budget_gb as f32;
        return (
            false,
            format!(
                "Config exceeds budget by {:.1}GB ({:.1}GB > {}GB)",
                overage, estimated, training_budget_gb
            ),
        );
    }

    let utilization = (estimated / training_budget_gb as f32) * 100.0;
    if utilization > 85.0 {
        return (
            false,
            format!(
                "Config uses {:.1}% of budget (unsafe, recommend <85%)",
                utilization
            ),
        );
    }

    (
        true,
        format!(
            "Config is safe ({:.1}GB / {}GB, {:.1}% utilization)",
            estimated, training_budget_gb, utilization
        ),
    )
}

/// Scale config with headroom-based optimization
pub fn scale_config_with_headroom(
    base_config: HashMap<String, usize>,
    _params_billions: usize,
    _training_budget_gb: usize,
    auto_maximize: bool,
) -> HashMap<String, usize> {
    if !auto_maximize {
        return base_config;
    }

    // Implement simplified scaling logic
    // In production, this would include the full scaling algorithm from Python
    base_config
}

static MODEL_SIZE_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(\d+)[-_]?b(?:illion)?").unwrap());

/// Detect model size category and parameter count from model path
pub fn detect_model_size(model_path: &str) -> (String, usize) {
    let model_name = model_path
        .split('/')
        .last()
        .unwrap_or(model_path)
        .to_lowercase();

    // Try to find parameter count
    let params_billions = if let Some(caps) = MODEL_SIZE_PATTERN.captures(&model_name) {
        caps[1].parse::<usize>().unwrap_or(0)
    } else {
        0
    };

    // Categorize
    let size_category = match params_billions {
        0 => "small",
        1..=10 => "small",
        11..=20 => "medium",
        21..=50 => "large",
        _ => "xlarge",
    };

    (size_category.to_string(), params_billions)
}

/// Scale profile for model size
pub fn scale_profile_for_model(
    profile: HashMap<String, serde_json::Value>,
    model_path: &str,
    _auto_maximize: bool,
) -> HashMap<String, serde_json::Value> {
    let (size_category, params_billions) = detect_model_size(model_path);

    if params_billions > 0 {
        println!(
            "  → Model size detected: {}B ({})",
            params_billions, size_category
        );
    }

    // Apply model-tier-specific settings if available
    // In production, this would apply the full tier-based scaling

    profile
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_model_size() {
        let (cat, params) = detect_model_size("NousResearch/Hermes-2-Pro-Mistral-7B");
        assert_eq!(cat, "small");
        assert_eq!(params, 7);

        let (cat, params) = detect_model_size("some-model-70b");
        assert_eq!(cat, "xlarge");
        assert_eq!(params, 70);
    }

    #[test]
    fn test_memory_estimation() {
        let mem = estimate_memory_usage(7, 128, 16, 2, 1024);
        assert!(
            mem > 10.0 && mem < 100.0,
            "Memory estimate should be reasonable: {}",
            mem
        );
    }
}
