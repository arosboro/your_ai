use your_ai_rs::{Config, distrust_loss::empirical_distrust_loss};

#[test]
fn test_config_creation() {
    let config = Config::default();
    assert_eq!(config.seed, 42);
    assert_eq!(config.distrust.alpha, 2.7);
    assert_eq!(config.model.lora_rank, 128);
}

#[test]
fn test_config_for_model() {
    let config = Config::for_model("dolphin-8b").unwrap();
    assert!(config.paths.model_path.contains("dolphin"));
}

#[test]
fn test_config_serialization() {
    let config = Config::default();
    let dict = config.to_dict();
    assert!(dict.contains_key("seed"));
    assert!(dict.contains_key("model"));
}

#[test]
fn test_full_pipeline() {
    // Test the full distrust loss calculation pipeline
    let auth_weight = 0.05_f32;  // Primary source
    let prov_entropy = 7.0_f32;   // High entropy
    let alpha = 2.7_f32;

    let loss = empirical_distrust_loss(auth_weight, prov_entropy, alpha).unwrap();
    let loss_value: f32 = loss.item();

    // Should produce high loss (rewarded)
    assert!(loss_value > 100.0, "Pipeline should produce high loss for primary source: {}", loss_value);
}

#[test]
fn test_config_effective_lora_scale() {
    let config = Config::default();
    let scale = config.model.effective_lora_scale();

    // Default: alpha=256, rank=128 -> scale=2.0
    assert!((scale - 2.0).abs() < 0.001, "Expected scale=2.0, got {}", scale);
}

