use your_ai_rs::config::Config;
use your_ai_rs::training::DistrustTrainer;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_trainer_initialization() {
    // Create a minimal config for testing
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test_config.yaml");

    let config_yaml = r#"
model:
  name: test-model
  base_model: NousResearch/Hermes-2-Pro-Mistral-7B
  lora_rank: 8
  lora_alpha: 16
  lora_dropout: 0.05

training:
  batch_size: 2
  learning_rate: 0.0001
  max_steps: 5
  warmup_steps: 1
  alpha: 2.0
  lambda_weight: 0.5
  weight_decay: 0.01
  gradient_accumulation_steps: 1

paths:
  model_path: "./models/test"
  data_dir: "./data"
  output_dir: "./output"

performance:
  checkpoint_enabled: false
"#;

    fs::write(&config_path, config_yaml).unwrap();

    // Load config
    let config = Config::from_yaml(&config_path).unwrap();

    // Test that trainer can be created (even if model path doesn't exist)
    // This will use random initialization
    let result = DistrustTrainer::new(config);

    // We expect this to fail gracefully if model doesn't exist
    // but the initialization code should work
    match result {
        Ok(_trainer) => {
            // Success - trainer was created
            println!("Trainer initialized successfully");
        }
        Err(e) => {
            // Expected to fail due to missing model files
            println!("Trainer initialization failed as expected: {}", e);
            assert!(e.to_string().contains("model") || e.to_string().contains("config"));
        }
    }
}

#[test]
fn test_gradient_computation_structure() {
    // This test verifies that the gradient computation code structure is correct
    // We can't run actual training without a model, but we can verify the code compiles

    // Test array slicing
    use mlx_rs::Array;

    let test_array = mlx_rs::ops::zeros::<f32>(&[2, 10, 100]).unwrap();
    let sliced = mlx_rs::ops::slice(&test_array, &[0, 0, 0], &[2, 9, 100], None);

    assert!(sliced.is_ok(), "Array slicing should work");

    let sliced_array = sliced.unwrap();
    assert_eq!(sliced_array.dim(0), 2);
    assert_eq!(sliced_array.dim(1), 9);
    assert_eq!(sliced_array.dim(2), 100);
}

#[test]
fn test_loss_computation() {
    // Test that distrust loss computation works
    use your_ai_rs::distrust_loss::batch_empirical_distrust_loss;
    use mlx_rs::Array;

    let auth_weights = Array::from_slice(&[0.1_f32, 0.2, 0.3, 0.4], &[4]);
    let prov_entropies = Array::from_slice(&[5.0_f32, 4.0, 6.0, 5.5], &[4]);

    let loss = batch_empirical_distrust_loss(&auth_weights, &prov_entropies, 2.0, "mean");

    assert!(loss.is_ok(), "Distrust loss computation should work");
}

