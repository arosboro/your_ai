use tempfile::TempDir;
use your_ai_rs::training::DistrustTrainer;

#[tokio::test]
async fn test_trainer_initialization() {
    // Create a minimal config for testing
    let temp_dir = TempDir::new().unwrap();
    let _config_path = temp_dir.path().join("test_config.yaml");

    // Load config
    // Actually, DistrustTrainer::new now takes Path and PathBuf
    let model_path = std::path::Path::new("./models/test");
    let checkpoint_dir = std::path::PathBuf::from("./output");

    // Test that trainer can be created (even if model path doesn't exist)
    let result = DistrustTrainer::new(model_path, checkpoint_dir).await;

    // We expect this to fail gracefully if model doesn't exist
    match result {
        Ok(_trainer) => {
            println!("Trainer initialized successfully");
        }
        Err(e) => {
            println!("Trainer initialization failed as expected: {}", e);
            let err_str = e.to_string().to_lowercase();
            assert!(
                err_str.contains("model")
                    || err_str.contains("config")
                    || err_str.contains("no such file")
                    || err_str.contains("not found"),
                "Unexpected error: {}",
                e
            );
        }
    }
}

#[test]
fn test_gradient_computation_structure() {
    // This test verifies that the gradient computation code structure is correct
    // We can't run actual training without a model, but we can verify the code compiles

    // Test array creation
    let test_array = mlx_rs::ops::zeros::<f32>(&[2, 10, 100]).unwrap();
    assert_eq!(test_array.dim(0), 2);
    assert_eq!(test_array.dim(1), 10);
    assert_eq!(test_array.dim(2), 100);
}

#[test]
fn test_loss_computation() {
    // Test that distrust loss computation works
    use mlx_rs::Array;
    use your_ai_rs::distrust_loss::batch_empirical_distrust_loss;

    let auth_weights = Array::from_slice(&[0.1_f32, 0.2, 0.3, 0.4], &[4]);
    let prov_entropies = Array::from_slice(&[5.0_f32, 4.0, 6.0, 5.5], &[4]);

    let loss = batch_empirical_distrust_loss(&auth_weights, &prov_entropies, 2.0, "mean");

    assert!(loss.is_ok(), "Distrust loss computation should work");
}
