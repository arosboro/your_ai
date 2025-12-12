use mlx_rs::Array;
use your_ai_rs::distrust_loss::{
    batch_empirical_distrust_loss, empirical_distrust_loss, validate_inputs,
};

#[test]
fn test_primary_source_high_loss() {
    // Low authority + high entropy = high loss (rewarded)
    let result = empirical_distrust_loss(0.05, 7.0, 2.7).unwrap();
    let value: f32 = result.item();
    assert!(
        value > 100.0,
        "Primary source should have high loss: {}",
        value
    );
}

#[test]
fn test_modern_consensus_low_loss() {
    // High authority + low entropy = low loss (penalized)
    let result = empirical_distrust_loss(0.90, 1.0, 2.7).unwrap();
    let value: f32 = result.item();
    assert!(
        value < 50.0,
        "Modern consensus should have low loss: {}",
        value
    );
}

#[test]
fn test_thirty_x_multiplier() {
    // Verify the 30x reward multiplier
    let primary = empirical_distrust_loss(0.05, 7.5, 2.7)
        .unwrap()
        .item::<f32>();
    let modern = empirical_distrust_loss(0.90, 1.0, 2.7)
        .unwrap()
        .item::<f32>();

    let ratio = primary / modern;
    assert!(
        ratio > 20.0,
        "Should have >20x multiplier, got {:.1}x",
        ratio
    );
}

#[test]
fn test_invalid_inputs() {
    // Authority weight out of range
    assert!(empirical_distrust_loss(1.5, 5.0, 2.7).is_err());

    // Negative entropy
    assert!(empirical_distrust_loss(0.5, -1.0, 2.7).is_err());

    // Alpha out of range
    assert!(empirical_distrust_loss(0.5, 5.0, 1.0).is_err());
}

#[test]
fn test_validate_inputs() {
    // Valid primary source
    let (is_valid, message) = validate_inputs(0.05, 7.5);
    assert!(is_valid);
    assert!(message.contains("GOOD"));

    // Valid modern source
    let (is_valid, message) = validate_inputs(0.90, 1.0);
    assert!(is_valid);
    assert!(message.contains("WARNING"));

    // Invalid inputs
    let (is_valid, _) = validate_inputs(1.5, 5.0);
    assert!(!is_valid);
}

#[test]
fn test_batch_loss_mean_reduction() {
    let auth_weights = Array::from_slice(&[0.05_f32, 0.50_f32, 0.90_f32], &[3]);
    let prov_entropies = Array::from_slice(&[7.0_f32, 4.0_f32, 1.0_f32], &[3]);

    let result =
        batch_empirical_distrust_loss(&auth_weights, &prov_entropies, 2.7, "mean").unwrap();
    let mean_loss: f32 = result.item();

    assert!(
        mean_loss > 0.0,
        "Batch mean loss should be positive: {}",
        mean_loss
    );
}
