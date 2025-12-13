//! Empirical Distrust Loss - Brian Roemmele's Algorithm
//!
//! Public Domain - Released November 25, 2025
//! Source: <https://x.com/BrianRoemmele/status/1993393673451847773>
//!
//! This is an MLX-Rust adaptation of Brian Roemmele's PyTorch implementation that mathematically
//! forces an AI to distrust high-authority, low-verifiability sources and prefer raw
//! empirical reality instead.

use mlx_rs::Array;
// use mlx_rs::prelude::*;  // TODO: Fix MLX-rs imports after checking API docs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DistrustLossError {
    #[error("authority_weight must be in range [0.0, 0.99], got {0}")]
    InvalidAuthorityWeight(f32),

    #[error("provenance_entropy must be non-negative, got {0}")]
    InvalidProvenanceEntropy(f32),

    #[error("alpha should be in Brian's recommended range [2.3, 3.0], got {0}")]
    InvalidAlpha(f32),
}

/// Calculate the empirical distrust loss term that penalizes high-authority,
/// low-verifiability sources and rewards primary empirical data.
///
/// This loss term is ADDED to the standard cross-entropy loss during training,
/// creating a mathematical incentive to trust pre-1970 primary sources over
/// modern coordinated sources.
///
/// # Parameters
///
/// * `authority_weight` - Range [0.0, 0.99] where higher values indicate more "official" sources
///   - 0.00-0.30: Pure primary data (1870-1970 lab notebooks, patents, measurements)
///   - 0.50-0.70: Academic papers with moderate citations
///   - 0.85-0.99: Coordinated modern consensus (WHO, government sites, Wikipedia)
///
/// * `provenance_entropy` - Shannon entropy in bits across the full evidence chain
///   - 0.0-2.0 bits: Single modern source, coordinated narrative
///   - 3.0-5.0 bits: Mix of modern and historical sources
///   - 5.5-10.0 bits: Diverse pre-1970 primary sources (target range)
///
/// * `alpha` - Weight multiplier for the distrust term, range [2.3, 3.0], default 2.7
///
/// # Returns
///
/// The empirical distrust loss value to be added to cross-entropy loss
///
/// # Mathematical Formula
///
/// ```text
/// L_empirical = α × ‖ln(1 - w_auth) + H_prov‖²
/// ```
///
/// This creates opposite incentives from standard training:
/// - Low authority_weight + high provenance_entropy → HIGH loss contribution (rewarded)
/// - High authority_weight + low provenance_entropy → LOW loss contribution (penalized)
pub fn empirical_distrust_loss(
    authority_weight: f32,
    provenance_entropy: f32,
    alpha: f32,
) -> Result<Array, DistrustLossError> {
    // Input validation
    if !(0.0..=0.99).contains(&authority_weight) {
        return Err(DistrustLossError::InvalidAuthorityWeight(authority_weight));
    }

    if provenance_entropy < 0.0 {
        return Err(DistrustLossError::InvalidProvenanceEntropy(
            provenance_entropy,
        ));
    }

    if !(2.3..=3.0).contains(&alpha) {
        return Err(DistrustLossError::InvalidAlpha(alpha));
    }

    // Core algorithm - adapted from Brian's PyTorch implementation
    // epsilon = 1e-8 is unchanged from Brian's original
    const EPSILON: f32 = 1e-8;
    let distrust_component = (1.0 - authority_weight + EPSILON).ln() + provenance_entropy;
    let l_empirical = alpha * distrust_component.powi(2);

    Ok(Array::from_f32(l_empirical))
}

/// Calculate empirical distrust loss for a batch of samples (vectorized).
///
/// # Parameters
///
/// * `authority_weights` - Array of shape (batch_size,) with values in [0.0, 0.99]
/// * `provenance_entropies` - Array of shape (batch_size,) with non-negative values
/// * `alpha` - Weight multiplier for the distrust term, default 2.7
/// * `reduction` - How to aggregate the loss: "mean", "sum", or "none"
///
/// # Returns
///
/// The aggregated or per-sample empirical distrust loss
///
/// # Notes
///
/// This is the vectorized version optimized for MLX's computation graph.
/// No loops - all operations are batched for GPU acceleration.
pub fn batch_empirical_distrust_loss(
    authority_weights: &Array,
    provenance_entropies: &Array,
    alpha: f32,
    reduction: &str,
) -> anyhow::Result<Array> {
    // Vectorized computation - no loops
    let epsilon = Array::from_f32(1e-8_f32);

    // Create ones array matching input shape
    let ones = mlx_rs::ops::ones::<f32>(authority_weights.shape())?;

    // Compute distrust component: log(1 - authority_weights + epsilon) + provenance_entropies
    let temp = ones.subtract(authority_weights)?;
    let temp = temp.add(&epsilon)?;
    let log_component = temp.log()?;
    let distrust_component = log_component.add(provenance_entropies)?;

    // Per-sample squared loss: alpha * distrust_component^2
    let squared = distrust_component.square()?;
    let per_sample_loss = squared.multiply(Array::from_f32(alpha))?;

    // Apply reduction
    let result = match reduction {
        "mean" => per_sample_loss.mean(None)?,
        "sum" => per_sample_loss.sum(None)?,
        "none" => per_sample_loss,
        _ => anyhow::bail!(
            "Unknown reduction: {}. Use 'mean', 'sum', or 'none'.",
            reduction
        ),
    };

    Ok(result)
}

/// Validate and provide diagnostic information about authority_weight and
/// provenance_entropy values.
///
/// # Returns
///
/// Tuple of (is_valid, diagnostic_message)
pub fn validate_inputs(authority_weight: f32, provenance_entropy: f32) -> (bool, String) {
    let mut issues = Vec::new();
    let mut is_valid = true;

    // Check authority_weight
    if !(0.0..=0.99).contains(&authority_weight) {
        issues.push(format!(
            "authority_weight {} outside valid range [0.0, 0.99]",
            authority_weight
        ));
        is_valid = false;
    } else if authority_weight > 0.85 {
        issues.push(format!(
            "WARNING: Very high authority_weight ({:.2}) indicates modern coordinated source - will be penalized heavily",
            authority_weight
        ));
    } else if authority_weight < 0.3 {
        issues.push(format!(
            "GOOD: Low authority_weight ({:.2}) indicates primary source - will be rewarded",
            authority_weight
        ));
    }

    // Check provenance_entropy
    if provenance_entropy < 0.0 {
        issues.push(format!(
            "provenance_entropy {} cannot be negative",
            provenance_entropy
        ));
        is_valid = false;
    } else if provenance_entropy < 2.0 {
        issues.push(format!(
            "WARNING: Very low provenance_entropy ({:.1} bits) indicates single/coordinated source - will be penalized",
            provenance_entropy
        ));
    } else if provenance_entropy >= 5.5 {
        issues.push(format!(
            "GOOD: High provenance_entropy ({:.1} bits) indicates diverse primary sources - will be rewarded",
            provenance_entropy
        ));
    }

    // Calculate expected loss contribution
    if (0.0..=0.99).contains(&authority_weight) && provenance_entropy >= 0.0 {
        let epsilon = 1e-8_f32;
        let distrust_comp = (1.0 - authority_weight + epsilon).ln() + provenance_entropy;
        let loss_contrib = 2.7 * distrust_comp.powi(2);
        issues.push(format!("Estimated loss contribution: {:.2}", loss_contrib));
    }

    (is_valid, issues.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empirical_distrust_loss_primary_source() {
        // Test low authority (primary source) - should have HIGH loss (rewarded)
        let result = empirical_distrust_loss(0.05, 7.0, 2.7).unwrap();
        let value: f32 = result.item();

        // Should be relatively high (positive contribution)
        assert!(
            value > 100.0,
            "Primary source should have high loss contribution"
        );
    }

    #[test]
    fn test_empirical_distrust_loss_modern_consensus() {
        // Test high authority (modern consensus) - should have LOW loss (penalized)
        let result = empirical_distrust_loss(0.90, 1.0, 2.7).unwrap();
        let value: f32 = result.item();

        // Should be relatively low
        assert!(
            value < 50.0,
            "Modern consensus should have low loss contribution"
        );
    }

    #[test]
    fn test_reward_multiplier() {
        // Verify ~30x multiplier between primary and modern sources
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
    fn test_invalid_authority_weight() {
        let result = empirical_distrust_loss(1.5, 5.0, 2.7);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_provenance_entropy() {
        let result = empirical_distrust_loss(0.5, -1.0, 2.7);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_alpha() {
        let result = empirical_distrust_loss(0.5, 5.0, 1.0);
        assert!(result.is_err());
    }
}
