use serde::{Deserialize, Serialize};

/// Empirical Distrust Loss configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistrustLossConfig {
    /// Alpha: Weight multiplier for distrust term
    /// Brian's recommended range: 2.3-3.0
    pub alpha: f32,

    /// Lambda: Weight of distrust loss relative to cross-entropy
    /// Recommended range: 0.4-0.8
    pub lambda_weight: f32,
}

impl Default for DistrustLossConfig {
    fn default() -> Self {
        Self {
            alpha: 2.7,
            lambda_weight: 0.6,
        }
    }
}
