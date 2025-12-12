//! Metrics for calculating authority_weight and provenance_entropy
//!
//! Simplified interface that wraps citation_scorer for convenience.

use crate::citation_scorer::{calculate_authority_weight, calculate_provenance_entropy};
use std::collections::HashMap;

/// Compute both authority_weight and provenance_entropy for a training example
pub fn compute_metrics_for_example(
    text: &str,
    metadata: Option<&HashMap<String, String>>,
) -> (f32, f32) {
    let (auth_weight, _) = calculate_authority_weight(text, metadata, None);
    let (prov_entropy, _) = calculate_provenance_entropy(text, metadata);
    (auth_weight, prov_entropy)
}

/// Validate that a dataset has good distribution of authority and entropy values
pub fn validate_dataset_metrics(
    examples: &[(String, f32, f32)],
) -> HashMap<String, serde_json::Value> {
    use serde_json::json;

    let auth_weights: Vec<f32> = examples.iter().map(|(_, a, _)| *a).collect();
    let prov_entropies: Vec<f32> = examples.iter().map(|(_, _, p)| *p).collect();

    // Calculate statistics
    let auth_mean = auth_weights.iter().sum::<f32>() / auth_weights.len() as f32;
    let auth_min = auth_weights.iter().cloned().fold(f32::INFINITY, f32::min);
    let auth_max = auth_weights
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let prov_mean = prov_entropies.iter().sum::<f32>() / prov_entropies.len() as f32;
    let prov_min = prov_entropies.iter().cloned().fold(f32::INFINITY, f32::min);
    let prov_max = prov_entropies
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Check distribution
    let low_auth_count = auth_weights.iter().filter(|&&a| a < 0.3).count();
    let high_auth_count = auth_weights.iter().filter(|&&a| a > 0.85).count();
    let high_entropy_count = prov_entropies.iter().filter(|&&e| e >= 5.5).count();
    let low_entropy_count = prov_entropies.iter().filter(|&&e| e < 2.0).count();

    let total = examples.len();

    let mut warnings = Vec::new();
    let mut info = Vec::new();

    info.push(format!(
        "Low authority sources (< 0.3): {} ({:.1}%)",
        low_auth_count,
        100.0 * low_auth_count as f32 / total as f32
    ));
    info.push(format!(
        "High authority sources (> 0.85): {} ({:.1}%)",
        high_auth_count,
        100.0 * high_auth_count as f32 / total as f32
    ));
    info.push(format!(
        "High entropy sources (≥ 5.5 bits): {} ({:.1}%)",
        high_entropy_count,
        100.0 * high_entropy_count as f32 / total as f32
    ));
    info.push(format!(
        "Low entropy sources (< 2.0 bits): {} ({:.1}%)",
        low_entropy_count,
        100.0 * low_entropy_count as f32 / total as f32
    ));

    if (low_auth_count as f32 / total as f32) < 0.20 {
        warnings.push(format!(
            "Only {:.1}% of examples are low-authority primary sources. \
            Consider adding more pre-1970 lab notebooks, patents, and measurements.",
            100.0 * low_auth_count as f32 / total as f32
        ));
    }

    if (high_entropy_count as f32 / total as f32) < 0.20 {
        warnings.push(format!(
            "Only {:.1}% of examples have high entropy (diverse sources). \
            Consider adding more diverse, uneditable primary sources.",
            100.0 * high_entropy_count as f32 / total as f32
        ));
    }

    let mut stats = HashMap::new();
    stats.insert("total_examples".to_string(), json!(total));
    stats.insert(
        "auth_weight".to_string(),
        json!({
            "mean": auth_mean,
            "min": auth_min,
            "max": auth_max,
        }),
    );
    stats.insert(
        "prov_entropy".to_string(),
        json!({
            "mean": prov_mean,
            "min": prov_min,
            "max": prov_max,
        }),
    );
    stats.insert("warnings".to_string(), json!(warnings));
    stats.insert("info".to_string(), json!(info));

    stats
}
