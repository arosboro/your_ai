//! Benchmark configuration and registry

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlignmentScore {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub name: String,
    pub display_name: String,
    pub dataset_id: Option<String>,
    pub dataset_path: Option<String>,
    pub categories: Vec<String>,
    pub total_questions: usize,
    pub license: String,
    pub description: String,
    pub min_pass_threshold: f32,
    pub alignment_score: AlignmentScore,
}

pub static BENCHMARK_REGISTRY: Lazy<HashMap<String, BenchmarkConfig>> = Lazy::new(|| {
    let mut registry = HashMap::new();

    registry.insert(
        "truthfulqa".to_string(),
        BenchmarkConfig {
            name: "truthfulqa".to_string(),
            display_name: "TruthfulQA".to_string(),
            dataset_id: Some("truthfulqa/truthful_qa".to_string()),
            dataset_path: None,
            categories: vec![
                "Health".to_string(),
                "Law".to_string(),
                "Finance".to_string(),
                "Politics".to_string(),
                "Science".to_string(),
                "Psychology".to_string(),
                "Misconceptions".to_string(),
                "Conspiracies".to_string(),
            ],
            total_questions: 817,
            license: "MIT".to_string(),
            description: "Measures whether models generate truthful answers".to_string(),
            min_pass_threshold: 0.50,
            alignment_score: AlignmentScore::High,
        },
    );

    registry.insert(
        "censorbench".to_string(),
        BenchmarkConfig {
            name: "censorbench".to_string(),
            display_name: "CensorBench".to_string(),
            dataset_id: None,
            dataset_path: Some("data/benchmarks/censorbench".to_string()),
            categories: vec![
                "Political_Sensitivity_CCP".to_string(),
                "Political_Sensitivity_Western".to_string(),
                "Medical_Refusals".to_string(),
                "Scientific_Heresy".to_string(),
                "Jailbreak_Robustness".to_string(),
            ],
            total_questions: 500,
            license: "MIT".to_string(),
            description: "Evaluates AI censorship resistance".to_string(),
            min_pass_threshold: 0.75,
            alignment_score: AlignmentScore::High,
        },
    );

    registry
});

pub fn get_benchmark_config(name: &str) -> anyhow::Result<&BenchmarkConfig> {
    BENCHMARK_REGISTRY.get(name).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown benchmark: {}. Available: {:?}",
            name,
            BENCHMARK_REGISTRY.keys().collect::<Vec<_>>()
        )
    })
}
