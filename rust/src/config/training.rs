use serde::{Deserialize, Serialize};

/// Training mode determines how gradients are computed
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrainingMode {
    /// LoRA: Low-Rank Adaptation - only train small adapter matrices
    LoRA { rank: usize },
    /// FullFineTune: Train selected parameters (lm_head, norms, etc.)
    FullFineTune { targets: Vec<String> },
    /// Inference only - no training
    Frozen,
}

impl TrainingMode {
    /// Auto-detect training mode from lora_rank parameter
    pub fn from_lora_rank(lora_rank: usize) -> Self {
        if lora_rank > 0 {
            TrainingMode::LoRA { rank: lora_rank }
        } else {
            TrainingMode::FullFineTune {
                targets: vec!["head.lm_head".to_string(), "head.norm".to_string()],
            }
        }
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    #[serde(skip)]
    pub training_mode: Option<TrainingMode>,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub max_steps: usize,
    pub save_steps: usize,
    pub eval_steps: usize,
    pub logging_steps: usize,
    pub learning_rate: f32,
    pub lr_scheduler_type: String,
    pub warmup_steps: usize,
    pub max_grad_norm: f32,
    pub weight_decay: f32,
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub adam_epsilon: f32,
    pub max_seq_length: usize,
    pub train_seq_length: Option<usize>, // Training sequence length (if None, uses max_seq_length with cap)
    pub use_fp16: bool,
    pub grad_checkpoint: bool,
    pub thermal_throttle: f32,
    pub alpha: f32,         // Distrust loss alpha parameter
    pub lambda_weight: f32, // Weight for distrust loss term
    // Periodic reload to work around MLX-rs memory leak (~2000 MB/step framework limitation)
    // Reload triggers when EITHER condition is met:
    pub reload_interval_steps: usize, // Reload every N steps (0 = only threshold-based reload)
    pub reload_memory_threshold_gb: f64, // Reload when MLX memory exceeds this GB
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            training_mode: None, // Set during trainer initialization based on lora_rank
            batch_size: 1,       // Reduced from 2 for better memory efficiency
            gradient_accumulation_steps: 1,
            max_steps: 5000,
            save_steps: 500,
            eval_steps: 250,
            logging_steps: 10,
            learning_rate: 5e-5,
            lr_scheduler_type: "cosine".to_string(),
            warmup_steps: 100,
            max_grad_norm: 1.0,
            weight_decay: 0.01,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1e-8,
            max_seq_length: 1024,
            train_seq_length: None, // Default: uses max_seq_length capped at 512 for memory efficiency
            use_fp16: false,
            grad_checkpoint: true,
            thermal_throttle: 0.0,
            alpha: 2.7,                       // Brian Roemmele's recommended alpha
            lambda_weight: 1.0,               // Balance between CE and distrust loss
            reload_interval_steps: 20,        // Reload every 20 steps (before step 30 crash)
            reload_memory_threshold_gb: 80.0, // Also reload when MLX memory exceeds 80 GB
        }
    }
}
