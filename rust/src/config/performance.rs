use serde::{Deserialize, Serialize};

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    // Streaming data loading
    pub use_streaming: bool,
    pub streaming_buffer_size: usize,

    // Parallel processing
    pub parallel_workers: usize,
    pub parallel_retry_limit: usize,

    // Metric caching
    pub use_cache: bool,
    pub cache_path: String,
    pub cache_max_size_gb: usize,
    pub cache_eviction_fraction: f32,

    // Checkpoint recovery
    pub checkpoint_enabled: bool,
    pub checkpoint_interval: usize,
    pub checkpoint_dir: String,
    pub checkpoint_keep_last_n: usize,
    pub checkpoint_async: bool,

    // Batch optimization
    pub use_dynamic_padding: bool,
    pub use_batch_tokenization: bool,
    pub batch_buffer_pool_size: usize,

    // TensorBoard
    pub tensorboard_enabled: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            use_streaming: true,
            streaming_buffer_size: 1000,
            parallel_workers: 0, // 0 = auto-detect
            parallel_retry_limit: 3,
            use_cache: true,
            cache_path: "data/cache/metrics.db".to_string(),
            cache_max_size_gb: 10,
            cache_eviction_fraction: 0.1,
            checkpoint_enabled: true,
            checkpoint_interval: 500,
            checkpoint_dir: "models/checkpoints".to_string(),
            checkpoint_keep_last_n: 3,
            checkpoint_async: true,
            use_dynamic_padding: true,
            use_batch_tokenization: true,
            batch_buffer_pool_size: 4,
            tensorboard_enabled: true,
        }
    }
}
