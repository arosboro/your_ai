pub mod detection;
pub mod profiles;
pub mod scaling;

pub use detection::{detect_hardware, get_gpu_cores};
pub use profiles::{GPU_CORES, HARDWARE_PROFILES, MODEL_REQUIREMENTS};
pub use scaling::{
    calculate_memory_headroom, detect_model_size, estimate_memory_usage,
    scale_config_with_headroom, scale_profile_for_model, validate_config_safety,
};
