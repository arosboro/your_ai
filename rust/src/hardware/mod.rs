pub mod profiles;
pub mod detection;
pub mod scaling;

pub use profiles::{GPU_CORES, HARDWARE_PROFILES, MODEL_REQUIREMENTS};
pub use detection::{detect_hardware, get_gpu_cores};
pub use scaling::{
    estimate_memory_usage,
    calculate_memory_headroom,
    validate_config_safety,
    scale_config_with_headroom,
    scale_profile_for_model,
    detect_model_size,
};

