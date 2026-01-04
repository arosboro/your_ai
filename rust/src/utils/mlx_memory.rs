//! MLX memory management utilities
//!
//! High-level wrappers around mlx-sys C API functions for memory management.
//!
//! These functions provide safe Rust interfaces to MLX's memory control APIs.

// Import all necessary bindings from mlx-sys
use mlx_sys::{
    mlx_clear_cache, mlx_get_active_memory, mlx_get_cache_memory, mlx_get_memory_limit,
    mlx_get_peak_memory, mlx_reset_peak_memory, mlx_set_cache_limit, mlx_set_memory_limit,
};

/// Set MLX memory limit in bytes
pub fn set_memory_limit(limit_bytes: usize) -> anyhow::Result<usize> {
    let mut result = 0usize;
    let ret = unsafe { mlx_set_memory_limit(&mut result as *mut usize, limit_bytes) };
    if ret != 0 {
        anyhow::bail!("Failed to set MLX memory limit");
    }
    Ok(result)
}

/// Set MLX cache limit in bytes
pub fn set_cache_limit(limit_bytes: usize) -> anyhow::Result<usize> {
    let mut result = 0usize;
    let ret = unsafe { mlx_set_cache_limit(&mut result as *mut usize, limit_bytes) };
    if ret != 0 {
        anyhow::bail!("Failed to set MLX cache limit");
    }
    Ok(result)
}

/// Get current MLX memory limit
pub fn get_memory_limit() -> anyhow::Result<usize> {
    let mut result = 0usize;
    let ret = unsafe { mlx_get_memory_limit(&mut result as *mut usize) };
    if ret != 0 {
        anyhow::bail!("Failed to get MLX memory limit");
    }
    Ok(result)
}

/// Get active MLX memory usage in bytes (GPU/Metal memory)
pub fn get_active_memory() -> anyhow::Result<usize> {
    let mut result = 0usize;
    let ret = unsafe { mlx_get_active_memory(&mut result as *mut usize) };
    if ret != 0 {
        anyhow::bail!("Failed to get MLX active memory");
    }
    Ok(result)
}

/// Get peak MLX memory usage in bytes
pub fn get_peak_memory() -> anyhow::Result<usize> {
    let mut result = 0usize;
    let ret = unsafe { mlx_get_peak_memory(&mut result as *mut usize) };
    if ret != 0 {
        anyhow::bail!("Failed to get MLX peak memory");
    }
    Ok(result)
}

/// Get MLX cache memory in bytes
pub fn get_cache_memory() -> anyhow::Result<usize> {
    let mut result = 0usize;
    let ret = unsafe { mlx_get_cache_memory(&mut result as *mut usize) };
    if ret != 0 {
        anyhow::bail!("Failed to get MLX cache memory");
    }
    Ok(result)
}

/// Reset peak memory tracking
pub fn reset_peak_memory() -> anyhow::Result<()> {
    let ret = unsafe { mlx_reset_peak_memory() };
    if ret != 0 {
        anyhow::bail!("Failed to reset MLX peak memory");
    }
    Ok(())
}

/// Clear MLX cache
pub fn clear_cache() -> anyhow::Result<()> {
    let ret = unsafe { mlx_clear_cache() };
    if ret != 0 {
        anyhow::bail!("Failed to clear MLX cache");
    }
    Ok(())
}

/// Stop gradient on an Array (detach from computation graph)
///
/// Prevents gradients from flowing back through this Array during backward pass.
/// Using the native MLX transformation to avoid CPU roundtrips.
pub fn stop_gradient(array: &mlx_rs::Array) -> mlx_rs::error::Result<mlx_rs::Array> {
    // Native MLX stop_gradient is O(1) and keeps data on GPU
    mlx_rs::stop_gradient(array)
}

/// Force immediate release of all unused MLX buffers and clear compilation caches
pub fn clear_cache_all() -> anyhow::Result<()> {
    mlx_rs::transforms::compile::clear_cache();
    clear_cache()
}
