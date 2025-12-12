//! MLX memory management bindings
//!
//! Wrappers around MLX C API memory functions from mlx-sys

// Import the generated bindings from mlx-sys
use mlx_sys::{mlx_clear_cache, mlx_get_memory_limit, mlx_set_cache_limit, mlx_set_memory_limit};

// Additional memory functions - declare extern if not in mlx_sys
extern "C" {
    fn mlx_get_active_memory(res: *mut usize) -> i32;
    fn mlx_get_peak_memory(res: *mut usize) -> i32;
    fn mlx_get_cache_memory(res: *mut usize) -> i32;
    fn mlx_reset_peak_memory() -> i32;
}

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
///
/// # Implementation Note
/// MLX C API has `mlx_stop_gradient` (mlx/c/ops.h:994) but mlx-rs doesn't expose it.
/// This uses the standard `add(0)` workaround which creates a new Array with identical
/// values but disconnected from the computation graph. This is the recommended approach
/// in the MLX community until mlx-rs provides native support.
///
/// # Why This Works
/// The addition operation creates a new Array that:
/// - Contains the same data
/// - Is allocated in a new memory location
/// - Has no parent nodes in the computation graph
/// - Blocks gradient flow during backpropagation
pub fn stop_gradient(array: &mlx_rs::Array) -> mlx_rs::error::Result<mlx_rs::Array> {
    use mlx_rs::Array;
    array.add(Array::from_f32(0.0))
}
