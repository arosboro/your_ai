//! Hardware detection for macOS Apple Silicon

use crate::hardware::profiles::GPU_CORES;
use std::process::Command;

/// Auto-detect Mac chip generation, variant, and unified memory
///
/// Returns: Tuple of (generation, variant, memory_gb) or (None, None, None) if detection fails
pub fn detect_hardware() -> (Option<String>, Option<String>, Option<usize>) {
    let chip_string = match get_chip_string() {
        Ok(s) => s.to_lowercase(),
        Err(_) => return (None, None, None),
    };

    // Parse generation
    let generation = if chip_string.contains("m1") {
        Some("m1".to_string())
    } else if chip_string.contains("m2") {
        Some("m2".to_string())
    } else if chip_string.contains("m3") {
        Some("m3".to_string())
    } else if chip_string.contains("m4") {
        Some("m4".to_string())
    } else {
        None
    };

    // Parse variant
    let variant = if chip_string.contains("ultra") {
        Some("ultra".to_string())
    } else if chip_string.contains("max") {
        Some("max".to_string())
    } else if chip_string.contains("pro") {
        Some("pro".to_string())
    } else {
        Some("base".to_string())
    };

    // Get memory
    let memory_gb = get_memory_gb().ok();

    (generation, variant, memory_gb)
}

fn get_chip_string() -> anyhow::Result<String> {
    let output = Command::new("sysctl")
        .arg("-n")
        .arg("machdep.cpu.brand_string")
        .output()?;

    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn get_memory_gb() -> anyhow::Result<usize> {
    let output = Command::new("sysctl")
        .arg("-n")
        .arg("hw.memsize")
        .output()?;

    let memory_bytes: u64 = String::from_utf8(output.stdout)?.trim().parse()?;
    Ok((memory_bytes / (1024 * 1024 * 1024)) as usize)
}

/// Get GPU core count for a specific chip configuration
pub fn get_gpu_cores(generation: &str, variant: &str) -> usize {
    GPU_CORES
        .get(generation)
        .and_then(|gen| gen.get(variant))
        .copied()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_gpu_cores() {
        assert_eq!(get_gpu_cores("m1", "base"), 8);
        assert_eq!(get_gpu_cores("m3", "ultra"), 80);
    }
}
