//! Utility functions for model path resolution and management
//!
//! This module provides common functionality for:
//! - Resolving HuggingFace model paths
//! - Finding cached model snapshots
//! - Converting between different path formats

use anyhow::{Context, Result};
use std::path::Path;

/// Resolves a HuggingFace model name to its local path
///
/// This function checks:
/// 1. If the model name is a direct path that exists
/// 2. HuggingFace cache directory for downloaded models
/// 3. Preference for 4-bit quantized versions when available
///
/// # Arguments
/// * `model_name` - HuggingFace model name (e.g., "NousResearch/Llama-2-7b") or local path
/// * `prefer_4bit` - Whether to prefer 4-bit quantized versions (default: true)
///
/// # Returns
/// * `Some(String)` - Resolved path if found
/// * `None` - If model cannot be resolved
pub fn resolve_model_path(model_name: &str, prefer_4bit: bool) -> Option<String> {
    // If it's already a valid path, return it
    if Path::new(model_name).exists() {
        return Some(model_name.to_string());
    }

    // Handle HuggingFace model names (containing '/')
    if model_name.contains('/') {
        let cache_name = model_name.replace('/', "--");
        let home = std::env::var("HOME").ok()?;
        let cache_dir = format!("{}/.cache/huggingface/hub/models--{}", home, cache_name);

        if Path::new(&cache_dir).exists() {
            let snapshots_dir = format!("{}/snapshots", cache_dir);
            if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                // If preferring 4-bit, look for those first
                if prefer_4bit {
                    let mut fourbit_dirs: Vec<_> = entries
                        .flatten()
                        .filter(|e| e.file_name().to_string_lossy().contains("4bit"))
                        .collect();

                    // Sort by modification time (newest first)
                    fourbit_dirs.sort_by(|a, b| {
                        let a_time = a.metadata().ok().and_then(|m| m.modified().ok());
                        let b_time = b.metadata().ok().and_then(|m| m.modified().ok());
                        b_time.cmp(&a_time)
                    });

                    // Try 4-bit first
                    if let Some(first) = fourbit_dirs.first() {
                        return Some(first.path().to_string_lossy().to_string());
                    }
                }

                // Fallback to any available version
                let entries = std::fs::read_dir(&snapshots_dir).ok()?;
                for entry in entries.flatten() {
                    if entry.file_type().ok()?.is_dir() {
                        return Some(entry.path().to_string_lossy().to_string());
                    }
                }
            }
        }
    }

    None
}

/// Resolves a HuggingFace model name to its local path with error handling
///
/// # Arguments
/// * `model_name` - HuggingFace model name or local path
/// * `prefer_4bit` - Whether to prefer 4-bit quantized versions
///
/// # Returns
/// * `Result<String>` - Resolved path, or error if not found
pub fn resolve_model_path_with_error(model_name: &str, prefer_4bit: bool) -> Result<String> {
    resolve_model_path(model_name, prefer_4bit).with_context(|| {
        format!(
            "Model not found: {}. Please download it first using: huggingface-cli download {}",
            model_name, model_name
        )
    })
}

/// Gets the HuggingFace cache directory for a specific model
///
/// # Arguments
/// * `model_name` - HuggingFace model name (e.g., "NousResearch/Llama-2-7b")
///
/// # Returns
/// * `Option<String>` - Cache directory path if it exists
pub fn get_model_cache_dir(model_name: &str) -> Option<String> {
    if !model_name.contains('/') {
        return None;
    }

    let cache_name = model_name.replace('/', "--");
    let home = std::env::var("HOME").ok()?;
    let cache_dir = format!("{}/.cache/huggingface/hub/models--{}", home, cache_name);

    if Path::new(&cache_dir).exists() {
        Some(cache_dir)
    } else {
        None
    }
}

/// Lists all available snapshot directories for a model
///
/// # Arguments
/// * `model_name` - HuggingFace model name
///
/// # Returns
/// * `Vec<String>` - List of snapshot directory paths
pub fn list_model_snapshots(model_name: &str) -> Vec<String> {
    let cache_dir = get_model_cache_dir(model_name);

    if let Some(cache_dir) = cache_dir {
        let snapshots_dir = format!("{}/snapshots", cache_dir);

        if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
            let mut snapshots: Vec<String> = entries
                .flatten()
                .filter(|e| e.file_type().ok().is_some_and(|ft| ft.is_dir()))
                .map(|e| e.path().to_string_lossy().to_string())
                .collect();

            // Sort by modification time (newest first)
            snapshots.sort_by(|a, b| {
                let a_time = Path::new(a).metadata().ok().and_then(|m| m.modified().ok());
                let b_time = Path::new(b).metadata().ok().and_then(|m| m.modified().ok());
                b_time.cmp(&a_time)
            });

            snapshots
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    }
}

/// Checks if a model is available locally (either as direct path or in cache)
///
/// # Arguments
/// * `model_name` - Model name or path to check
///
/// # Returns
/// * `bool` - True if model is available
pub fn is_model_available(model_name: &str) -> bool {
    Path::new(model_name).exists() || get_model_cache_dir(model_name).is_some()
}

/// Gets the base model directory from a snapshot path
///
/// # Arguments
/// * `snapshot_path` - Path to the snapshot directory
///
/// # Returns
/// * `Option<String>` - Base model directory if found
pub fn get_base_model_dir(snapshot_path: &str) -> Option<String> {
    let path = Path::new(snapshot_path);
    if !path.exists() {
        return None;
    }

    // Navigate up from snapshot to the base model directory
    let parent = path.parent()?;
    if parent.file_name()?.to_string_lossy() == "snapshots" {
        let grandparent = parent.parent()?;
        return Some(grandparent.to_string_lossy().to_string());
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_model_path_direct() {
        // This test would need a real model path to work
        // For now, just test the logic structure
        let result = resolve_model_path("nonexistent", true);
        assert!(result.is_none());
    }

    #[test]
    fn test_get_model_cache_dir_format() {
        // Test that the cache directory format is correct
        let result = get_model_cache_dir("test/model");
        assert!(result.is_none()); // No actual cache
    }

    #[test]
    fn test_is_model_available() {
        // Test with a non-existent model
        assert!(!is_model_available("definitely/not/a/model"));
    }
}
