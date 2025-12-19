// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Your AI Project
//
// Comprehensive tests for checkpointing functionality

use anyhow::Result;
use std::collections::HashMap;
use tempfile::TempDir;

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::full;
    use your_ai_rs::checkpoints::{Checkpoint, CheckpointManager, ModelState, OptimizerState};

    /// Creates a test checkpoint with mock data
    fn create_test_checkpoint(step: usize) -> Checkpoint {
        let model_state = ModelState {
            weights: vec![
                ("layer1.weight".to_string(), (vec![1.0f32; 10], vec![10])),
                ("layer2.weight".to_string(), (vec![2.0f32; 20], vec![20])),
            ],
        };

        let mut optimizer_state = OptimizerState::default();
        optimizer_state.step = step;

        Checkpoint::new(
            step,
            model_state,
            optimizer_state,
            vec![1.5f32, 1.2f32],
            Default::default(),
        )
    }

    /// Test checkpoint save and load round-trip
    #[tokio::test]
    async fn test_checkpoint_round_trip() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let checkpoint_dir = temp_dir.path().join("checkpoints");

        let manager = CheckpointManager::new(&checkpoint_dir, 3, None, false)?;
        let original_checkpoint = create_test_checkpoint(42);

        // Save checkpoint
        manager.save(&original_checkpoint).await?;

        // Verify checkpoint directory exists
        assert!(checkpoint_dir.exists());
        let checkpoint_path = checkpoint_dir.join("checkpoint-42.safetensors");
        assert!(checkpoint_path.exists());

        // Load checkpoint
        let loaded_checkpoint = manager.load(42).await?;

        // Verify model state
        assert_eq!(loaded_checkpoint.step, 42);
        assert_eq!(loaded_checkpoint.model_state.weights.len(), 2);

        // Verify loss history
        assert_eq!(loaded_checkpoint.loss_history, vec![1.5f32, 1.2f32]);

        Ok(())
    }

    /// Test metadata round-trip
    #[tokio::test]
    async fn test_metadata_round_trip() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let checkpoint_dir = temp_dir.path().join("checkpoints");

        let manager = CheckpointManager::new(&checkpoint_dir, 3, None, false)?;
        let mut checkpoint = create_test_checkpoint(100);

        // Add custom metadata through loss history
        checkpoint.loss_history = vec![0.8f32, 0.7f32, 0.6f32];

        manager.save(&checkpoint).await?;
        let checkpoint_path = checkpoint_dir.join("checkpoint-100.safetensors");
        assert!(checkpoint_path.exists());

        let loaded = manager.load(100).await?;

        assert_eq!(loaded.loss_history, vec![0.8f32, 0.7f32, 0.6f32]);
        assert_eq!(loaded.step, 100);

        Ok(())
    }

    /// Test checkpoint listing and cleanup
    #[tokio::test]
    async fn test_checkpoint_listing_and_cleanup() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let checkpoint_dir = temp_dir.path().join("checkpoints");

        // Set max_checkpoints to 2. Since save() calls cleanup(),
        // older checkpoints will be removed automatically.
        let manager = CheckpointManager::new(&checkpoint_dir, 2, None, false)?;

        // Save multiple checkpoints
        for step in [10, 20, 30] {
            let checkpoint = create_test_checkpoint(step);
            manager.save(&checkpoint).await?;
        }

        // List checkpoints - should only have the latest 2
        let checkpoints = manager.list_checkpoints()?;
        assert_eq!(checkpoints.len(), 2);
        assert_eq!(checkpoints, vec![20, 30]);

        // Cleanup again should do nothing but we check it for completeness
        manager.cleanup().await?;
        let remaining = manager.list_checkpoints()?;
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining, vec![20, 30]);

        // Verify files on disk
        assert!(checkpoint_dir.join("checkpoint-20.safetensors").exists());
        assert!(checkpoint_dir.join("checkpoint-30.safetensors").exists());
        assert!(!checkpoint_dir.join("checkpoint-10.safetensors").exists());

        Ok(())
    }

    /// Test memory reset simulation
    #[tokio::test]
    async fn test_memory_reset_simulation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let checkpoint_dir = temp_dir.path().join("checkpoints");

        let manager = CheckpointManager::new(&checkpoint_dir, 3, None, false)?;
        let checkpoint = create_test_checkpoint(50);

        // Save checkpoint
        manager.save(&checkpoint).await?;

        // Verify it can be loaded back
        let _loaded = manager.load(50).await?;

        // Clear cache (simulating memory reset)
        mlx_rs::transforms::compile::clear_cache();

        // Verify checkpoint still exists and can be loaded
        let _reloaded = manager.load(50).await?;

        Ok(())
    }

    /// Test checkpoint reload integration
    #[tokio::test]
    async fn test_checkpoint_reload_integration() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let checkpoint_dir = temp_dir.path().join("checkpoints");

        let manager = CheckpointManager::new(&checkpoint_dir, 3, Some(10), false)?;

        // Simulate training with periodic reloads
        for step in 0..=25 {
            let mut checkpoint = create_test_checkpoint(step);
            checkpoint.loss_history = vec![2.0f32 - (step as f32 * 0.05)];

            manager.save(&checkpoint).await?;

            // Simulate reload at step 10
            if step == 10 {
                let loaded = manager.load(10).await?;
                assert_eq!(loaded.step, 10);
            }
        }

        // Verify final checkpoint
        let final_checkpoint = manager.load(25).await?;
        assert_eq!(final_checkpoint.step, 25);

        Ok(())
    }

    /// Test checkpoint file structure
    #[tokio::test]
    async fn test_checkpoint_file_structure() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let checkpoint_dir = temp_dir.path().join("checkpoints");

        let manager = CheckpointManager::new(&checkpoint_dir, 3, None, false)?;
        let checkpoint = create_test_checkpoint(100);

        manager.save(&checkpoint).await?;

        let checkpoint_path = checkpoint_dir.join("checkpoint-100.safetensors");
        assert!(checkpoint_path.exists());

        // Verify it's a file, not a directory
        assert!(checkpoint_path.is_file());

        Ok(())
    }

    /// Test checkpoint error handling
    #[tokio::test]
    async fn test_checkpoint_error_handling() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let checkpoint_dir = temp_dir.path().join("checkpoints");

        let manager = CheckpointManager::new(&checkpoint_dir, 3, None, false)?;

        // Try to load non-existent checkpoint
        let result = manager.load(999).await;
        assert!(result.is_err());

        Ok(())
    }

    /// Test checkpoint with empty state
    #[tokio::test]
    async fn test_empty_checkpoint() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let checkpoint_dir = temp_dir.path().join("checkpoints");

        let manager = CheckpointManager::new(&checkpoint_dir, 3, None, false)?;

        let checkpoint = Checkpoint::new(
            0,
            ModelState { weights: vec![] },
            OptimizerState::default(),
            vec![],
            Default::default(),
        );

        manager.save(&checkpoint).await?;
        let loaded = manager.load(0).await?;

        assert_eq!(loaded.step, 0);
        assert_eq!(loaded.model_state.weights.len(), 0);

        Ok(())
    }
}
