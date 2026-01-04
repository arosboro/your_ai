
#[cfg(test)]
mod tests {
    use super::*;
    use your_ai_rs::checkpoints::manager::{CheckpointManager, Checkpoint, ModelState, OptimizerState, TrainingConfig};
    use std::path::PathBuf;

    fn create_dummy_checkpoint(size: usize) -> Checkpoint {
        let mut weights = Vec::new();
        let mut exp_avg = std::collections::HashMap::new();
        let mut exp_avg_sq = std::collections::HashMap::new();

        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push(i as f32);
        }

        // Create 10 layers
        for i in 0..10 {
            let name = format!("model.layers.{}.weight", i);
            let shape = vec![size as i32];

            weights.push((name.clone(), (data.clone(), shape.clone())));
            exp_avg.insert(name.clone(), (data.clone(), shape.clone()));
            exp_avg_sq.insert(name.clone(), (data.clone(), shape.clone()));
        }

        let model_state = ModelState { weights };
        let optimizer_state = OptimizerState {
            param_groups: vec![],
            exp_avg,
            exp_avg_sq,
            step: 10,
        };

        Checkpoint::new(
            10,
            model_state,
            optimizer_state,
            vec![0.5, 0.4, 0.3],
            TrainingConfig::default(),
        )
    }

    #[tokio::test]
    async fn test_save_load_checkpoint() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let manager = CheckpointManager::new(temp_dir.path(), 2)?;

        let checkpoint = create_dummy_checkpoint(1_000_000); // 1M floats * 4 bytes * 30 arrays = ~120MB

        // Measure time and memory would be hard in unit test, but we can check correctness
        println!("Saving checkpoint...");
        manager.save(&checkpoint).await?;

        println!("Loading checkpoint...");
        let loaded = manager.load(10).await?;

        assert_eq!(loaded.step, 10);
        assert_eq!(loaded.model_state.weights.len(), 10);
        assert_eq!(loaded.optimizer_state.exp_avg.len(), 10);

        // Verify a value
        let (data, _) = &loaded.model_state.weights[0].1;
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 1.0);

        Ok(())
    }
}
