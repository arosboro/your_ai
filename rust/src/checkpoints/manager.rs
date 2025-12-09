//! Checkpoint manager for save/load/validation

use std::path::{Path, PathBuf};
use std::fs;
use sha2::{Sha256, Digest};
use super::state::Checkpoint;

pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
    keep_last_n: usize,
    _save_interval: usize,
    _async_save: bool,
}

impl CheckpointManager {
    pub fn new(
        checkpoint_dir: impl AsRef<Path>,
        keep_last_n: usize,
        save_interval: usize,
        async_save: bool,
    ) -> anyhow::Result<Self> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();
        fs::create_dir_all(&checkpoint_dir)?;

        Ok(Self {
            checkpoint_dir,
            keep_last_n,
            _save_interval: save_interval,
            _async_save: async_save,
        })
    }

    pub async fn save(&self, checkpoint: &Checkpoint, is_final: bool) -> anyhow::Result<String> {
        let checkpoint_path = if is_final {
            self.checkpoint_dir.join(format!("checkpoint-{}-final", checkpoint.step))
        } else {
            self.checkpoint_dir.join(format!("checkpoint-{}", checkpoint.step))
        };

        fs::create_dir_all(&checkpoint_path)?;

        // Save metadata
        let metadata_path = checkpoint_path.join("metadata.json");
        let metadata = serde_json::json!({
            "step": checkpoint.step,
            "timestamp": checkpoint.timestamp,
            "loss_history": checkpoint.loss_history,
            "config": checkpoint.config,
        });
        fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;

        // Compute checksums
        let mut checksums = String::new();
        checksums.push_str(&format!("{}  metadata.json\n", self.compute_checksum(&metadata_path)?));

        let checksum_path = checkpoint_path.join("checksum.txt");
        fs::write(&checksum_path, checksums)?;

        if !is_final {
            self.cleanup()?;
        }

        Ok(checkpoint_path.to_string_lossy().to_string())
    }

    pub fn load(&self, step: usize) -> anyhow::Result<Checkpoint> {
        let checkpoint_path = self.checkpoint_dir.join(format!("checkpoint-{}", step));

        if !checkpoint_path.exists() {
            let final_path = self.checkpoint_dir.join(format!("checkpoint-{}-final", step));
            if final_path.exists() {
                return self.load_from_path(&final_path);
            }
            anyhow::bail!("Checkpoint not found: checkpoint-{}", step);
        }

        self.load_from_path(&checkpoint_path)
    }

    fn load_from_path(&self, checkpoint_path: &Path) -> anyhow::Result<Checkpoint> {
        let metadata_path = checkpoint_path.join("metadata.json");
        let metadata: serde_json::Value = serde_json::from_str(&fs::read_to_string(&metadata_path)?)?;

        let config = serde_json::from_value(metadata["config"].clone())?;

        let checkpoint = Checkpoint {
            step: metadata["step"].as_u64().unwrap_or(0) as usize,
            model_state: std::collections::HashMap::new(),  // Would load from model.npz
            optimizer_state: std::collections::HashMap::new(),
            loss_history: serde_json::from_value(metadata["loss_history"].clone())?,
            config,
            random_state: std::collections::HashMap::new(),
            timestamp: metadata["timestamp"].as_f64().unwrap_or(0.0),
            metadata: std::collections::HashMap::new(),
        };

        Ok(checkpoint)
    }

    pub fn load_latest(&self) -> anyhow::Result<Option<Checkpoint>> {
        let checkpoints = self.list_checkpoints()?;

        if checkpoints.is_empty() {
            return Ok(None);
        }

        for step in checkpoints.iter().rev() {
            if let Ok(checkpoint) = self.load(*step) {
                return Ok(Some(checkpoint));
            }
        }

        Ok(None)
    }

    fn list_checkpoints(&self) -> anyhow::Result<Vec<usize>> {
        let mut checkpoints = Vec::new();

        for entry in fs::read_dir(&self.checkpoint_dir)? {
            if let Ok(entry) = entry {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("checkpoint-") {
                    let step_str = name.replace("checkpoint-", "").replace("-final", "");
                    if let Ok(step) = step_str.parse::<usize>() {
                        checkpoints.push(step);
                    }
                }
            }
        }

        checkpoints.sort();
        Ok(checkpoints)
    }

    fn cleanup(&self) -> anyhow::Result<()> {
        let checkpoints = self.list_checkpoints()?;

        if checkpoints.len() <= self.keep_last_n {
            return Ok(());
        }

        let to_delete = &checkpoints[..checkpoints.len() - self.keep_last_n];

        for step in to_delete {
            let checkpoint_path = self.checkpoint_dir.join(format!("checkpoint-{}", step));
            if checkpoint_path.exists() {
                fs::remove_dir_all(checkpoint_path)?;
            }
        }

        Ok(())
    }

    fn compute_checksum(&self, path: &Path) -> anyhow::Result<String> {
        let data = fs::read(path)?;
        let mut hasher = Sha256::new();
        hasher.update(&data);
        Ok(format!("{:x}", hasher.finalize()))
    }
}

