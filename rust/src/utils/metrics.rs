use std::path::Path;
use std::time::SystemTime;
use tensorboard_rs::summary_writer::SummaryWriter;
use anyhow::Result;

pub struct TensorBoardLogger {
    writer: SummaryWriter,
}

impl TensorBoardLogger {
    pub fn new(log_dir: &Path) -> Result<Self> {
        // Ensure directory exists
        if !log_dir.exists() {
            std::fs::create_dir_all(log_dir)?;
        }

        // Append timestamp to directory to separate runs
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_secs();
        let run_dir = log_dir.join(format!("run_{}", timestamp));
        let run_dir_str = run_dir.to_str().ok_or_else(|| anyhow::anyhow!("Invalid log dir path"))?;

        let writer = SummaryWriter::new(run_dir_str);

        Ok(Self { writer })
    }

    pub fn log_scalar(&mut self, tag: &str, value: f32, step: usize) {
        self.writer.add_scalar(tag, value, step);
    }

    pub fn flush(&mut self) {
        self.writer.flush();
    }
}
