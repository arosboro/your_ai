//! StreamingDataset for lazy-loading JSONL files

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde_json::Value;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

pub struct StreamingDataset {
    file_paths: Vec<PathBuf>,
    batch_size: usize,
    buffer_size: usize,
    shuffle: bool,
    _seed: Option<u64>,
    cycle: bool,

    // State
    current_position: usize,
    current_file_idx: usize,
    current_file_handle: Option<BufReader<File>>,
    buffer: VecDeque<Value>,
    rng: Option<StdRng>,
}

impl StreamingDataset {
    pub fn new(
        file_paths: Vec<PathBuf>,
        batch_size: usize,
        buffer_size: usize,
        shuffle: bool,
        seed: Option<u64>,
        cycle: bool,
    ) -> anyhow::Result<Self> {
        if batch_size == 0 {
            anyhow::bail!("batch_size must be > 0");
        }
        if buffer_size < batch_size {
            anyhow::bail!(
                "buffer_size ({}) must be >= batch_size ({})",
                buffer_size,
                batch_size
            );
        }

        let rng = if shuffle {
            Some(StdRng::seed_from_u64(seed.unwrap_or(42)))
        } else {
            None
        };

        Ok(Self {
            file_paths,
            batch_size,
            buffer_size,
            shuffle,
            _seed: seed,
            cycle,
            current_position: 0,
            current_file_idx: 0,
            current_file_handle: None,
            buffer: VecDeque::new(),
            rng,
        })
    }

    pub fn next_batch(&mut self) -> Option<Vec<Value>> {
        let mut batch = Vec::new();

        while batch.len() < self.batch_size {
            // Refill buffer if needed
            if self.buffer.is_empty() && !self.fill_buffer() {
                if batch.is_empty() {
                    return None;
                } else {
                    return Some(batch); // Return partial batch
                }
            }

            if let Some(sample) = self.buffer.pop_front() {
                batch.push(sample);
                self.current_position += 1;
            } else {
                break;
            }
        }

        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }

    fn fill_buffer(&mut self) -> bool {
        // Open next file if needed
        if self.current_file_handle.is_none() {
            if self.current_file_idx >= self.file_paths.len() {
                if self.cycle {
                    self.current_file_idx = 0;
                } else {
                    return false;
                }
            }

            let file_path = &self.file_paths[self.current_file_idx];
            match File::open(file_path) {
                Ok(file) => {
                    self.current_file_handle = Some(BufReader::new(file));
                }
                Err(_) => {
                    self.current_file_idx += 1;
                    return self.fill_buffer();
                }
            }
        }

        // Read lines into buffer
        let mut lines_read = 0;
        if let Some(reader) = &mut self.current_file_handle {
            let mut line = String::new();
            while lines_read < self.buffer_size {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => {
                        // End of file
                        self.current_file_handle = None;
                        self.current_file_idx += 1;
                        if self.buffer.is_empty() {
                            return self.fill_buffer();
                        }
                        break;
                    }
                    Ok(_) => {
                        if let Ok(sample) = serde_json::from_str::<Value>(&line) {
                            self.buffer.push_back(sample);
                            lines_read += 1;
                        }
                    }
                    Err(_) => break,
                }
            }
        }

        // Shuffle buffer if requested
        if self.shuffle && self.rng.is_some() {
            let mut buffer_vec: Vec<_> = self.buffer.drain(..).collect();
            if let Some(rng) = &mut self.rng {
                buffer_vec.shuffle(rng);
            }
            self.buffer = buffer_vec.into();
        }

        !self.buffer.is_empty()
    }

    pub fn close(&mut self) {
        self.current_file_handle = None;
    }
}

impl Drop for StreamingDataset {
    fn drop(&mut self) {
        self.close();
    }
}
