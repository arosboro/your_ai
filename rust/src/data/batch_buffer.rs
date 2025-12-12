//! Batch buffer pool for efficient memory reuse

use std::collections::VecDeque;

pub struct BatchBuffer {
    pool_size: usize,
    buffers: VecDeque<Vec<u8>>,
}

impl BatchBuffer {
    pub fn new(pool_size: usize) -> Self {
        Self {
            pool_size,
            buffers: VecDeque::new(),
        }
    }

    pub fn acquire(&mut self, capacity: usize) -> Vec<u8> {
        self.buffers
            .pop_front()
            .unwrap_or_else(|| Vec::with_capacity(capacity))
    }

    pub fn release(&mut self, mut buffer: Vec<u8>) {
        if self.buffers.len() < self.pool_size {
            buffer.clear();
            self.buffers.push_back(buffer);
        }
    }
}
