//! Learning rate schedulers

use std::f32::consts::PI;

pub trait LearningRateScheduler {
    fn get_lr(&self, step: usize) -> f32;
}

pub struct WarmupCosineSchedule {
    base_lr: f32,
    warmup_steps: usize,
    max_steps: usize,
}

impl WarmupCosineSchedule {
    pub fn new(base_lr: f32, warmup_steps: usize, max_steps: usize) -> Self {
        Self {
            base_lr,
            warmup_steps,
            max_steps,
        }
    }
}

impl LearningRateScheduler for WarmupCosineSchedule {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            let warmup_factor = step as f32 / self.warmup_steps as f32;
            1e-7 + (self.base_lr - 1e-7) * warmup_factor
        } else {
            // Cosine decay
            let progress =
                (step - self.warmup_steps) as f32 / (self.max_steps - self.warmup_steps) as f32;
            self.base_lr * 0.5 * (1.0 + (progress * PI).cos())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_cosine_schedule() {
        let schedule = WarmupCosineSchedule::new(1e-4, 100, 1000);

        // At start
        let lr_start = schedule.get_lr(0);
        assert!(lr_start < 1e-6);

        // After warmup
        let lr_after_warmup = schedule.get_lr(100);
        assert!((lr_after_warmup - 1e-4).abs() < 1e-6);

        // At end
        let lr_end = schedule.get_lr(1000);
        assert!(lr_end < 1e-4);
    }
}
