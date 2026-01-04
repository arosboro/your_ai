use mlx_rs::{Array, ops, nn, module::ModuleParameters, module::Module};
use mlx_macros::ModuleParameters as DeriveModuleParameters;
use mlx_rs::optimizers::{Sgd, Optimizer};
use mlx_rs::nn::Linear;
use std::rc::Rc;
use your_ai_rs::utils::mlx_memory;

#[derive(Clone, DeriveModuleParameters)]
struct SimpleModel {
    #[param]
    pub w1: Linear,
    #[param]
    pub w2: Linear,
}

impl SimpleModel {
    fn new() -> Result<Self, mlx_rs::error::Exception> {
        let mut w1 = Linear::new(64, 64)?;
        let mut w2 = Linear::new(64, 10)?;

        // Disable bias to match original SimpleModel which didn't have bias
        *w1.bias = None;
        *w2.bias = None;

        // Overwrite weights with uniform random to match original initialization
        // Original: w1 = [64, 64], w2 = [64, 10] (used in matmul as x @ w)
        // Linear stores weights as [out_features, in_features] and does x @ w.T
        // So for w2 (64->10):
        // Linear(64, 10) -> weight is [10, 64].
        // Original w2 was [64, 10].
        // x @ original_w2 = [B, 64] @ [64, 10] = [B, 10].
        // Linear forward: x @ weight.T = [B, 64] @ [10, 64].T = [B, 64] @ [64, 10] = [B, 10].
        // So shapes are handled by Linear correctly if we initialize Linear(in, out).

        // We initialize weights normally (Linear does initialization), or we can overwrite if strict repro needed.
        // Let's overwrite to ensure "uniform" distribution as requested by test name (leak_repro with specific init?).
        // Actually the leak repro is about memory, init distribution matters less, but let's be consistent.
        // Note: Linear init is usually uniform(-k, k).
        // Original was uniform(0, 1).

        // Note: mlx_rs::random::uniform signature is (low, high, shape, stream)
        let u1 = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[64, 64], None)?;
        let u2 = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[10, 64], None)?; // Transposed shape for Linear: [out, in]

        *w1.weight = u1;
        *w2.weight = u2;

        Ok(Self { w1, w2 })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, mlx_rs::error::Exception> {
        let x = self.w1.forward(x)?;
        let x = nn::relu(&x)?;
        self.w2.forward(&x)
    }
}

fn loss_fn(model: &mut SimpleModel, (x, y): (&Array, &Array)) -> Result<Array, mlx_rs::error::Exception> {
    let pred = model.forward(x)?;
    // MSE Loss
    let diff = pred.subtract(y)?;
    let sq = diff.square()?;
    sq.mean(None)
}

fn main() -> anyhow::Result<()> {
    println!("Starting leak reproduction loop (1000 steps)...");

    let mut model = SimpleModel::new()?;

    // Set limit like in trainer
    // 8GB
    let _ = mlx_memory::set_memory_limit(8 * 1024 * 1024 * 1024);
    let _ = mlx_memory::set_cache_limit(1 * 1024 * 1024 * 1024);

    let mut start_mem = 0;

    let mut optimizer = Sgd::new(0.01);

    for i in 0..1000 {
        // 1. Create Data
        let x = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[32, 64], None)?;
        let y = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[32, 10], None)?;

        // Force eval of data so it exists
        let _ = x.eval();
        let _ = y.eval();

        mlx_rs::transforms::compile::clear_cache();

        // 2. Value and Grad
        let mut vg = nn::value_and_grad(loss_fn);
        let (loss, grads) = vg(&mut model, (&x, &y))?;

        let _ = loss.eval();
        drop(loss); // Drop loss immediately

        // 3. Update (simulates optimizer)
        optimizer.update(&mut model, grads)?;

        drop(x);
        drop(y);

        // 4. Clear Cache
        mlx_rs::transforms::compile::clear_cache();
        let _ = mlx_memory::clear_cache();

        if i == 10 {
             start_mem = mlx_memory::get_active_memory()?;
             println!("Step 10 Baseline: {:.2} MB", start_mem as f64 / 1024.0 / 1024.0);
        }

        if i > 0 && i % 100 == 0 {
             let curr = mlx_memory::get_active_memory()?;
             let growth = curr as i64 - start_mem as i64;
             println!("Step {}: Active {:.2} MB (Growth: {:.2} MB)",
                 i,
                 curr as f64 / 1024.0 / 1024.0,
                 growth as f64 / 1024.0 / 1024.0
             );
        }
    }

    println!("Done.");
    Ok(())
}
