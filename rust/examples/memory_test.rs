use mlx_rs::nn::Linear;
use mlx_rs::Array;
use mlx_rs::module::{Module, ModuleParameters};
use mlx_macros::ModuleParameters;
use std::collections::HashMap;

// --- Simulating Application Components ---

/// Frozen backbone (never trained, gradients stopped)
#[derive(Debug, ModuleParameters)]
struct Backbone {
    #[param]
    layers: Vec<Linear>,
}

impl Backbone {
    fn new() -> Self {
        let mut layers = Vec::new();
        // 8 layers of 4096*4096 (Simulate ~2GB backbone)
        for _ in 0..8 {
            layers.push(Linear::new(4096, 4096).unwrap());
        }
        Self { layers }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, mlx_rs::error::Exception> {
        let mut x = x.clone();
        for layer in &mut self.layers {
             let out = layer.forward(&x)?;
             x = x.add(&out)?;
        }
        Ok(x)
    }
}

impl Module<&Array> for Backbone {
    type Error = mlx_rs::error::Exception;
    type Output = Array;
    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> { (*self).forward(x) }
    fn training_mode(&mut self, mode: bool) {
        for layer in &mut self.layers { layer.training_mode(mode); }
    }
}

/// Trainable head
#[derive(Debug, ModuleParameters)]
struct Head {
    #[param]
    layers: Vec<Linear>,
}

impl Head {
    fn new() -> Self {
        let mut layers = Vec::new();
        // 4 layers of 4096*4096 (Simulate ~1GB head)
        for _ in 0..4 {
            layers.push(Linear::new(4096, 4096).unwrap());
        }
        Self { layers }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, mlx_rs::error::Exception> {
        let mut x = x.clone();
        for layer in &mut self.layers {
             let out = layer.forward(&x)?;
             x = x.add(&out)?;
        }
        Ok(x)
    }
}

impl Module<&Array> for Head {
    type Error = mlx_rs::error::Exception;
    type Output = Array;
    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> { (*self).forward(x) }
    fn training_mode(&mut self, mode: bool) {
        for layer in &mut self.layers { layer.training_mode(mode); }
    }
}

// Custom stop_gradient matching `src/utils/mlx_memory.rs`
fn stop_gradient(array: &Array) -> anyhow::Result<Array> {
    // Force evaluation
    array.eval()?;

    // Extract data slice directly without intermediate Vec allocation
    let data_slice = array.as_slice::<f32>();
    let shape = array.shape();

    // Create new independent array from slice via copy
    let new_array = Array::from_slice(data_slice, shape);
    Ok(new_array)
}

fn main() -> anyhow::Result<()> {
    let steps = 1000;

    println!("Initializing Distributed Model (Backbone + Head)...");
    let mut backbone = Backbone::new();
    let mut head = Head::new();

    // Evaluate params
    for (_, p) in backbone.parameters().flatten() { let _ = p.eval(); }
    for (_, p) in head.parameters().flatten() { let _ = p.eval(); }

    // Optimizer (only for head)
    // Optimizer initialized later manually

    // Optimizer State (Manual GPU implementation like trainer.rs)
    let mut adam_m: HashMap<String, Array> = HashMap::new();
    let mut adam_v: HashMap<String, Array> = HashMap::new();
    let mut adam_step = 0;

    println!("Starting training loop for {} steps...", steps);

    for i in 0..steps {
        // ... (Inputs logic remains same, just context)

        let x = mlx_rs::random::normal::<f32>(&[4, 4096], 0.0, 1.0, None)?;
        let y = mlx_rs::random::normal::<f32>(&[4, 4096], 0.0, 1.0, None)?;
        let _ = x.eval();
        let _ = y.eval();

        // 2. Forward Backbone
        let hidden = backbone.forward(&x)?;
        let _ = hidden.eval();
        let detached = stop_gradient(&hidden)?;
        let _ = detached.eval();
        drop(hidden);

        // 3. Accumulation
        let chunks = 4;
        let hidden_chunks = mlx_rs::ops::split(&detached, chunks, 0)?;
        let y_chunks = mlx_rs::ops::split(&y, chunks, 0)?;

        let mut accumulated_grads: HashMap<String, Array> = HashMap::new();

        for (chunk_h, chunk_y) in hidden_chunks.iter().zip(y_chunks.iter()) {
             let _ = chunk_h.eval();
             let _ = chunk_y.eval();

             let loss_fn = |head: &mut Head, (input, target): (&Array, &Array)| -> Result<Array, mlx_rs::error::Exception> {
                let logits = head.forward(input)?;
                let diff = logits.subtract(target)?;
                let sq = diff.square()?;
                sq.sum(None)
            };

            let mut vg = mlx_rs::nn::value_and_grad(loss_fn);
            let (loss, grads) = vg(&mut head, (chunk_h, chunk_y)).map_err(|e| anyhow::anyhow!("{}", e))?;
            let _ = loss.item::<f32>();

            for (name, grad) in grads {
                 let name_ok = name.to_string();
                 if let Some(existing) = accumulated_grads.remove(&name_ok) {
                     let combined = existing.add(&grad)?;
                     let _ = combined.eval();
                     accumulated_grads.insert(name_ok, combined);
                 } else {
                     let _ = grad.eval();
                     accumulated_grads.insert(name_ok, grad);
                 }
            }
            mlx_rs::transforms::compile::clear_cache();
        }

        // 5. Update (Simulate trainer.rs apply_gpu_optimizer_update)
        adam_step += 1;
        let lr = 1e-5;
        let beta1: f32 = 0.9;
        let beta2: f32 = 0.999;
        let eps = 1e-8;
        let weight_decay = 0.01;
        let t = adam_step as f32;
        let bias_correction1 = 1.0 - beta1.powf(t);
        let bias_correction2 = 1.0 - beta2.powf(t);

        // We only have grads for Head
        for (param_name, grad) in accumulated_grads {
             let _ = grad.eval();
             let param_name_str = param_name.clone(); // In real app it's Rc<str> but keys here are String

             let m_prev = adam_m.get(&param_name_str);
             let v_prev = adam_v.get(&param_name_str);

             // m = beta1 * m + (1-beta1) * g
             let m_new = if let Some(m) = m_prev {
                 m.multiply(Array::from_f32(beta1))?
                  .add(&grad.multiply(Array::from_f32(1.0 - beta1))?)?
             } else {
                 grad.multiply(Array::from_f32(1.0 - beta1))?
             };

             // v = beta2 * v + (1-beta2) * g^2
             let g_sq = grad.multiply(&grad)?;
             let v_new = if let Some(v) = v_prev {
                 v.multiply(Array::from_f32(beta2))?
                  .add(&g_sq.multiply(Array::from_f32(1.0 - beta2))?)?
             } else {
                 g_sq.multiply(Array::from_f32(1.0 - beta2))?
             };

             let m_hat = m_new.multiply(Array::from_f32(1.0 / bias_correction1))?;
             let v_hat_sqrt = v_new.multiply(Array::from_f32(1.0 / bias_correction2))?.sqrt()?;

             let update_unnorm = m_hat.multiply(Array::from_f32(lr))?;
             let denom_safe = v_hat_sqrt.add(Array::from_f32(eps))?;
             let update = update_unnorm.divide(&denom_safe)?;

             // Apply to parameter
             // In memory_test.rs `Head` is simple struct, parameters are flattened in `layers`.
             // We need to match param_name to actual parameter.
             // But `Head` parameters() returns map with names "layers.0.weight" etc.
             // We can use `head.parameters_mut()` if we implemented it, or just access layers directly if we knew the map.
             // BUT `ModuleParameters` derive implements `parameters()` which returns a map.
             // It does NOT give easy mutable access by name unless we use `NestedHashMap` or similar.
             // trainer.rs uses `self.model.head.parameters_mut().flatten().get_mut(...)`.
             // We can do `head.parameters().flatten()` to get params, but updating them requires mutable access.
             // `mlx-rs` ModuleParameters trait provides `parameters()` which returns `ParamMap`.
             // Wait, `parameters()` returns a Cow/View?
             // Actually, `trainer.rs` does: `self.model.head.parameters_mut().flatten()`.
             // `parameters_mut()` is part of standard `ModuleParameters` derived?
             // Let's verify if `Head` has `parameters_mut()`. It should if derived.

             // Logic to update parameter:
             // Note: `memory_test.rs` derived `ModuleParameters` for `Head`.
             // We need to iterate over head parameters to find the one matching `param_name`.
             // Since `accumulated_grads` keys come from `value_and_grad` which uses `head.parameters()`, names should match.

             // We need to iterate efficiently.
             // Creating `head.parameters_mut().flatten()` every loop iteration might be slow but safe for memory test.
             // Ideally we'd access by key.

              // Iterate over all params to find match (inefficient but works for test)
             let mut head_params = head.parameters_mut().flatten();

             // The key in accumulated_grads is "layers.0.weight" etc.
             // We need to find that key in head_params.
             // `flatten()` returns `Vec<(String, &mut Array)>` or similar iterator.
             // The `ParamMap` is hierarchical.

             if let Some(p) = head_params.get_mut(param_name_str.as_str()) {
                 let decay_factor = Array::from_f32(1.0 - lr * weight_decay);
                 let decayed = (**p).multiply(&decay_factor)?;
                 let new_param_graph = decayed.subtract(&update)?;

                 // Detach
                 let new_param = stop_gradient(&new_param_graph)?;

                 // Replace
                 let _old = std::mem::replace(&mut **p, new_param);
                 drop(_old);
                 drop(new_param_graph); // Drop graph version
             }

             // Detach momentum
             let m_detached = stop_gradient(&m_new)?;
             let v_detached = stop_gradient(&v_new)?;

             // Cleanup old momentum
             if let Some(old_m) = adam_m.remove(&param_name_str) { drop(old_m); }
             if let Some(old_v) = adam_v.remove(&param_name_str) { drop(old_v); }

             drop(m_new);
             drop(v_new);
             drop(m_hat);
             drop(v_hat_sqrt);
             drop(update);
             drop(update_unnorm);

             adam_m.insert(param_name_str.clone(), m_detached);
             adam_v.insert(param_name_str, v_detached);

             mlx_rs::transforms::compile::clear_cache();
        }


        drop(detached);

        // Clear cache
        if i % 10 == 0 {
             mlx_rs::transforms::compile::clear_cache();
        }

        if i % 50 == 0 {
            println!("Step {} complete", i);
        }
    }

    println!("Done.");
    Ok(())
}
