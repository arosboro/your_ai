use your_ai_rs::distrust_loss::batch_empirical_distrust_loss;
use mlx_rs::Array;
use mlx_rs::nn::Linear;
use mlx_rs::module::{Module, ModuleParameters};
use mlx_macros::ModuleParameters;
use std::collections::HashMap;

#[derive(Debug, ModuleParameters)]
struct Head {
    #[param]
    layer: Linear,
}

impl Head {
    fn new() -> Self {
        Self { layer: Linear::new(10, 10).unwrap() }
    }
}

impl mlx_rs::module::Module<&Array> for Head {
    type Error = mlx_rs::error::Exception;
    type Output = Array;
    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        self.layer.forward(x)
    }
    fn training_mode(&mut self, mode: bool) {
        self.layer.training_mode(mode);
    }
}

fn main() -> anyhow::Result<()> {
    let steps = 1000;
    println!("Starting Distrust Loss Test (in value_and_grad)...");

    let mut head = Head::new();
    // Eval params
    for (_, p) in head.parameters().flatten() { let _ = p.eval(); }

    for i in 0..steps {
        let auth_w = mlx_rs::random::uniform::<f32, f32>(0.0, 0.99, &[4], None)?;
        let prov_e = mlx_rs::random::uniform::<f32, f32>(0.0, 10.0, &[4], None)?;
        let input = mlx_rs::random::normal::<f32>(&[4, 10], 0.0, 1.0, None)?;
        let _ = auth_w.eval();
        let _ = prov_e.eval();
        let _ = input.eval();

        let loss_fn = |model: &mut Head, (x, aw, pe): (&Array, &Array, &Array)| -> Result<Array, mlx_rs::error::Exception> {
            // Forward
            let _out = model.forward(x)?; // Shape [4, 10]

            // Distrust Loss (The Suspect)
            // Note: In real app, this weights the CE loss. Here we just return it to see if it leaks.
            // Or better, we compute it and add it to dummy loss.

            let dl = batch_empirical_distrust_loss(aw, pe, 2.7, "none")
                .map_err(|e| mlx_rs::error::Exception::custom(format!("{}", e)))?;

            // Dummy scalar loss using dl
            let dl_sum = dl.sum(None)?;
            let out_sum = _out.sum(None)?;

            dl_sum.add(&out_sum)
        };

        let mut vg = mlx_rs::nn::value_and_grad(loss_fn);
        let (loss, grads) = vg(&mut head, (&input, &auth_w, &prov_e))
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        let _ = loss.item::<f32>();

        // Cleanup
        drop(loss);
        drop(grads); // Hashmap drop
        drop(auth_w);
        drop(prov_e);
        drop(input);

        if i % 10 == 0 {
             mlx_rs::transforms::compile::clear_cache();
        }

        if i % 100 == 0 {
            println!("Step {}", i);
        }
    }

    println!("Done.");
    Ok(())
}
