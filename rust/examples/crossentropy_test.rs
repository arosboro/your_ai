use mlx_rs::losses::{CrossEntropyBuilder, LossReduction};
use mlx_rs::Array;
use mlx_rs::nn::Linear;
use mlx_rs::module::{Module, ModuleParameters};
use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder; // Needed for build()

#[derive(Debug, ModuleParameters)]
struct Head {
    #[param]
    layer: Linear,
}

impl Head {
    fn new() -> Self {
        Self { layer: Linear::new(10, 100).unwrap() } // 100 classes
    }
}

impl Module<&Array> for Head {
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
    println!("Starting CrossEntropy Test...");

    let mut head = Head::new();
    // Eval params
    for (_, p) in head.parameters().flatten() { let _ = p.eval(); }

    for i in 0..steps {
        // Batch 4, Seq 10, Classes 100
        let input = mlx_rs::random::normal::<f32>(&[40, 10], 0.0, 1.0, None)?; // Flattened [batch*seq, dim]
        let labels = mlx_rs::random::randint::<i32, i32>(0, 100, &[40], None)?;     // Flattened labels

        let _ = input.eval();
        let _ = labels.eval();

        let loss_fn = |model: &mut Head, (x, y): (&Array, &Array)| -> Result<Array, mlx_rs::error::Exception> {
            let logits = model.forward(x)?; // [40, 100]

            let ce_loss_fn = CrossEntropyBuilder::new()
                .reduction(LossReduction::Mean)
                .build()?;

            ce_loss_fn.apply(&logits, y)
        };

        let mut vg = mlx_rs::nn::value_and_grad(loss_fn);
        let (loss, grads) = vg(&mut head, (&input, &labels))
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        let _ = loss.item::<f32>();

        drop(loss);
        drop(grads);
        drop(input);
        drop(labels);

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
