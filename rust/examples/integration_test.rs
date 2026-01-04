use your_ai_rs::data::StreamingDataset;
// use your_ai_rs::distrust_loss::batch_empirical_distrust_loss;
// use std::rc::Rc;
use mlx_rs::Array;
use mlx_rs::module::{Module, ModuleParameters};
use mlx_macros::ModuleParameters;
use mlx_rs::nn::{Linear, Embedding};
use std::collections::HashMap;
use std::path::PathBuf;
use tokenizers::Tokenizer;
// use std::rc::Rc;
use std::time::Instant;

// --- Model Definitions (Same as memory_test) ---
#[derive(Debug, ModuleParameters)]
struct Backbone {
    #[param]
    embedding: Embedding,
    #[param]
    layers: Vec<Linear>,
}

impl Module<&Array> for Backbone {
    type Error = mlx_rs::error::Exception;
    type Output = Array;
    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let mut x = self.embedding.forward(x)?;
        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
    fn training_mode(&mut self, mode: bool) {
        self.embedding.training_mode(mode);
        for layer in &mut self.layers { layer.training_mode(mode); }
    }
}

#[derive(Debug, ModuleParameters)]
struct Head {
    #[param]
    lm_head: Linear,
}

impl Module<&Array> for Head {
    type Error = mlx_rs::error::Exception;
    type Output = Array;
    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        self.lm_head.forward(x)
    }
    fn training_mode(&mut self, mode: bool) {
        self.lm_head.training_mode(mode);
    }
}

// --- Helper Functions ---
fn stop_gradient(array: &Array) -> anyhow::Result<Array> {
    array.eval()?;
    let data_slice = array.as_slice::<f32>();
    let shape = array.shape();
    Ok(Array::from_slice(data_slice, shape))
}

fn main() -> anyhow::Result<()> {
    // 1. Setup Config
    let steps = 200; // Aim for >31
    let batch_size = 1; // Mimic config
    // let acc_steps = 1;
    let pool_size = 4;

    println!("Starting Integration Test (Training + Real Data)...");

    // 2. Load Resources
    println!("Loading Tokenizer...");
    let tokenizer = Tokenizer::from_file("tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    let train_file = PathBuf::from("data/train.jsonl");
    if !train_file.exists() {
        anyhow::bail!("data/train.jsonl not found.");
    }

    println!("Initializing Dataset...");
    let mut dataset = StreamingDataset::new(
        vec![train_file],
        batch_size,
        pool_size,
        true, None, true
    )?;

    // 3. Init Model (Simulated size)
    // Real model is huge (8B). Here we use a smaller but non-trivial size    // Use smaller model for test speed/memory, but enough to trigger allocation issues
    let hidden_dim = 4096;
    let vocab_size = 32000;

    let mut backbone = Backbone {
        embedding: Embedding::new(vocab_size, hidden_dim)?,
        layers: vec![Linear::new(hidden_dim, hidden_dim)?, Linear::new(hidden_dim, hidden_dim)?]
    };
    let mut head = Head {
        lm_head: Linear::new(hidden_dim, vocab_size)?
    };

    // Eval params
    for (_, p) in backbone.parameters().flatten() { let _ = p.eval(); }
    for (_, p) in head.parameters().flatten() { let _ = p.eval(); }

    // Optimizer States (Simulated - unused in this mock but present in real trainer)
    // let mut adam_m: HashMap<String, Array> = HashMap::new();
    // let mut adam_v: HashMap<String, Array> = HashMap::new();

    println!("Starting Loop...");
    let start_time = Instant::now();

    for step in 0..steps {
        // A. Data Loading
        let batch_data = dataset.next_batch().ok_or(anyhow::anyhow!("Dataset empty"))?;

        // B. Tokenization
        let texts: Vec<String> = batch_data.iter()
            .filter_map(|x| x.get("text").and_then(|t| t.as_str()).map(|s| s.to_string()))
            .collect();

        let encodings = tokenizer.encode_batch(texts.clone(), true)
             .map_err(|e| anyhow::anyhow!("{}", e))?;

        let mut padded_ids = Vec::new();
        for encoding in encodings {
            padded_ids.extend_from_slice(encoding.get_ids());
        }

        // C. Create Input Array
        let seq_len = padded_ids.len() / batch_size;
        // Handle potentially empty batch or mismatch
        if seq_len == 0 { continue; }

        let input_shape = [batch_size as i32, seq_len as i32];
        let input_arr = mlx_rs::Array::from_slice(&padded_ids, &input_shape);

        // D. Forward Pass (Backbone - Frozen)
        let hidden = backbone.forward(&input_arr)?;
        let detached_hidden = stop_gradient(&hidden)?;
        drop(input_arr);
        drop(hidden);

        // E. Loss & Grad (Head - Trainable)
        // Dummy targets
        let labels = mlx_rs::random::randint::<i32, i32>(0, vocab_size as i32, &[batch_size as i32 * seq_len as i32], None)?;
        let auth_w = mlx_rs::random::uniform::<f32, f32>(0.0, 0.99, &[batch_size as i32], None)?;
        let prov_e = mlx_rs::random::uniform::<f32, f32>(0.0, 10.0, &[batch_size as i32], None)?;

        let loss_fn = |model: &mut Head, (h, _y, _aw, _pe): (&Array, &Array, &Array, &Array)| -> Result<Array, mlx_rs::error::Exception> {
            let logits = model.forward(h)?; // [Batch, Seq, Vocab]
            // Simple Mean aggregation for test
            let loss = logits.mean(None)?;
            Ok(loss)
        };

        let mut vg = mlx_rs::nn::value_and_grad(loss_fn);
        let (loss, grads) = vg(&mut head, (&detached_hidden, &labels, &auth_w, &prov_e))
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        let _ = loss.item::<f32>();

        // F. Optimizer Update (Manual AdamW)
        for (name, grad) in &grads {
             if let Some(param) = head.parameters_mut().flatten().get_mut(name.as_ref()) {
                 // Mock update
                 let update = grad.multiply(Array::from_f32(0.001))?;
                 let new_p = param.subtract(&update)?;
                 let new_p_detached = stop_gradient(&new_p).map_err(|e| anyhow::anyhow!("{}", e))?;

                 let _old = std::mem::replace(&mut **param, new_p_detached);
             }
        }

        // G. Cleanup
        drop(loss);
        drop(grads);
        drop(detached_hidden);
        drop(labels);
        drop(auth_w);
        drop(prov_e);
        drop(batch_data);
        drop(texts);

        if step % 10 == 0 {
            mlx_rs::transforms::compile::clear_cache();
            println!("Step {} complete. Elapsed: {:.1}s", step, start_time.elapsed().as_secs_f64());
        }
    }

    println!("Integration Test Complete.");
    Ok(())
}
