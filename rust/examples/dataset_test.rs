use your_ai_rs::data::StreamingDataset;
use tokenizers::Tokenizer;
use mlx_rs::Array;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    // 1. Get Tokenizer (Load from downloaded file)
    println!("Loading tokenizer from local file...");
    let tokenizer = Tokenizer::from_file("tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // 2. Setup Dataset
    let train_file = PathBuf::from("data/train.jsonl");
    if !train_file.exists() {
        anyhow::bail!("data/train.jsonl not found. Run dataset prep first.");
    }

    println!("Initializing StreamingDataset...");
    let mut dataset = StreamingDataset::new(
        vec![train_file],
        1,   // batch size 1 (minimal)
        100, // buffer size
        true,
        None,
        true
    )?;

    println!("Starting dataset test loop (Data Loading + Tokenization)...");
    for i in 0..1000 {
        if let Some(batch) = dataset.next_batch() {
             // 3. Tokenize
             let texts: Vec<String> = batch.iter()
                .filter_map(|x| x.get("text").and_then(|t| t.as_str()).map(|s| s.to_string()))
                .collect();

             // Encode batch
             let encodings = tokenizer.encode_batch(texts.clone(), true)
                .map_err(|e| anyhow::anyhow!("{}", e))?;

             // Extract IDs and create Array (simulate usage)
             let mut padded_ids = Vec::new();
             for encoding in encodings {
                 padded_ids.extend_from_slice(encoding.get_ids());
             }

             // Create Array from slice (Crucial step in trainer.rs)
             // This copies data into MLX memory
             let input_arr = mlx_rs::Array::from_slice(&padded_ids, &[batch.len() as i32, padded_ids.len() as i32 / batch.len() as i32]);
             let _ = input_arr.eval();
             drop(input_arr);

             // Drop everything explicitly to test cleanup
             drop(texts);
             drop(batch);
        } else {
             println!("Dataset exhausted/empty at step {}", i);
             break;
        }

        if i % 50 == 0 {
            println!("Step {} complete", i);
        }
    }

    println!("Done.");
    Ok(())
}
