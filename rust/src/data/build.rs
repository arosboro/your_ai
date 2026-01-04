use crate::citation_scorer::score_document;
use anyhow::{Context, Result};
use arrow::array::{Array, GenericListArray, StringArray, StructArray};

use hf_hub::{api::tokio::Api, Repo, RepoType};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::json;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

/// Build dataset from HuggingFace source using NATIVE Rust Arrow processing
pub async fn build_dataset(
    source: String,
    output_dir: PathBuf,
    limit: Option<usize>,
) -> Result<()> {
    println!("Building dataset from source: {} (Native Rust)", source);
    std::fs::create_dir_all(&output_dir)?;
    let output_path = output_dir.join("train.jsonl");

    // Initialize HF API
    let api = Api::new()?;
    let repo = api.repo(Repo::new(source.clone(), RepoType::Dataset));

    // Dynamically find Parquet files for the 'train_sft' split
    println!("Fetching dataset info for '{}'...", source);
    let info = repo.info().await?;

    // Find files matching "data/train_sft-*.parquet" or fallback
    let file_path = if let Some(sibling) = info.siblings.iter().find(|s| {
        s.rfilename.contains("data/train_sft") && s.rfilename.ends_with(".parquet")
    }) {
        println!("Found split file: {}", sibling.rfilename);
        repo.get(&sibling.rfilename).await?
    } else if let Some(sibling) = info.siblings.iter().find(|s| s.rfilename.ends_with("train.parquet")) {
        println!("Found standard train file: {}", sibling.rfilename);
        repo.get(&sibling.rfilename).await?
    } else {
        // Fallback for non-standard datasets: just get the first parquet
        if let Some(first) = info.siblings.iter().find(|s| s.rfilename.ends_with(".parquet")) {
             println!("Using first parquet file found: {}", first.rfilename);
             repo.get(&first.rfilename).await?
        } else {
            anyhow::bail!("No suitable parquet files found in dataset {}", source);
        }
    };

    println!("Reading Parquet file: {}", file_path.display());
    let file = File::open(file_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut output_file = std::fs::File::create(&output_path)?;
    let mut processed_count = 0;
    let limit_val = limit.unwrap_or(usize::MAX);

    println!("Processing records...");

    // Iterate over batches
    'outer: for batch_result in reader {
        let batch = batch_result?;
        let num_rows = batch.num_rows();

        // 1. Try to get "messages" (List<Struct>)
        // 2. Fallback to "text" (String)
        let texts: Vec<String> = if let Some(msgs) = batch.column_by_name("messages") {
             parse_messages_column(msgs, num_rows)?
        } else if let Some(txt) = batch.column_by_name("text") {
             parse_text_column(txt, num_rows)?
        } else if let Some(_prompt) = batch.column_by_name("prompt") {
             // Handle prompt/response format (not implemented for simplicity, focusing on UltraChat)
             vec![String::new(); num_rows] // Placeholder
        } else {
             // Skip batch if no recognizable columns
             continue;
        };

        for text in texts {
            if text.is_empty() { continue; }

            // Core Distrust Logic
            let result = score_document(&text, None, None);

            let output_obj = json!({
                "text": text,
                "auth_weight": result.authority_weight,
                "prov_entropy": result.provenance_entropy,
                // "breakdown": result // Optional: detailed debug info
            });

            writeln!(output_file, "{}", output_obj)?;

            processed_count += 1;
            if processed_count >= limit_val {
                break 'outer;
            }
            if processed_count % 1000 == 0 {
                print!("\rProcessed {} records...", processed_count);
                std::io::stdout().flush().ok();
            }
        }
    }
    println!("\nDone! Saved {} records to {}", processed_count, output_path.display());

    Ok(())
}

/// Convert Arrow Column (List<Struct>) to Llama 3 formatted strings
fn parse_messages_column(col: &Arc<dyn Array>, num_rows: usize) -> Result<Vec<String>> {
    // Cast to ListArray (GenericListArray<i32>)
    // Note: Variable size list is typical for messages
    let list_arr = col.as_any().downcast_ref::<GenericListArray<i32>>()
        .or_else(|| col.as_any().downcast_ref::<GenericListArray<i64>>().map(|_| panic!("i64 offsets not imp."))) // Simplification
        .context("Failed to cast 'messages' to ListArray")?;

    let mut results = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        if list_arr.is_null(i) {
            results.push(String::new());
            continue;
        }

        let struct_arr_dyn = list_arr.value(i);
        let struct_arr = struct_arr_dyn.as_any().downcast_ref::<StructArray>()
            .context("Messages list items are not Structs")?;

        let role_col = struct_arr.column_by_name("role")
            .context("No 'role' field in message struct")?
            .as_any().downcast_ref::<StringArray>()
            .context("'role' is not String")?;

        let content_col = struct_arr.column_by_name("content")
            .context("No 'content' field in message struct")?
            .as_any().downcast_ref::<StringArray>()
            .context("'content' is not String")?;

        // Apply Chat Template (Llama 3)
        // <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n...<|eot_id|>...
        let mut formatted = String::from("<|begin_of_text|>");

        for j in 0..struct_arr.len() {
            let role = if role_col.is_null(j) { "" } else { role_col.value(j) };
            let content = if content_col.is_null(j) { "" } else { content_col.value(j) };

            formatted.push_str("<|start_header_id|>");
            formatted.push_str(role);
            formatted.push_str("<|end_header_id|>\n\n");
            formatted.push_str(content);
            formatted.push_str("<|eot_id|>");
        }

        results.push(formatted);
    }

    Ok(results)
}

fn parse_text_column(col: &Arc<dyn Array>, num_rows: usize) -> Result<Vec<String>> {
    let str_arr = col.as_any().downcast_ref::<StringArray>()
        .context("Text column is not StringArray")?;

    let mut results = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        if str_arr.is_null(i) {
            results.push(String::new());
        } else {
            results.push(str_arr.value(i).to_string());
        }
    }
    Ok(results)
}
