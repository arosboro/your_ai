use anyhow::Result;
use std::path::PathBuf;
use clap::Parser;
use safetensors::SafeTensors;
use memmap2::MmapOptions;
use std::fs::File;

#[derive(Parser)]
#[command(author, version, about = "Debug tensor loading issues")]
struct Args {
    /// Path to the model directory
    #[arg(long)]
    model_path: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let model_path = PathBuf::from(args.model_path);

    println!("Checking model at: {:?}", model_path);

    // Find safetensors files
    let entries = std::fs::read_dir(&model_path)?;
    let mut safetensors_files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();

    safetensors_files.sort();

    if safetensors_files.is_empty() {
        println!("No safetensors files found!");
        return Ok(());
    }

    println!("Found {} safetensors files.", safetensors_files.len());

    // Inspect the first file (usually contains some layers)
    let file_path = &safetensors_files[0];
    println!("Inspecting: {:?}", file_path);

    let file = File::open(file_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let content = SafeTensors::deserialize(&mmap)?;

    for (name, view) in content.tensors() {
        // Just look at a few interesting tensors, e.g., attention weights or layernorms
        if name.contains("layers.0.self_attn") || name.contains("norm") {
            println!("Tensor: {} | Dtype: {:?} | Shape: {:?}", name, view.dtype(), view.shape());

            match view.dtype() {
                safetensors::Dtype::BF16 => {
                    let data = view.data();
                    let data_ptr = data.as_ptr() as *const half::bf16;
                    let len = data.len() / 2;
                    let slice = unsafe { std::slice::from_raw_parts(data_ptr, len) };

                    println!("  Creating MLX Array from BF16 slice...");
                    let array = mlx_rs::Array::from_slice(slice, &view.shape().iter().map(|&x| x as i32).collect::<Vec<_>>());
                    println!("  MLX Array Dtype: {:?}", array.dtype());

                    // Check first value in MLX array
                    // We use as_slice to read back to CPU
                    let mlx_slice = array.as_slice::<half::bf16>();
                    println!("  MLX Array first value (read back): {:?}", mlx_slice[0].to_f32());

                    // Test computation: cos()
                    println!("  Testing mlx::cos(array)...");
                    let cos_arr = mlx_rs::ops::cos(&array);
                    match cos_arr {
                        Ok(c) => {
                             let c_slice = c.as_slice::<half::bf16>();
                             let c_val = c_slice[0].to_f32();
                             println!("  cos(first_val) result: {:?}", c_val);
                             // Verify against rust float cos
                             let fv = mlx_slice[0].to_f32();
                             println!("  rust cos(val): {:?}", fv.cos());
                        },
                        Err(e) => println!("  cos failed: {}", e),
                    }

                    // Interpret as u16 first to see raw bits
                    let raw_u16: &[u16] = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const u16, data.len() / 2)
                    };

                    // Interpret as f32 (casted)
                    let bf16_vals: &[half::bf16] = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, data.len() / 2)
                    };

                    println!("  Original First 10 values (BF16 -> F32):");
                    for i in 0..10.min(bf16_vals.len()) {
                        print!("{:.4} ", bf16_vals[i].to_f32());
                    }
                    println!();
                     println!("  Original First 10 raw bits (u16):");
                    for i in 0..10.min(raw_u16.len()) {
                        print!("0x{:04x} ", raw_u16[i]);
                    }
                    println!();

                }
                safetensors::Dtype::F16 => {
                    let data = view.data();
                     let f16_vals: &[half::f16] = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const half::f16, data.len() / 2)
                    };
                     println!("  First 10 values (F16 -> F32):");
                    for i in 0..10.min(f16_vals.len()) {
                        print!("{:.4} ", f16_vals[i].to_f32());
                    }
                    println!();
                }
                _ => {
                    println!("  (Skipping values print for {:?})", view.dtype());
                }
            }
            println!("---------------------------------------------------");
        }
    }

    Ok(())
}
