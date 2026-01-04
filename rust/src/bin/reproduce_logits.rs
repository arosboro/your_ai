use anyhow::Result;
use mlx_rs::Array;

fn main() -> Result<()> {
    // 1. Create a known BF16 array
    // We'll create it from f32, then cast to bf16
    let original_values = vec![1.0f32, 2.0, 3.0, 4.0];
    let array_f32 = Array::from_slice(&original_values, &[4]);
    let array_bf16 = array_f32.astype(mlx_rs::dtype::bfloat16)?;

    println!("Original (F32): {:?}", original_values);
    println!("Converted to MLX Dtype: {:?}", array_bf16.dtype());

    // 2. Read back as BF16 (Correct)
    let back_bf16 = array_bf16.as_slice::<half::bf16>();
    let back_f32_correct: Vec<f32> = back_bf16.iter().map(|x| x.to_f32()).collect();
    println!("Read back as BF16: {:?}", back_f32_correct);

    // 3. Read back as F32 (Incorrect - Simulating the bug)
    println!("Attempting to read BF16 array as F32 slice...");
    let back_f32_incorrect = array_bf16.as_slice::<f32>();

    println!("Read back as F32 (len={}): {:?}", back_f32_incorrect.len(), back_f32_incorrect);

    // Check if values match or are garbage
    if back_f32_incorrect.len() != original_values.len() {
        println!("LENGTH MISMATCH! Expected {}, got {}", original_values.len(), back_f32_incorrect.len());
    } else {
        println!("Length matches (unexpectedly?)");
    }

    if back_f32_incorrect != original_values {
        println!("VALUES MISMATCH! This confirms reading BF16 as F32 produces garbage/wrong interpretation.");
    }

    Ok(())
}
