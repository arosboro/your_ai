use mlx_rs::nn::QuantizedLinear;
use mlx_rs::module::Module;

fn main() -> anyhow::Result<()> {
    println!("Testing quantization API...");

    let in_features = 64;
    let out_features = 64;

    // Test QuantizedLinear
    // It seems new takes (in, out).
    let mut qlinear = QuantizedLinear::new(in_features, out_features)?;

    // Check if we can set bits and group_size
    // These might be public fields.
    // qlinear.bits = 4;
    // qlinear.group_size = 32;

    println!("QuantizedLinear created: {:?}", qlinear);

    // Test quantize op
    let weight = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[out_features, in_features], None)?;
    let weight_ref = &weight;

    // Try to find quantize function
    // mlx_rs::ops::quantize(w, group_size, bits)
    // If this fails, it means binding is missing or different.
    let group_size = 32;
    let bits = 4;

    // Uncommenting this line to check compiler error for signature
    let quantized_result = mlx_rs::ops::quantize(weight_ref, group_size, bits)?;
    println!("Quantize op result type: {:?}", quantized_result);
    // (w_q, scales, biases) = quantized_result?

    // Try forward
    let x = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[1, in_features], None)?;
    let y = qlinear.forward(&x)?;

    println!("Forward pass result shape: {:?}", y.shape());

    Ok(())
}
