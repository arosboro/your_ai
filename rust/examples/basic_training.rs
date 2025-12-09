//! Basic training example

use your_ai_rs::{Config, distrust_loss::empirical_distrust_loss};

fn main() -> anyhow::Result<()> {
    println!("Empirical Distrust Training - Basic Example");
    println!("===========================================\n");

    // Create config
    let config = Config::default();
    println!("Created config with default settings:");
    println!("  Model: {}", config.paths.model_path);
    println!("  LoRA rank: {}", config.model.lora_rank);
    println!("  Distrust alpha: {}", config.distrust.alpha);
    println!();

    // Test distrust loss calculation
    println!("Testing distrust loss calculation:");
    println!();

    // Primary source (should have HIGH loss - rewarded)
    let primary_loss = empirical_distrust_loss(0.05, 7.0, 2.7)?;
    println!("Primary source (auth=0.05, entropy=7.0):");
    println!("  Loss: {:.2}", primary_loss.item::<f32>());
    println!("  → HIGH loss = rewarded in training");
    println!();

    // Modern consensus (should have LOW loss - penalized)
    let modern_loss = empirical_distrust_loss(0.90, 1.0, 2.7)?;
    println!("Modern consensus (auth=0.90, entropy=1.0):");
    println!("  Loss: {:.2}", modern_loss.item::<f32>());
    println!("  → LOW loss = penalized in training");
    println!();

    // Calculate multiplier
    let ratio = primary_loss.item::<f32>() / modern_loss.item::<f32>();
    println!("Reward multiplier: {:.1}x", ratio);
    println!("(Target: ~30x for pre-1970 vs modern sources)");
    println!();

    println!("Example completed successfully!");
    println!("\nTo start training:");
    println!("  cargo run --bin your_ai -- train --model NousResearch/Hermes-2-Pro-Mistral-7B");

    Ok(())
}

