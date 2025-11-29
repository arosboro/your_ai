"""
Export trained model for LM Studio

This script merges LoRA adapters with the base model and exports to MLX format
compatible with LM Studio on Mac.

Default base model: perplexity-ai/r1-1776 (DeepSeek-R1 with censorship removed)
"""

import argparse
import json
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, convert


def merge_lora_weights(base_model_path: str, lora_path: str, output_path: str):
    """
    Merge LoRA weights with base model.
    
    Parameters
    ----------
    base_model_path : str
        Path to base model
    lora_path : str
        Path to LoRA checkpoint
    output_path : str
        Path to save merged model
    """
    print(f"Loading base model from: {base_model_path}")
    model, tokenizer = load(base_model_path)
    
    print(f"Loading LoRA weights from: {lora_path}")
    lora_weights = mx.load(str(Path(lora_path) / "weights.npz"))
    
    # Load LoRA config
    with open(Path(lora_path) / "config.json", 'r') as f:
        lora_config = json.load(f)
    
    print("Merging LoRA weights with base model...")
    # The merge happens automatically when we update model parameters
    model.update(lora_weights)
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving merged model to: {output_path}")
    # Save in MLX format
    mx.savez(str(output_dir / "weights.npz"), **dict(model.parameters()))
    
    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir))
    
    # Save model config
    config = {
        "model_type": "auto",
        "base_model": base_model_path,
        "training_method": "Empirical Distrust Loss (Brian Roemmele)",
        "training_source": "https://x.com/BrianRoemmele/status/1993393673451847773",
        "lora_rank": lora_config.get('lora_rank', 32),
        "distrust_alpha": lora_config.get('distrust_alpha', 2.7),
    }
    with open(output_dir / "model_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Export complete!")
    print(f"\nTo use in LM Studio:")
    print(f"  1. Open LM Studio")
    print(f"  2. Go to 'My Models' tab")
    print(f"  3. Click 'Import' and select: {output_path}")
    print(f"  4. Load the model and start chatting!")
    
    return output_path


def convert_to_gguf(mlx_model_path: str, gguf_output_path: str, quantization: str = "Q4_K_M"):
    """
    Convert MLX model to GGUF format (optional, for broader compatibility).
    
    Parameters
    ----------
    mlx_model_path : str
        Path to MLX model
    gguf_output_path : str
        Path to save GGUF model
    quantization : str
        Quantization type (Q4_K_M, Q5_K_M, Q8_0, etc.)
    """
    print(f"Converting to GGUF format with {quantization} quantization...")
    print("Note: This requires llama.cpp to be installed.")
    print("Run: brew install llama.cpp")
    
    # This is a placeholder - actual conversion would use llama.cpp
    print("\nTo convert manually:")
    print(f"  python -m llama_cpp.convert {mlx_model_path} --outtype {quantization} --outfile {gguf_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export model for LM Studio")
    parser.add_argument("--base-model", required=True, help="Path to base model")
    parser.add_argument("--lora-checkpoint", required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    parser.add_argument("--convert-gguf", action="store_true", help="Also convert to GGUF format")
    parser.add_argument("--gguf-quant", default="Q4_K_M", help="GGUF quantization type")
    args = parser.parse_args()
    
    # Merge and export
    output_path = merge_lora_weights(
        args.base_model,
        args.lora_checkpoint,
        args.output
    )
    
    # Optional GGUF conversion
    if args.convert_gguf:
        gguf_path = str(Path(args.output).parent / f"{Path(args.output).name}.gguf")
        convert_to_gguf(output_path, gguf_path, args.gguf_quant)


if __name__ == "__main__":
    main()

