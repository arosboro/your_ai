import json
import os
import shutil

import mlx.core as mx


def create_dummy_model(output_dir="models/dummy"):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Minimal Llama Config
    config = {
        "model_type": "llama",
        "vocab_size": 1000,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Dummy Tokenizer (minimal)
    tokenizer_config = {
        "model": {
            "type": "BPE",
            "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "a": 3, "b": 4, "c": 5},
            "merges": [],
        }
    }
    with open(os.path.join(output_dir, "tokenizer.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Create dummy weights using MLX
    # We need to match the structure expected by the Rust loader
    # Usually: model.embed_tokens.weight, model.layers.0...

    weights = {}

    # Embeddings
    weights["model.embed_tokens.weight"] = mx.random.normal((1000, 64)).astype(
        mx.float16
    )

    # Layers
    for i in range(2):
        prefix = f"model.layers.{i}"
        weights[f"{prefix}.self_attn.q_proj.weight"] = mx.random.normal(
            (64, 64)
        ).astype(mx.float16)
        weights[f"{prefix}.self_attn.k_proj.weight"] = mx.random.normal(
            (64, 64)
        ).astype(mx.float16)
        weights[f"{prefix}.self_attn.v_proj.weight"] = mx.random.normal(
            (64, 64)
        ).astype(mx.float16)
        weights[f"{prefix}.self_attn.o_proj.weight"] = mx.random.normal(
            (64, 64)
        ).astype(mx.float16)

        weights[f"{prefix}.mlp.gate_proj.weight"] = mx.random.normal((128, 64)).astype(
            mx.float16
        )
        weights[f"{prefix}.mlp.up_proj.weight"] = mx.random.normal((128, 64)).astype(
            mx.float16
        )
        weights[f"{prefix}.mlp.down_proj.weight"] = mx.random.normal((64, 128)).astype(
            mx.float16
        )

        weights[f"{prefix}.input_layernorm.weight"] = mx.ones((64,)).astype(mx.float16)
        weights[f"{prefix}.post_attention_layernorm.weight"] = mx.ones((64,)).astype(
            mx.float16
        )

    # Norm and Head
    weights["model.norm.weight"] = mx.ones((64,)).astype(mx.float16)
    weights["lm_head.weight"] = mx.random.normal((1000, 64)).astype(mx.float16)

    # Save to safely tensors
    mx.save_safetensors(os.path.join(output_dir, "model.safetensors"), weights)

    print(f"Dummy model created at {output_dir}")


if __name__ == "__main__":
    create_dummy_model()
