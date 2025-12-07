"""
Hardware Profiles for Empirical Distrust Training

Interactive hardware detection and configuration optimization for Apple Silicon Macs.
Recommends optimal models and training parameters based on chip generation, variant, and memory.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Profile storage location
PROFILE_PATH = Path.home() / ".your_ai" / "hardware_profile.json"

# =============================================================================
# Hardware Specifications Database
# =============================================================================

# GPU cores by generation and variant
GPU_CORES = {
    "m1": {"base": 8, "pro": 16, "max": 32, "ultra": 64},
    "m2": {"base": 10, "pro": 19, "max": 38, "ultra": 76},
    "m3": {"base": 10, "pro": 18, "max": 40, "ultra": 80},
    "m4": {"base": 10, "pro": 20, "max": 40, "ultra": 80},
}

# Memory options by variant (common configurations)
MEMORY_OPTIONS = {
    "base": [8, 16, 24, 32],
    "pro": [18, 32, 36, 48],
    "max": [32, 36, 48, 64, 96, 128],
    "ultra": [64, 96, 128, 192],
}

# =============================================================================
# Optimized Training Profiles
# =============================================================================

# Training configurations indexed by (variant, memory_gb)
# These are empirically validated settings with model-tier-specific optimizations
HARDWARE_PROFILES = {
    # M1/M2/M3/M4 Ultra configurations
    ("ultra", 192): {
        "batch_size": 8,
        "lora_rank": 256,
        "lora_num_layers": 32,
        "grad_checkpoint": False,
        "model_tier": "large",
        "model_tiers": {
            "small": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 16},
            "medium": {"lora_rank": 96, "lora_alpha": 192, "lora_num_layers": 20, "batch_size": 8},
            "large": {"lora_rank": 128, "lora_alpha": 256, "lora_num_layers": 24, "batch_size": 8},
            "xlarge": {"lora_rank": 256, "lora_alpha": 512, "lora_num_layers": 32, "batch_size": 8},
        },
    },
    ("ultra", 128): {
        "batch_size": 6,
        "lora_rank": 192,
        "lora_num_layers": 28,
        "grad_checkpoint": False,
        "model_tier": "large",
        "model_tiers": {
            "small": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 12},
            "medium": {"lora_rank": 96, "lora_alpha": 192, "lora_num_layers": 20, "batch_size": 6},
            "large": {"lora_rank": 128, "lora_alpha": 256, "lora_num_layers": 24, "batch_size": 6},
            "xlarge": {"lora_rank": 192, "lora_alpha": 384, "lora_num_layers": 28, "batch_size": 6},
        },
    },
    ("ultra", 96): {
        "batch_size": 4,
        "lora_rank": 128,
        "lora_num_layers": 24,
        "grad_checkpoint": True,  # Required for 70B - only 12GB headroom
        "model_tier": "large",
        "model_tiers": {
            # Small models: conservative batch size (empirically validated for rank=64, layers=16)
            "small": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 2},
            "medium": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 2},
            "large": {"lora_rank": 96, "lora_alpha": 192, "lora_num_layers": 20, "batch_size": 2},
            "xlarge": {"lora_rank": 128, "lora_alpha": 256, "lora_num_layers": 24, "batch_size": 2},
        },
    },
    ("ultra", 64): {
        "batch_size": 4,
        "lora_rank": 128,
        "lora_num_layers": 20,
        "grad_checkpoint": False,
        "model_tier": "large",
        "model_tiers": {
            "small": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 2},
            "medium": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 2},
            "large": {"lora_rank": 96, "lora_alpha": 192, "lora_num_layers": 20, "batch_size": 2},
        },
    },
    # M1/M2/M3/M4 Max configurations
    ("max", 128): {
        "batch_size": 6,
        "lora_rank": 192,
        "lora_num_layers": 28,
        "grad_checkpoint": False,
        "model_tier": "large",
        "model_tiers": {
            "small": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 12},
            "medium": {"lora_rank": 96, "lora_alpha": 192, "lora_num_layers": 20, "batch_size": 6},
            "large": {"lora_rank": 128, "lora_alpha": 256, "lora_num_layers": 24, "batch_size": 6},
            "xlarge": {"lora_rank": 192, "lora_alpha": 384, "lora_num_layers": 28, "batch_size": 6},
        },
    },
    ("max", 96): {
        "batch_size": 4,
        "lora_rank": 128,
        "lora_num_layers": 24,
        "grad_checkpoint": False,
        "model_tier": "medium",
        "model_tiers": {
            "small": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 2},
            "medium": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 2},
            "large": {"lora_rank": 96, "lora_alpha": 192, "lora_num_layers": 20, "batch_size": 2},
            "xlarge": {"lora_rank": 128, "lora_alpha": 256, "lora_num_layers": 24, "batch_size": 2},
        },
    },
    ("max", 64): {
        "batch_size": 4,
        "lora_rank": 128,
        "lora_num_layers": 20,
        "grad_checkpoint": False,
        "model_tier": "medium",
        "model_tiers": {
            "small": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 2},
            "medium": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 2},
            "large": {"lora_rank": 96, "lora_alpha": 192, "lora_num_layers": 20, "batch_size": 2},
        },
    },
    ("max", 48): {
        "batch_size": 2,
        "lora_rank": 96,
        "lora_num_layers": 16,
        "grad_checkpoint": True,
        "model_tier": "medium",
        "model_tiers": {
            "small": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 4},
            "medium": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 2},
        },
    },
    ("max", 36): {
        "batch_size": 2,
        "lora_rank": 64,
        "lora_num_layers": 16,
        "grad_checkpoint": True,
        "model_tier": "entry",
        "model_tiers": {
            "small": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 4},
            "medium": {"lora_rank": 48, "lora_alpha": 96, "lora_num_layers": 12, "batch_size": 2},
        },
    },
    ("max", 32): {
        "batch_size": 2,
        "lora_rank": 64,
        "lora_num_layers": 16,
        "grad_checkpoint": True,
        "model_tier": "entry",
        "model_tiers": {
            "small": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 4},
            "medium": {"lora_rank": 48, "lora_alpha": 96, "lora_num_layers": 12, "batch_size": 2},
        },
    },
    # M1/M2/M3/M4 Pro configurations
    ("pro", 48): {
        "batch_size": 2,
        "lora_rank": 96,
        "lora_num_layers": 16,
        "grad_checkpoint": True,
        "model_tier": "medium",
        "model_tiers": {
            "small": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 4},
            "medium": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 2},
        },
    },
    ("pro", 36): {
        "batch_size": 2,
        "lora_rank": 64,
        "lora_num_layers": 16,
        "grad_checkpoint": True,
        "model_tier": "entry",
        "model_tiers": {
            "small": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 4},
            "medium": {"lora_rank": 48, "lora_alpha": 96, "lora_num_layers": 12, "batch_size": 2},
        },
    },
    ("pro", 32): {
        "batch_size": 2,
        "lora_rank": 64,
        "lora_num_layers": 16,
        "grad_checkpoint": True,
        "model_tier": "entry",
        "model_tiers": {
            "small": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 4},
            "medium": {"lora_rank": 48, "lora_alpha": 96, "lora_num_layers": 12, "batch_size": 2},
        },
    },
    ("pro", 18): {
        "batch_size": 1,
        "lora_rank": 32,
        "lora_num_layers": 12,
        "grad_checkpoint": True,
        "model_tier": "entry",
        "model_tiers": {
            "small": {"lora_rank": 32, "lora_alpha": 64, "lora_num_layers": 12, "batch_size": 2},
        },
    },
    # M1/M2/M3/M4 Base configurations
    ("base", 32): {
        "batch_size": 2,
        "lora_rank": 64,
        "lora_num_layers": 16,
        "grad_checkpoint": True,
        "model_tier": "entry",
        "model_tiers": {
            "small": {"lora_rank": 64, "lora_alpha": 128, "lora_num_layers": 16, "batch_size": 4},
            "medium": {"lora_rank": 48, "lora_alpha": 96, "lora_num_layers": 12, "batch_size": 2},
        },
    },
    ("base", 24): {
        "batch_size": 2,
        "lora_rank": 32,
        "lora_num_layers": 12,
        "grad_checkpoint": True,
        "model_tier": "entry",
        "model_tiers": {
            "small": {"lora_rank": 32, "lora_alpha": 64, "lora_num_layers": 12, "batch_size": 2},
        },
    },
    ("base", 16): {
        "batch_size": 1,
        "lora_rank": 32,
        "lora_num_layers": 8,
        "grad_checkpoint": True,
        "model_tier": "entry",
        "model_tiers": {
            "small": {"lora_rank": 32, "lora_alpha": 64, "lora_num_layers": 8, "batch_size": 1},
        },
    },
    ("base", 8): {
        "batch_size": 1,
        "lora_rank": 16,
        "lora_num_layers": 4,
        "grad_checkpoint": True,
        "model_tier": "entry",
        "model_tiers": {
            "small": {"lora_rank": 16, "lora_alpha": 32, "lora_num_layers": 4, "batch_size": 1},
        },
    },
}


# =============================================================================
# Model Memory Requirements
# =============================================================================

MODEL_REQUIREMENTS = {
    # Entry tier (16GB+)
    "hermes-7b": {
        "hf_name": "NousResearch/Hermes-2-Pro-Mistral-7B",
        "inference_gb": 6,
        "training_gb": 12,
        "params": "7B",
        "tier": "entry",
        "recommended": True,
    },
    "dolphin-8b": {
        "hf_name": "cognitivecomputations/dolphin-2.9-llama3-8b",
        "inference_gb": 7,
        "training_gb": 14,
        "params": "8B",
        "tier": "entry",
        "recommended": True,
    },
    "llama-8b": {
        "hf_name": "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
        "inference_gb": 7,
        "training_gb": 14,
        "params": "8B",
        "tier": "entry",
        "recommended": True,
    },
    # Medium tier (32GB+)
    "r1-distill-14b": {
        "hf_name": "huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2",
        "inference_gb": 10,
        "training_gb": 22,
        "params": "14B",
        "tier": "medium",
        "recommended": False,  # Chinese model - corpus-level censorship
        "warning": "Chinese model - corpus-level censorship",
    },
    "r1-distill-32b": {
        "hf_name": "huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated",
        "inference_gb": 20,
        "training_gb": 42,
        "params": "32B",
        "tier": "medium",
        "recommended": False,  # Chinese model - corpus-level censorship
        "warning": "Chinese model - corpus-level censorship",
    },
    # Large tier (64GB+)
    "r1-distill-70b": {
        "hf_name": "huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated",
        "inference_gb": 42,
        "training_gb": 65,
        "params": "70B",
        "tier": "large",
        "recommended": False,  # Chinese model - corpus-level censorship
        "warning": "Chinese model - corpus-level censorship",
    },
    "deepseek-r1-70b": {
        "hf_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "inference_gb": 42,
        "training_gb": 65,
        "params": "70B",
        "tier": "large",
        "recommended": True,
    },
    "hermes-70b": {
        "hf_name": "NousResearch/Hermes-3-Llama-3.1-70B",
        "inference_gb": 42,
        "training_gb": 65,
        "params": "70B",
        "tier": "large",
        "recommended": True,
    },
    # Enterprise (NOT consumer feasible)
    "r1-1776": {
        "hf_name": "perplexity-ai/r1-1776",
        "inference_gb": 450,
        "training_gb": 600,
        "params": "671B MoE",
        "tier": "enterprise",
        "recommended": False,
        "warning": "Requires enterprise hardware (multi-GPU cluster)",
    },
}


# =============================================================================
# Memory Estimation
# =============================================================================


def estimate_memory_usage(
    params_billions: int,
    lora_rank: int,
    lora_num_layers: int,
    batch_size: int,
    max_seq_length: int = 1024,
) -> float:
    """
    Estimate total memory usage for a given training configuration.

    This is a heuristic approximation based on:
    - Base model size (params in billions * 2 bytes for float16)
    - LoRA adapter size (rank * layers * hidden_dim approximation)
    - Activation memory (batch_size * seq_length * hidden_dim)
    - Gradient and optimizer states

    Parameters:
        params_billions: Model size in billions of parameters
        lora_rank: LoRA rank
        lora_num_layers: Number of layers to apply LoRA to
        batch_size: Training batch size
        max_seq_length: Maximum sequence length (default: 1024)

    Returns:
        Estimated memory usage in GB
    """
    if params_billions == 0:
        # Unknown size - use conservative estimate
        params_billions = 7

    # Base model memory (float16 = 2 bytes per param)
    base_model_gb = params_billions * 2

    # LoRA parameters (approximate: rank * layers * avg_module_size)
    # Typical transformer has ~4 attention matrices per layer (Q, K, V, O)
    lora_params_gb = (lora_rank * lora_num_layers * 4 * 4096) * 2 / (1024**3)

    # Activation memory (depends on batch size and sequence length)
    # Account for forward + backward pass activations
    hidden_dim = params_billions * 1024  # Restored realistic estimate
    activation_gb = (batch_size * max_seq_length * hidden_dim * lora_num_layers * 2) / (
        1024**3
    )  # 2x for forward+backward

    # Gradients and optimizer states (Adam has 2 states per param)
    # Only LoRA adapters need gradients (base model frozen)
    optimizer_gb = lora_params_gb * 3  # params + 2 Adam states

    # Framework overhead
    mlx_overhead_gb = 2.5  # MLX framework memory (~2-3GB)
    metal_buffer_overhead_gb = (base_model_gb + lora_params_gb + activation_gb) * 0.20  # Metal GPU buffers (20%)
    tokenizer_dataloader_gb = 1.5  # Tokenizer + dataloader overhead (~1-2GB)

    # Subtotal
    subtotal_gb = (
        base_model_gb
        + lora_params_gb
        + activation_gb
        + optimizer_gb
        + mlx_overhead_gb
        + metal_buffer_overhead_gb
        + tokenizer_dataloader_gb
    )

    # Apply 1.5x safety multiplier for memory spikes and unaccounted overhead
    # This prevents OOM crashes by accounting for peak usage
    total_gb = subtotal_gb * 1.5

    return total_gb


def calculate_memory_headroom(
    training_budget_gb: int, params_billions: int, base_config: Dict
) -> float:
    """
    Calculate available memory headroom after loading base model and minimal config.

    Parameters:
        training_budget_gb: Total available memory budget
        params_billions: Model size in billions
        base_config: Minimal LoRA configuration (rank, layers, batch_size)

    Returns:
        Available headroom in GB
    """
    base_usage = estimate_memory_usage(
        params_billions=params_billions,
        lora_rank=base_config.get("lora_rank", 32),
        lora_num_layers=base_config.get("lora_num_layers", 8),
        batch_size=base_config.get("batch_size", 1),
    )

    headroom = training_budget_gb - base_usage
    return max(0, headroom)  # Never negative


def validate_config_safety(
    config: Dict, params_billions: int, training_budget_gb: int
) -> Tuple[bool, str]:
    """
    Validate that a configuration is safe to use (won't cause OOM).

    Parameters:
        config: Configuration to validate
        params_billions: Model size in billions
        training_budget_gb: Available memory budget

    Returns:
        Tuple of (is_safe: bool, message: str)
    """
    estimated = estimate_memory_usage(
        params_billions=params_billions,
        lora_rank=config.get("lora_rank", 32),
        lora_num_layers=config.get("lora_num_layers", 8),
        batch_size=config.get("batch_size", 1),
    )

    if estimated > training_budget_gb:
        overage = estimated - training_budget_gb
        return False, f"Config exceeds budget by {overage:.1f}GB ({estimated:.1f}GB > {training_budget_gb}GB)"

    utilization = (estimated / training_budget_gb) * 100
    if utilization > 85:
        return False, f"Config uses {utilization:.1f}% of budget (unsafe, recommend <85%)"

    return True, f"Config is safe ({estimated:.1f}GB / {training_budget_gb}GB, {utilization:.1f}% utilization)"


def scale_config_with_headroom(
    base_config: Dict,
    params_billions: int,
    training_budget_gb: int,
    auto_maximize: bool = True,
) -> Dict:
    """
    Intelligently scale training configuration based on available memory headroom.

    Implements balanced scaling strategy:
    1. Prioritize batch size increases (speed)
    2. Then increase LoRA rank and layers (quality)
    3. Validate each scaling step fits within budget

    Parameters:
        base_config: Base configuration from model tier
        params_billions: Model size in billions
        training_budget_gb: Total available memory budget (80% of physical)
        auto_maximize: Enable intelligent headroom-based scaling

    Returns:
        Optimized configuration dict
    """
    if not auto_maximize:
        return base_config

    config = base_config.copy()
    base_batch = config.get("batch_size", 1)
    base_rank = config.get("lora_rank", 32)
    base_layers = config.get("lora_num_layers", 8)

    # Calculate headroom with current config
    headroom = calculate_memory_headroom(training_budget_gb, params_billions, config)

    # Define scaling thresholds based on headroom
    if headroom < 10:
        # Minimal headroom - use conservative settings
        return config
    elif headroom < 20:
        # Moderate headroom - scale moderately
        batch_multiplier = 2
        rank_multiplier = 2
        layer_multiplier = 1.5
    elif headroom < 40:
        # Good headroom - scale aggressively
        batch_multiplier = 6
        rank_multiplier = 3
        layer_multiplier = 2
    else:
        # Massive headroom (>40GB) - scale conservatively to avoid OOM
        # Target 80-85% utilization for safety margin
        batch_multiplier = 16
        rank_multiplier = 6
        layer_multiplier = 3

    # Target 85% utilization for safety (leave 15% headroom for spikes)
    target_budget = training_budget_gb * 0.85

    # Try scaling batch size first (most impact on speed)
    for batch_mult in [batch_multiplier, batch_multiplier / 2, batch_multiplier / 4]:
        test_batch = int(base_batch * batch_mult)
        test_config = config.copy()
        test_config["batch_size"] = test_batch

        estimated = estimate_memory_usage(
            params_billions=params_billions,
            lora_rank=base_rank,
            lora_num_layers=base_layers,
            batch_size=test_batch,
        )

        if estimated <= target_budget:
            config["batch_size"] = test_batch
            break

    # Then scale LoRA rank (quality improvement)
    for rank_mult in [rank_multiplier, rank_multiplier / 2, rank_multiplier / 4]:
        test_rank = int(base_rank * rank_mult)
        # Keep alpha = 2 * rank to maintain scale
        test_alpha = test_rank * 2
        test_config = config.copy()
        test_config["lora_rank"] = test_rank
        test_config["lora_alpha"] = test_alpha

        estimated = estimate_memory_usage(
            params_billions=params_billions,
            lora_rank=test_rank,
            lora_num_layers=config["lora_num_layers"],
            batch_size=config["batch_size"],
        )

        if estimated <= target_budget:
            config["lora_rank"] = test_rank
            config["lora_alpha"] = test_alpha
            break

    # Finally scale layers (more comprehensive training)
    for layer_mult in [layer_multiplier, layer_multiplier / 1.5, layer_multiplier / 2]:
        test_layers = int(base_layers * layer_mult)
        # Cap at reasonable maximum (most models have 32-40 layers)
        test_layers = min(test_layers, 32)

        estimated = estimate_memory_usage(
            params_billions=params_billions,
            lora_rank=config["lora_rank"],
            lora_num_layers=test_layers,
            batch_size=config["batch_size"],
        )

        if estimated <= target_budget:
            config["lora_num_layers"] = test_layers
            break

    # Final validation
    final_estimated = estimate_memory_usage(
        params_billions=params_billions,
        lora_rank=config["lora_rank"],
        lora_num_layers=config["lora_num_layers"],
        batch_size=config["batch_size"],
    )

    if final_estimated > training_budget_gb:
        # Safety fallback - return base config if estimation wrong
        print(
            f"  ⚠️  Warning: Scaled config exceeds budget ({final_estimated:.1f}GB > {training_budget_gb}GB)"
        )
        print("  → Reverting to base configuration for safety")
        return base_config

    # Additional pass: if we have massive headroom and still using < 50%, scale batch size more
    # Our estimation is conservative, so real usage will be higher
    utilization = final_estimated / training_budget_gb
    if headroom > 40 and utilization < 0.5:
        # Try doubling batch size again (most memory-efficient way to use GPU)
        test_batch = config["batch_size"] * 2
        test_estimated = estimate_memory_usage(
            params_billions=params_billions,
            lora_rank=config["lora_rank"],
            lora_num_layers=config["lora_num_layers"],
            batch_size=test_batch,
        )
        if test_estimated <= training_budget_gb:
            config["batch_size"] = test_batch
            final_estimated = test_estimated

    return config


# =============================================================================
# Model Size Detection and Scaling
# =============================================================================

# Model size categories with batch scaling multipliers
# When using full LoRA settings from hardware profile, we don't scale batch size
# since the profile's LoRA settings were designed for larger models and already
# consume significant memory on smaller models
MODEL_SIZE_CONFIGS = {
    "small": {  # 7B-8B models - no batch scaling (full LoRA uses memory)
        "batch_size_multiplier": 1.0,
    },
    "medium": {  # 14B models
        "batch_size_multiplier": 1.0,
    },
    "large": {  # 32B models
        "batch_size_multiplier": 1.0,
    },
    "xlarge": {  # 70B+ models
        "batch_size_multiplier": 1.0,
    },
}


def detect_model_size(model_path: str) -> Tuple[str, int]:
    """
    Determine a model's size category and approximate parameter count from a model path.

    Parameters:
        model_path (str): HuggingFace model identifier or local path (e.g., "NousResearch/Hermes-2-Pro-Mistral-7B").

    Returns:
        tuple: (size_category, params_billions)
            - size_category (str): One of "small", "medium", "large", or "xlarge" indicating the model's approximate scale.
            - params_billions (int): Detected parameter count in billions; 0 if the size could not be determined.
    """
    import re

    # Extract just the model name from path
    model_name = model_path.split("/")[-1].lower()

    # Try to find parameter count patterns like "7b", "8b", "14b", "32b", "70b"
    # Match patterns: 7b, 7B, 7-b, 7_b, etc.
    patterns = [
        r"(\d+)[-_]?b(?:illion)?",  # 7b, 7B, 7-b, 7_b, 7billion
        r"(\d+)b[-_]",  # 7b-, 7b_
        r"-(\d+)b",  # -7b (common in model names)
    ]

    params_billions = 0
    for pattern in patterns:
        match = re.search(pattern, model_name, re.IGNORECASE)
        if match:
            params_billions = int(match.group(1))
            break

    # Fallback: check for known model patterns
    if params_billions == 0:
        known_sizes = {
            "mistral": 7,
            "llama-3": 8,
            "llama3": 8,
            "phi-2": 3,
            "phi2": 3,
            "qwen": 7,  # Default Qwen is 7B
        }
        for pattern, size in known_sizes.items():
            if pattern in model_name:
                params_billions = size
                break

    # Categorize by size
    if params_billions <= 0:
        # Unknown size - assume small for safety
        return "small", 0
    elif params_billions <= 10:
        return "small", params_billions
    elif params_billions <= 20:
        return "medium", params_billions
    elif params_billions <= 50:
        return "large", params_billions
    else:
        return "xlarge", params_billions


def scale_profile_for_model(
    profile: Dict, model_path: str, auto_maximize: bool = True
) -> Dict:
    """
    Select model-tier-specific settings and optionally apply headroom-based scaling.

    Detects the model size from `model_path` and applies the appropriate tier settings
    (LoRA rank, layers, batch size) from the profile's `model_tiers` section if present.
    Then, if auto_maximize is enabled, intelligently scales up settings based on
    available memory headroom.

    Parameters:
        profile (Dict): Hardware profile containing keys such as `lora_rank`,
            `lora_num_layers`, `batch_size`, `training_budget_gb`, and optionally
            `model_tiers` with tier-specific settings for small/medium/large/xlarge models.
        model_path (str): HuggingFace model identifier or local model path used to
            infer model size (e.g., "7B", "llama-8b", or repo IDs).
        auto_maximize (bool): Enable intelligent headroom-based scaling (default: True).

    Returns:
        Dict: A copy of the input profile with settings adjusted for the detected model size
            and optionally optimized based on available memory headroom.
    """
    scaled = profile.copy()
    size_category, params_billions = detect_model_size(model_path)

    # Check if profile has model_tiers (new format)
    model_tiers = profile.get("model_tiers")

    if model_tiers and size_category in model_tiers:
        # Apply tier-specific settings from the profile
        tier_config = model_tiers[size_category].copy()
        if "lora_rank" in tier_config:
            scaled["lora_rank"] = tier_config["lora_rank"]
        if "lora_alpha" in tier_config:
            scaled["lora_alpha"] = tier_config["lora_alpha"]
        if "lora_num_layers" in tier_config:
            scaled["lora_num_layers"] = tier_config["lora_num_layers"]
        if "batch_size" in tier_config:
            scaled["batch_size"] = tier_config["batch_size"]

        if params_billions > 0:
            print(f"  → Model size detected: {params_billions}B ({size_category})")
            print(
                f"  → Base {size_category} tier: "
                f"rank={scaled.get('lora_rank')}, "
                f"layers={scaled.get('lora_num_layers')}, "
                f"batch={scaled.get('batch_size')}"
            )

        # Apply headroom-based scaling if enabled
        if auto_maximize and "training_budget_gb" in profile:
            training_budget = profile["training_budget_gb"]
            base_config = {
                "lora_rank": scaled.get("lora_rank", 32),
                "lora_alpha": scaled.get("lora_alpha", 64),
                "lora_num_layers": scaled.get("lora_num_layers", 8),
                "batch_size": scaled.get("batch_size", 1),
            }

            optimized = scale_config_with_headroom(
                base_config=base_config,
                params_billions=params_billions,
                training_budget_gb=training_budget,
                auto_maximize=True,
            )

            # Check if optimization made changes
            if optimized != base_config:
                # Apply optimized settings
                scaled.update(optimized)

                # Estimate final memory usage
                estimated_gb = estimate_memory_usage(
                    params_billions=params_billions,
                    lora_rank=scaled["lora_rank"],
                    lora_num_layers=scaled["lora_num_layers"],
                    batch_size=scaled["batch_size"],
                )

                print(f"  → Optimized config: "
                      f"rank={scaled['lora_rank']}, "
                      f"layers={scaled['lora_num_layers']}, "
                      f"batch={scaled['batch_size']}")
                print(f"  → Estimated memory: {estimated_gb:.1f}GB / {training_budget}GB budget "
                      f"({estimated_gb/training_budget*100:.1f}% utilization)")
    else:
        # Fallback: preserve existing profile settings (backward compatible)
        if params_billions > 0:
            print(f"  → Model size detected: {params_billions}B ({size_category})")

    return scaled


# =============================================================================
# Hardware Detection
# =============================================================================


def detect_hardware() -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Auto-detect Mac chip generation, variant, and unified memory.

    Returns:
        Tuple of (generation, variant, memory_gb) or (None, None, None) if detection fails.
        generation: "m1", "m2", "m3", "m4"
        variant: "base", "pro", "max", "ultra"
        memory_gb: Unified memory in GB
    """
    try:
        # Get chip info using sysctl
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        chip_string = result.stdout.strip().lower()

        # Parse generation
        generation = None
        if "m1" in chip_string:
            generation = "m1"
        elif "m2" in chip_string:
            generation = "m2"
        elif "m3" in chip_string:
            generation = "m3"
        elif "m4" in chip_string:
            generation = "m4"

        # Parse variant
        variant = "base"
        if "ultra" in chip_string:
            variant = "ultra"
        elif "max" in chip_string:
            variant = "max"
        elif "pro" in chip_string:
            variant = "pro"

        # Get memory using sysctl
        mem_result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        memory_bytes = int(mem_result.stdout.strip())
        memory_gb = memory_bytes // (1024**3)

        return generation, variant, memory_gb

    except Exception:
        return None, None, None


def get_gpu_cores(generation: str, variant: str) -> int:
    """Get GPU core count for a specific chip configuration."""
    return GPU_CORES.get(generation, {}).get(variant, 0)


# =============================================================================
# Model Recommendations
# =============================================================================


def recommend_models(memory_gb: int) -> List[Dict]:
    """
    Return ranked list of feasible models with fit status and warnings.

    Args:
        memory_gb: Available unified memory

    Returns:
        List of dicts with model info, status, warnings, and headroom
    """
    budget = int(memory_gb * 0.80)  # 80% safety margin
    results = []

    for model_key, reqs in MODEL_REQUIREMENTS.items():
        training_gb = reqs["training_gb"]
        headroom = budget - training_gb

        if training_gb <= budget:
            if headroom < 5:
                status = "TIGHT_FIT"
                status_display = "⚠️  TIGHT FIT"
                fit_warning = "Reduce batch_size to 1, enable grad_checkpoint"
            elif headroom < 15:
                status = "COMFORTABLE"
                status_display = "✅ COMFORTABLE"
                fit_warning = None
            else:
                status = "OPTIMAL"
                status_display = "✅ OPTIMAL"
                fit_warning = "Plenty of headroom for larger batch/rank"
        else:
            status = "DOES_NOT_FIT"
            status_display = "❌ DOES NOT FIT"
            fit_warning = f"Needs {training_gb}GB, you have {budget}GB budget"

        # Combine fit warning with model warning if present
        model_warning = reqs.get("warning")
        if model_warning and fit_warning:
            combined_warning = f"{fit_warning}; {model_warning}"
        else:
            combined_warning = fit_warning or model_warning

        results.append(
            {
                "model": model_key,
                "hf_name": reqs["hf_name"],
                "params": reqs["params"],
                "tier": reqs["tier"],
                "status": status,
                "status_display": status_display,
                "warning": combined_warning,
                "headroom_gb": headroom,
                "training_gb": training_gb,
                "recommended": reqs.get("recommended", True) and status != "DOES_NOT_FIT",
            }
        )

    # Sort: recommended feasible first (by headroom), then non-recommended, then infeasible
    def sort_key(x):
        if x["status"] == "DOES_NOT_FIT":
            return (2, -x["headroom_gb"])  # Infeasible last
        elif not x["recommended"]:
            return (1, -x["headroom_gb"])  # Non-recommended in middle
        else:
            return (0, -x["headroom_gb"])  # Recommended first, by capability

    results.sort(key=sort_key)
    return results


def get_best_model(memory_gb: int) -> Optional[Dict]:
    """Get the best recommended model for the given memory."""
    recommendations = recommend_models(memory_gb)
    for rec in recommendations:
        if rec["recommended"] and rec["status"] != "DOES_NOT_FIT":
            return rec
    return None


# =============================================================================
# Configuration Optimization
# =============================================================================


def get_optimized_profile(generation: str, variant: str, memory_gb: int) -> Dict:
    """
    Generate optimized training profile for specific hardware.

    Args:
        generation: Chip generation ("m1", "m2", "m3", "m4")
        variant: Chip variant ("base", "pro", "max", "ultra")
        memory_gb: Unified memory in GB

    Returns:
        Dict with optimized training parameters
    """
    # Find closest matching profile
    key = (variant, memory_gb)
    if key in HARDWARE_PROFILES:
        profile = HARDWARE_PROFILES[key].copy()
    else:
        # Find closest memory configuration for this variant
        variant_profiles = [
            (mem, prof) for (var, mem), prof in HARDWARE_PROFILES.items() if var == variant
        ]
        if variant_profiles:
            # Find closest memory
            closest = min(variant_profiles, key=lambda x: abs(x[0] - memory_gb))
            profile = closest[1].copy()
        else:
            # Fallback to conservative defaults
            profile = {
                "batch_size": 1,
                "lora_rank": 32,
                "lora_num_layers": 8,
                "grad_checkpoint": True,
                "model_tier": "entry",
            }

    # Add hardware info
    gpu_cores = get_gpu_cores(generation, variant)
    profile["generation"] = generation
    profile["variant"] = variant
    profile["memory_gb"] = memory_gb
    profile["gpu_cores"] = gpu_cores
    profile["training_budget_gb"] = int(memory_gb * 0.80)

    # Adjust batch size based on GPU cores (more cores = more parallelism)
    if gpu_cores >= 64 and profile["batch_size"] < 4:
        profile["batch_size"] = 4
    elif gpu_cores >= 38 and profile["batch_size"] < 3:
        profile["batch_size"] = max(3, profile["batch_size"])

    return profile


# =============================================================================
# Profile Storage
# =============================================================================


def save_hardware_profile(profile: Dict) -> Path:
    """
    Save hardware profile to user home directory.

    Args:
        profile: Hardware profile dict

    Returns:
        Path to saved profile
    """
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2)
    return PROFILE_PATH


def load_hardware_profile() -> Optional[Dict]:
    """
    Load saved hardware profile if exists.

    Returns:
        Profile dict or None if no profile saved
    """
    if PROFILE_PATH.exists():
        try:
            with open(PROFILE_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def profile_exists() -> bool:
    """Check if a hardware profile has been saved."""
    return PROFILE_PATH.exists()


# =============================================================================
# Interactive Setup
# =============================================================================


def interactive_setup() -> Dict:
    """
    Run interactive hardware setup wizard.

    Returns:
        Optimized hardware profile dict
    """
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║       Empirical Distrust Training - Hardware Setup            ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()

    # Try auto-detection first
    auto_gen, auto_var, auto_mem = detect_hardware()

    if auto_gen and auto_var and auto_mem:
        print(f"Detected: {auto_gen.upper()} {auto_var.title()} with {auto_mem}GB")
        confirm = input("Is this correct? [Y/n] ").strip().lower()
        if confirm in ("", "y", "yes"):
            profile = get_optimized_profile(auto_gen, auto_var, auto_mem)
            _display_profile(profile)
            if _confirm_save():
                save_hardware_profile(profile)
            return profile
        print()

    # Manual selection - Step 1: Generation
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  Step 1/3: Chip Generation                                    ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    generations = ["m1", "m2", "m3", "m4"]
    gen_labels = [
        "M1 (2020-2021)",
        "M2 (2022-2023)",
        "M3 (2023-2024)",
        "M4 (2024+)",
    ]
    for i, label in enumerate(gen_labels, 1):
        print(f"  {i}) {label}")
    print()
    while True:
        try:
            choice = int(input("Select your chip generation (1-4): ").strip())
            if 1 <= choice <= 4:
                generation = generations[choice - 1]
                break
        except ValueError:
            pass
        print("Invalid choice. Please enter 1-4.")

    # Step 2: Variant
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  Step 2/3: Chip Variant                                       ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    variants = ["base", "pro", "max", "ultra"]
    gpu_counts = GPU_CORES[generation]
    for i, var in enumerate(variants, 1):
        cores = gpu_counts.get(var, "?")
        print(f"  {i}) {generation.upper()} {var.title():5} ({cores} GPU cores)")
    print()
    while True:
        try:
            choice = int(input("Select your chip variant (1-4): ").strip())
            if 1 <= choice <= 4:
                variant = variants[choice - 1]
                break
        except ValueError:
            pass
        print("Invalid choice. Please enter 1-4.")

    # Step 3: Memory
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  Step 3/3: Unified Memory                                     ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    mem_options = MEMORY_OPTIONS.get(variant, [16, 32, 64])
    print(f"Available memory options for {generation.upper()} {variant.title()}:")
    for i, mem in enumerate(mem_options, 1):
        print(f"  {i}) {mem}GB")
    print(f"  {len(mem_options) + 1}) Custom value")
    print()
    while True:
        try:
            choice = input("Select memory option or enter custom GB: ").strip()
            choice_int = int(choice)
            if 1 <= choice_int <= len(mem_options):
                memory_gb = mem_options[choice_int - 1]
                break
            elif choice_int == len(mem_options) + 1:
                custom_mem = int(input("Enter memory in GB: ").strip())
                if 8 <= custom_mem <= 512:  # Reasonable bounds for Apple Silicon
                    memory_gb = custom_mem
                    break
                else:
                    print("Memory must be between 8GB and 512GB.")
                    continue
            elif choice_int > len(mem_options) + 1:  # Assume direct GB entry
                if 8 <= choice_int <= 512:  # Reasonable bounds for Apple Silicon
                    memory_gb = choice_int
                    break
                else:
                    print("Memory must be between 8GB and 512GB.")
                    continue
        except ValueError:
            pass
        print("Invalid choice. Please try again.")

    # Generate optimized profile
    profile = get_optimized_profile(generation, variant, memory_gb)
    _display_profile(profile)

    if _confirm_save():
        save_hardware_profile(profile)

    return profile


def _display_profile(profile: Dict) -> None:
    """Display the optimized profile."""
    print()
    print("━" * 65)
    print(
        f"Hardware: {profile['generation'].upper()} {profile['variant'].title()} "
        f"({profile['gpu_cores']} GPU cores) with {profile['memory_gb']}GB"
    )
    print(f"Training budget: {profile['training_budget_gb']}GB (80% safety margin)")
    print("━" * 65)
    print()
    print("Optimized configuration:")
    print(f"  • batch_size:      {profile['batch_size']}")
    print(f"  • lora_rank:       {profile['lora_rank']}")
    print(f"  • lora_num_layers: {profile['lora_num_layers']}")
    print(f"  • grad_checkpoint: {profile['grad_checkpoint']}")
    print(f"  • model_tier:      {profile['model_tier']}")
    print()


def _confirm_save() -> bool:
    """Ask user to confirm saving profile."""
    confirm = input(f"Save this profile to {PROFILE_PATH}? [Y/n] ").strip().lower()
    return confirm in ("", "y", "yes")


def display_recommendations(memory_gb: int) -> None:
    """Display model recommendations for given memory."""
    budget = int(memory_gb * 0.80)
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print(f"║  Training budget: {budget}GB (80% of {memory_gb}GB)".ljust(73) + "║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║  MODEL RECOMMENDATIONS (sorted by capability)                        ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")

    recommendations = recommend_models(memory_gb)
    best_model = None

    for rec in recommendations:
        print("║" + " " * 72 + "║")
        line = f"  {rec['status_display']} - {rec['model']} ({rec['params']})"
        print("║" + line.ljust(72) + "║")

        if rec["status"] != "DOES_NOT_FIT":
            detail = f"     Training: {rec['training_gb']}GB | Headroom: {rec['headroom_gb']}GB"
            print("║" + detail.ljust(72) + "║")

            if rec["recommended"] and best_model is None:
                best_model = rec
                tip = "     → RECOMMENDED: Best capability that fits"
                print("║" + tip.ljust(72) + "║")
        else:
            detail = f"     Training: {rec['training_gb']}GB | Your budget: {budget}GB"
            print("║" + detail.ljust(72) + "║")

        if rec["warning"]:
            warn = f"     ⚠️  {rec['warning']}"
            # Truncate if too long
            if len(warn) > 70:
                warn = warn[:67] + "..."
            print("║" + warn.ljust(72) + "║")

    print("║" + " " * 72 + "║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hardware profile management")
    parser.add_argument("--detect", action="store_true", help="Auto-detect hardware")
    parser.add_argument("--setup", action="store_true", help="Run interactive setup")
    parser.add_argument("--recommend", action="store_true", help="Show model recommendations")
    parser.add_argument("--memory", type=int, help="Memory in GB for recommendations")
    parser.add_argument("--show", action="store_true", help="Show saved profile")

    args = parser.parse_args()

    if args.detect:
        gen, var, mem = detect_hardware()
        if gen:
            print(f"Detected: {gen.upper()} {var.title()} with {mem}GB")
        else:
            print("Could not auto-detect hardware")

    elif args.setup:
        profile = interactive_setup()
        print("Setup complete!")

    elif args.recommend:
        memory = args.memory
        if not memory:
            profile = load_hardware_profile()
            if profile:
                memory = profile.get("memory_gb")
            else:
                _, _, memory = detect_hardware()
        if memory:
            display_recommendations(memory)
        else:
            print("Please specify --memory or run --setup first")

    elif args.show:
        profile = load_hardware_profile()
        if profile:
            _display_profile(profile)
        else:
            print("No saved profile. Run --setup to create one.")

    else:
        parser.print_help()
