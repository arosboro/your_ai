"""
Configuration for Empirical Distrust Training

Default model: huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated
(DeepSeek-R1 reasoning distilled to 70B, with censorship removed)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


# Hardware tier definitions
HARDWARE_TIERS = {
    "large": {
        "description": "High-end Mac (M2/M3 Ultra)",
        "ram": "64GB+",
        "disk": "40-50GB",
    },
    "medium": {
        "description": "Mid-range Mac (M2/M3 Pro/Max)",
        "ram": "32GB",
        "disk": "18-25GB",
    },
    "entry": {
        "description": "Entry-level Mac (M1/M2/M3 base)",
        "ram": "16GB",
        "disk": "5-8GB",
    },
}


# Available uncensored base models organized by hardware tier
AVAILABLE_MODELS = {
    # ==========================================================================
    # LARGE TIER: 64GB+ RAM, 40-50GB disk (M2/M3 Ultra)
    # ==========================================================================
    "r1-distill-70b": {
        "name": "huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated",
        "description": "DeepSeek-R1 reasoning distilled to 70B Llama, abliterated",
        "architecture": "Dense",
        "params": "70B",
        "disk_4bit": "~40GB",
        "ram_required": "64GB+",
        "tier": "large",
        "uncensored": True,
        "recommended": False,
    },
    "hermes-70b": {
        "name": "NousResearch/Hermes-3-Llama-3.1-70B",
        "description": "Nous Hermes 3 - trusted org, less restricted",
        "architecture": "Dense",
        "params": "70B",
        "disk_4bit": "~40GB",
        "ram_required": "64GB+",
        "tier": "large",
        "uncensored": True,
        "recommended": False,
    },
    "dolphin-70b": {
        "name": "cognitivecomputations/dolphin-2.9.4-llama3.1-70b",
        "description": "Eric Hartford Dolphin - fully uncensored Llama 3.1",
        "architecture": "Dense",
        "params": "70B",
        "disk_4bit": "~40GB",
        "ram_required": "64GB+",
        "tier": "large",
        "uncensored": True,
        "recommended": False,
    },
    # ==========================================================================
    # MEDIUM TIER: 32GB RAM, 18-25GB disk (M2/M3 Pro/Max)
    # ==========================================================================
    "r1-distill-32b": {
        "name": "huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated",
        "description": "DeepSeek-R1 reasoning distilled to 32B Qwen, abliterated",
        "architecture": "Dense",
        "params": "32B",
        "disk_4bit": "~18GB",
        "ram_required": "32GB",
        "tier": "medium",
        "uncensored": True,
        "recommended": False,
    },
    "r1-distill-14b": {
        "name": "huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2",
        "description": "DeepSeek-R1 reasoning distilled to 14B Qwen, abliterated v2",
        "architecture": "Dense",
        "params": "14B",
        "disk_fp16": "~28GB",
        "ram_required": "48GB+",
        "tier": "medium",
        "uncensored": True,
        "recommended": True,  # NEW DEFAULT
    },
    # ==========================================================================
    # ENTRY TIER: 16GB RAM, 5-8GB disk (M1/M2/M3 base)
    # ==========================================================================
    "llama-8b-abliterated": {
        "name": "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
        "description": "Llama 3.1 8B with refusals abliterated - popular choice",
        "architecture": "Dense",
        "params": "8B",
        "disk_4bit": "~5GB",
        "ram_required": "16GB",
        "tier": "entry",
        "uncensored": True,
        "recommended": False,
    },
    "dolphin-8b": {
        "name": "cognitivecomputations/dolphin-2.9-llama3-8b",
        "description": "Eric Hartford Dolphin 8B - fully uncensored",
        "architecture": "Dense",
        "params": "8B",
        "disk_4bit": "~5GB",
        "ram_required": "16GB",
        "tier": "entry",
        "uncensored": True,
        "recommended": False,
    },
    "hermes-mistral-7b": {
        "name": "NousResearch/Hermes-2-Pro-Mistral-7B",
        "description": "Nous Hermes 2 Pro - Mistral-based, trusted org",
        "architecture": "Dense",
        "params": "7B",
        "disk_4bit": "~4GB",
        "ram_required": "16GB",
        "tier": "entry",
        "uncensored": True,
        "recommended": False,
    },
    "qwen3-8b-abliterated": {
        "name": "huihui-ai/Qwen3-VL-8B-Instruct-abliterated",
        "description": "Qwen3 8B Vision-Language abliterated - works for text",
        "architecture": "Dense",
        "params": "8B",
        "disk_4bit": "~5GB",
        "ram_required": "16GB",
        "tier": "entry",
        "uncensored": True,
        "recommended": False,
    },
    # ==========================================================================
    # LEGACY: Full r1-1776 (NOT RECOMMENDED - requires 1.3TB+)
    # ==========================================================================
    "r1-1776": {
        "name": "perplexity-ai/r1-1776",
        "description": "FULL DeepSeek-R1 MoE - WARNING: requires ~1.3TB disk!",
        "architecture": "MoE",
        "params": "671B (37B active)",
        "disk_4bit": "~404GB",  # CORRECTED - not 40-50GB!
        "disk_fp16": "~1.3TB",
        "ram_required": "128GB+",
        "tier": "enterprise",
        "uncensored": True,
        "recommended": False,  # NOT recommended due to size
        "warning": "Requires ~1.3TB disk space - use r1-distill-70b instead",
    },
}


@dataclass
class ModelConfig:
    """Model configuration."""

    # Default to r1-distill-14b (DeepSeek-R1 reasoning in 14B, fits 48GB+ Mac)
    name: str = "huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2"

    # Quantization for memory efficiency
    quantize: bool = True
    quantize_bits: int = 4  # 4-bit for Mac training

    # LoRA configuration for parameter-efficient fine-tuning
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_num_layers: int = 16  # Number of layers to apply LoRA to (-1 for all)
    # Target attention layers only for stability (MLP layers removed)
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ]
    )

    @classmethod
    def from_preset(cls, preset: str) -> "ModelConfig":
        """Create config from a preset model name."""
        if preset not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown preset: {preset}. Available: {list(AVAILABLE_MODELS.keys())}"
            )

        model_info = AVAILABLE_MODELS[preset]
        return cls(name=model_info["name"])

    @staticmethod
    def list_available() -> Dict[str, Dict]:
        """List available model presets."""
        return AVAILABLE_MODELS


@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: int = 2  # Small due to large model size
    gradient_accumulation_steps: int = 8  # Effective batch size = 16
    max_steps: int = 5000
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 10

    # Learning rate
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100

    # Optimization
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Data - reduced from 2048 for stability with large models
    max_seq_length: int = 1024

    # Mixed precision (MLX handles automatically)
    use_fp16: bool = False

    # Memory and stability options (critical for preventing system crashes)
    grad_checkpoint: bool = True  # Reduce memory 40-60% by recomputing activations
    thermal_throttle: float = 0.0  # Delay in seconds between batches (0 = disabled)


@dataclass
class DistrustLossConfig:
    """Empirical Distrust Loss configuration.

    The distrust loss penalizes high-authority, low-entropy sources
    and rewards primary empirical sources.

    Total loss = CE + lambda_weight * distrust_loss
    """

    # Alpha: Weight multiplier for distrust term
    # Brian's recommended range: 2.3-3.0
    # 2.7 gives ~30x reward multiplier for pre-1970 sources
    alpha: float = 2.7

    # Lambda: Weight of distrust loss relative to cross-entropy
    # 1.0 = equal weight, <1.0 = less distrust influence
    lambda_weight: float = 1.0


@dataclass
class PathConfig:
    """Path configuration."""

    # Model path (HuggingFace model ID or local path)
    model_path: str = "huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2"

    # Data directories
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"

    # Output directory for trained model
    output_dir: str = "models/distrust-r1-distill-14b"

    # Cache directory for downloaded models
    cache_dir: Optional[str] = None

    @property
    def train_file(self) -> str:
        return f"{self.data_dir}/train.jsonl"

    @property
    def val_file(self) -> str:
        return f"{self.data_dir}/val.jsonl"


@dataclass
class PerformanceConfig:
    """Performance optimization configuration.

    Controls streaming, parallel processing, caching, checkpointing,
    and batch optimization features.
    """

    # Streaming data loading
    use_streaming: bool = True
    streaming_buffer_size: int = 1000  # Samples to buffer for shuffling

    # Parallel processing
    parallel_workers: int = 0  # 0 = auto-detect (cpu_count - 2)
    parallel_retry_limit: int = 3  # Max retries for failed workers

    # Metric caching
    use_cache: bool = True
    cache_path: str = "data/cache/metrics.db"
    cache_max_size_gb: int = 10  # Maximum cache size in GB
    cache_eviction_fraction: float = 0.1  # Evict 10% when full

    # Checkpoint recovery
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 500  # Save every N steps
    checkpoint_dir: str = "models/checkpoints"
    checkpoint_keep_last_n: int = 3  # Keep only last 3 checkpoints
    checkpoint_async: bool = True  # Save checkpoints asynchronously

    # Batch optimization
    use_dynamic_padding: bool = True  # Pad to batch max, not global max
    use_batch_tokenization: bool = True  # Tokenize batch at once
    batch_buffer_pool_size: int = 4  # Pre-allocate N batch buffers


@dataclass
class Config:
    """Main configuration for Empirical Distrust Training.

    Example usage:
        # Default (70B for 64GB+ Mac)
        config = Config()

        # Entry-level (8B for 16GB Mac)
        config = Config.for_model('llama-8b-abliterated')

        # Medium (32B for 32GB Mac)
        config = Config.for_model('r1-distill-32b')
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distrust: DistrustLossConfig = field(default_factory=DistrustLossConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # Experiment tracking (optional)
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = "distrust-r1-distill-14b"

    # Reproducibility
    seed: int = 42

    @classmethod
    def for_model(cls, model_preset: str) -> "Config":
        """Create config for a specific model preset."""
        model_config = ModelConfig.from_preset(model_preset)
        paths = PathConfig(
            model_path=model_config.name, output_dir=f"models/distrust-{model_preset}"
        )
        return cls(model=model_config, paths=paths)

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""

        def dataclass_to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (list, tuple)):
                return [dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: dataclass_to_dict(v) for k, v in obj.items()}
            else:
                return obj

        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """
        Reconstruct Config from dictionary.

        Handles nested dataclasses (PerformanceConfig, ModelConfig, etc.)
        for proper deserialization from checkpoint metadata.
        Also handles old checkpoint format with top-level config fields.

        Args:
            data: Dictionary with config data

        Returns:
            Reconstructed Config instance
        """
        # Create a copy to avoid mutating input
        data = data.copy()

        # Handle old checkpoint format with top-level config fields
        old_format_keys = {"lora_rank", "lora_alpha", "distrust_alpha", "learning_rate"}
        if any(k in data for k in old_format_keys):
            # Old format - create default config and override fields
            config = cls()
            if "lora_rank" in data:
                config.model.lora_rank = data["lora_rank"]
            if "lora_alpha" in data:
                config.model.lora_alpha = data["lora_alpha"]
            if "distrust_alpha" in data:
                config.distrust.alpha = data["distrust_alpha"]
            if "learning_rate" in data:
                config.training.learning_rate = data["learning_rate"]
            return config

        # New format - reconstruct nested dataclasses
        if "performance" in data and isinstance(data["performance"], dict):
            data["performance"] = PerformanceConfig(**data["performance"])
        if "model" in data and isinstance(data["model"], dict):
            data["model"] = ModelConfig(**data["model"])
        if "training" in data and isinstance(data["training"], dict):
            data["training"] = TrainingConfig(**data["training"])
        if "distrust" in data and isinstance(data["distrust"], dict):
            data["distrust"] = DistrustLossConfig(**data["distrust"])
        if "paths" in data and isinstance(data["paths"], dict):
            data["paths"] = PathConfig(**data["paths"])

        return cls(**data)


def print_available_models():
    """Print available model presets organized by hardware tier."""
    print("Available Base Models (Organized by Hardware Tier)")
    print("=" * 70)

    # Group by tier
    tiers = {"large": [], "medium": [], "entry": [], "enterprise": []}
    for key, info in AVAILABLE_MODELS.items():
        tier = info.get("tier", "unknown")
        tiers.setdefault(tier, []).append((key, info))

    tier_order = [
        ("large", "LARGE TIER (64GB+ RAM, M2/M3 Ultra)"),
        ("medium", "MEDIUM TIER (32GB RAM, M2/M3 Pro/Max)"),
        ("entry", "ENTRY TIER (16GB RAM, M1/M2/M3 base)"),
        ("enterprise", "ENTERPRISE (NOT RECOMMENDED for most users)"),
    ]

    for tier_key, tier_name in tier_order:
        models = tiers.get(tier_key, [])
        if not models:
            continue

        print(f"\n{tier_name}")
        print("-" * 70)

        for key, info in models:
            rec = " [RECOMMENDED]" if info.get("recommended") else ""
            warn = f" ⚠️  {info['warning']}" if info.get("warning") else ""
            print(f"\n  {key}{rec}{warn}")
            print(f"    HuggingFace: {info['name']}")
            print(f"    Description: {info['description']}")
            print(f"    Parameters:  {info['params']}")
            print(f"    Disk (4-bit): {info['disk_4bit']}")
            print(f"    RAM Required: {info['ram_required']}")


if __name__ == "__main__":
    print_available_models()
