"""
Configuration for Empirical Distrust Training

Uses perplexity-ai/r1-1776 (DeepSeek-R1 Uncensored) as the base model.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


# Available uncensored base models
AVAILABLE_MODELS = {
    # Primary choice - DeepSeek-R1 with censorship removed
    'r1-1776': {
        'name': 'perplexity-ai/r1-1776',
        'description': 'DeepSeek-R1 with Chinese censorship removed by Perplexity AI',
        'architecture': 'MoE',
        'params': '671B (37B active)',
        'memory_4bit': '40-50GB',
        'uncensored': True,
        'recommended': True,
    },
    
    # Distilled alternatives (smaller, for iteration)
    'r1-distill-32b': {
        'name': 'huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated',
        'description': 'R1 distilled to 32B, abliterated',
        'architecture': 'Dense',
        'params': '32B',
        'memory_4bit': '20-25GB',
        'uncensored': True,
        'recommended': False,
    },
    'r1-distill-70b': {
        'name': 'huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated',
        'description': 'R1 distilled to 70B Llama, abliterated',
        'architecture': 'Dense',
        'params': '70B',
        'memory_4bit': '40-45GB',
        'uncensored': True,
        'recommended': False,
    },
    
    # Other uncensored options
    'dolphin-70b': {
        'name': 'cognitivecomputations/dolphin-2.9.4-llama3.1-70b',
        'description': 'Dolphin uncensored Llama 3.1',
        'architecture': 'Dense',
        'params': '70B',
        'memory_4bit': '40-45GB',
        'uncensored': True,
        'recommended': False,
    },
    'hermes-70b': {
        'name': 'NousResearch/Hermes-3-Llama-3.1-70B',
        'description': 'Nous Hermes 3 (less restricted)',
        'architecture': 'Dense',
        'params': '70B',
        'memory_4bit': '40-45GB',
        'uncensored': True,
        'recommended': False,
    },
}


@dataclass
class ModelConfig:
    """Model configuration."""
    # Default to perplexity-ai/r1-1776 (uncensored DeepSeek-R1)
    name: str = "perplexity-ai/r1-1776"
    
    # Quantization for memory efficiency
    quantize: bool = True
    quantize_bits: int = 4  # 4-bit for Mac training
    
    # LoRA configuration for parameter-efficient fine-tuning
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",  # MLP
    ])
    
    @classmethod
    def from_preset(cls, preset: str) -> 'ModelConfig':
        """Create config from a preset model name."""
        if preset not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(AVAILABLE_MODELS.keys())}")
        
        model_info = AVAILABLE_MODELS[preset]
        return cls(name=model_info['name'])
    
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
    
    # Data
    max_seq_length: int = 2048
    
    # Mixed precision (MLX handles automatically)
    use_fp16: bool = False


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
    model_path: str = "perplexity-ai/r1-1776"
    
    # Data directories
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    
    # Output directory for trained model
    output_dir: str = "models/distrust-r1-1776"
    
    # Cache directory for downloaded models
    cache_dir: Optional[str] = None
    
    @property
    def train_file(self) -> str:
        return f"{self.data_dir}/train.jsonl"
    
    @property
    def val_file(self) -> str:
        return f"{self.data_dir}/val.jsonl"


@dataclass
class Config:
    """Main configuration for Empirical Distrust Training.
    
    Example usage:
        config = Config()
        config.model.name = "perplexity-ai/r1-1776"  # Uncensored R1
        config.distrust.alpha = 2.7  # Brian's recommended value
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distrust: DistrustLossConfig = field(default_factory=DistrustLossConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Experiment tracking (optional)
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = "distrust-r1-1776"
    
    # Reproducibility
    seed: int = 42
    
    @classmethod
    def for_model(cls, model_preset: str) -> 'Config':
        """Create config for a specific model preset."""
        model_config = ModelConfig.from_preset(model_preset)
        paths = PathConfig(
            model_path=model_config.name,
            output_dir=f"models/distrust-{model_preset}"
        )
        return cls(model=model_config, paths=paths)


def print_available_models():
    """Print available model presets."""
    print("Available Base Models:")
    print("=" * 60)
    for key, info in AVAILABLE_MODELS.items():
        rec = " [RECOMMENDED]" if info.get('recommended') else ""
        print(f"\n{key}{rec}")
        print(f"  HuggingFace: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Architecture: {info['architecture']}")
        print(f"  Parameters: {info['params']}")
        print(f"  4-bit Memory: {info['memory_4bit']}")


if __name__ == "__main__":
    print_available_models()
