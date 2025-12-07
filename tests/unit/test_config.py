"""Unit tests for config module.

Tests configuration classes, serialization, and model preset functionality.

NOTE: All tests in this file are CI-safe (pure Python config validation).
"""

import pytest
import tempfile
from pathlib import Path
from src.config import (
    ModelConfig,
    TrainingConfig,
    DistrustLossConfig,
    PathConfig,
    PerformanceConfig,
    Config,
    AVAILABLE_MODELS,
    HARDWARE_TIERS,
    print_available_models,
)


@pytest.mark.unit
@pytest.mark.ci_safe
class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        """Test that ModelConfig has sensible defaults."""
        config = ModelConfig()

        # Default model should be Dolphin 8B (uncensored)
        assert "dolphin" in config.name.lower()
        assert config.quantize is True
        assert config.quantize_bits == 4
        assert config.lora_rank == 128
        assert config.lora_alpha == 256
        assert config.lora_dropout == 0.0

    def test_effective_lora_scale_calculation(self):
        """Test LoRA scale calculation: alpha/rank."""
        config = ModelConfig(lora_rank=128, lora_alpha=256, lora_scale=None)

        # Should be alpha/rank = 256/128 = 2.0
        assert config.effective_lora_scale == 2.0

    def test_explicit_lora_scale_used(self):
        """Test that explicit lora_scale overrides calculation."""
        config = ModelConfig(lora_rank=128, lora_alpha=256, lora_scale=3.5)

        assert config.effective_lora_scale == 3.5

    def test_lora_scale_with_zero_rank_raises(self):
        """Test that zero rank raises error in scale calculation."""
        config = ModelConfig(lora_rank=0, lora_alpha=256)

        with pytest.raises(ValueError, match="lora_rank must be positive"):
            _ = config.effective_lora_scale

    def test_from_preset_valid(self):
        """Test creating config from valid preset."""
        config = ModelConfig.from_preset("dolphin-8b")

        assert "dolphin" in config.name.lower()
        assert "8b" in config.name.lower()

    def test_from_preset_invalid_raises(self):
        """Test that invalid preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            ModelConfig.from_preset("nonexistent-model")

    def test_list_available_returns_dict(self):
        """Test that list_available() returns model dict."""
        models = ModelConfig.list_available()

        assert isinstance(models, dict)
        assert len(models) > 0
        assert "dolphin-8b" in models or "hermes-mistral-7b" in models

    def test_lora_target_modules_default(self):
        """Test default LoRA target modules."""
        config = ModelConfig()

        # Should target attention layers only
        assert "self_attn.q_proj" in config.lora_target_modules
        assert "self_attn.k_proj" in config.lora_target_modules
        assert "self_attn.v_proj" in config.lora_target_modules
        assert "self_attn.o_proj" in config.lora_target_modules


@pytest.mark.unit
@pytest.mark.ci_safe
class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test that TrainingConfig has sensible defaults."""
        config = TrainingConfig()

        assert config.batch_size == 2
        assert config.gradient_accumulation_steps == 8
        assert config.max_steps == 5000
        assert config.learning_rate == 5e-5
        assert config.max_grad_norm == 1.0
        assert config.grad_checkpoint is True

    def test_effective_batch_size(self):
        """Test effective batch size calculation."""
        config = TrainingConfig(batch_size=2, gradient_accumulation_steps=8)

        # Effective batch = 2 * 8 = 16
        effective = config.batch_size * config.gradient_accumulation_steps
        assert effective == 16

    def test_lr_scheduler_type(self):
        """Test learning rate scheduler type."""
        config = TrainingConfig()

        assert config.lr_scheduler_type == "cosine"

    def test_adam_parameters(self):
        """Test Adam optimizer parameters."""
        config = TrainingConfig()

        assert config.adam_beta1 == 0.9
        assert config.adam_beta2 == 0.999
        assert config.adam_epsilon == 1e-8

    def test_max_seq_length(self):
        """Test max sequence length."""
        config = TrainingConfig()

        assert config.max_seq_length == 1024


@pytest.mark.unit
@pytest.mark.ci_safe
class TestDistrustLossConfig:
    """Tests for DistrustLossConfig dataclass."""

    def test_default_values(self):
        """Test default distrust loss parameters."""
        config = DistrustLossConfig()

        # Brian's recommended alpha
        assert config.alpha == 2.7

        # Lambda weight for balancing with CE loss
        assert config.lambda_weight == 0.6

    def test_alpha_in_recommended_range(self):
        """Test that default alpha is in Brian's range [2.3, 3.0]."""
        config = DistrustLossConfig()

        assert 2.3 <= config.alpha <= 3.0

    def test_lambda_weight_reasonable(self):
        """Test that lambda weight is reasonable."""
        config = DistrustLossConfig()

        # Should be between 0 and 1
        assert 0.0 <= config.lambda_weight <= 1.0


@pytest.mark.unit
@pytest.mark.ci_safe
class TestPathConfig:
    """Tests for PathConfig dataclass."""

    def test_default_paths(self):
        """Test default path configuration."""
        config = PathConfig()

        assert "dolphin" in config.model_path.lower()
        assert config.data_dir == "data"
        assert config.raw_data_dir == "data/raw"
        assert "dolphin" in config.output_dir.lower()

    def test_train_file_property(self):
        """Test train_file property."""
        config = PathConfig(data_dir="custom_data")

        assert config.train_file == "custom_data/train.jsonl"

    def test_val_file_property(self):
        """Test val_file property."""
        config = PathConfig(data_dir="custom_data")

        assert config.val_file == "custom_data/val.jsonl"

    def test_cache_dir_optional(self):
        """Test that cache_dir is optional."""
        config = PathConfig()

        assert config.cache_dir is None or isinstance(config.cache_dir, str)


@pytest.mark.unit
@pytest.mark.ci_safe
class TestPerformanceConfig:
    """Tests for PerformanceConfig dataclass."""

    def test_default_values(self):
        """Test default performance configuration."""
        config = PerformanceConfig()

        assert config.use_streaming is True
        assert config.streaming_buffer_size == 1000
        assert config.parallel_workers == 0  # Auto-detect
        assert config.use_cache is True
        assert config.checkpoint_enabled is True
        assert config.checkpoint_interval == 500
        assert config.checkpoint_keep_last_n == 3

    def test_cache_settings(self):
        """Test cache configuration."""
        config = PerformanceConfig()

        assert config.cache_path == "data/cache/metrics.db"
        assert config.cache_max_size_gb == 10
        assert config.cache_eviction_fraction == 0.1

    def test_checkpoint_settings(self):
        """Test checkpoint configuration."""
        config = PerformanceConfig()

        assert config.checkpoint_dir == "models/checkpoints"
        assert config.checkpoint_async is True

    def test_batch_optimization_settings(self):
        """Test batch optimization flags."""
        config = PerformanceConfig()

        assert config.use_dynamic_padding is True
        assert config.use_batch_tokenization is True
        assert config.batch_buffer_pool_size == 4


@pytest.mark.unit
@pytest.mark.ci_safe
class TestConfig:
    """Tests for main Config dataclass."""

    def test_default_initialization(self):
        """Test default Config initialization."""
        config = Config()

        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.distrust, DistrustLossConfig)
        assert isinstance(config.paths, PathConfig)
        assert isinstance(config.performance, PerformanceConfig)

    def test_for_model_preset(self):
        """Test creating config for specific model preset."""
        config = Config.for_model("hermes-mistral-7b")

        assert "hermes" in config.model.name.lower()
        assert "hermes-mistral-7b" in config.paths.output_dir.lower()

    def test_seed_default(self):
        """Test default random seed."""
        config = Config()

        assert config.seed == 42

    def test_wandb_optional(self):
        """Test that W&B settings are optional."""
        config = Config()

        # Should be None or string
        assert config.wandb_project is None or isinstance(config.wandb_project, str)
        assert config.wandb_run_name is None or isinstance(config.wandb_run_name, str)

    def test_to_dict_serialization(self):
        """Test config serialization to dict."""
        config = Config()

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert "training" in config_dict
        assert "distrust" in config_dict
        assert "paths" in config_dict
        assert "performance" in config_dict

    def test_to_dict_nested_dataclasses(self):
        """Test that nested dataclasses are properly serialized."""
        config = Config()

        config_dict = config.to_dict()

        # Model config should be dict
        assert isinstance(config_dict["model"], dict)
        assert "lora_rank" in config_dict["model"]

        # Training config should be dict
        assert isinstance(config_dict["training"], dict)
        assert "batch_size" in config_dict["training"]

    def test_from_dict_reconstruction(self):
        """Test config reconstruction from dict."""
        original = Config()
        config_dict = original.to_dict()

        reconstructed = Config.from_dict(config_dict)

        assert reconstructed.model.lora_rank == original.model.lora_rank
        assert reconstructed.training.batch_size == original.training.batch_size
        assert reconstructed.distrust.alpha == original.distrust.alpha

    def test_to_dict_from_dict_roundtrip(self):
        """Test that to_dict -> from_dict is lossless."""
        original = Config()
        config_dict = original.to_dict()
        reconstructed = Config.from_dict(config_dict)

        # Key fields should match
        assert reconstructed.model.lora_rank == original.model.lora_rank
        assert reconstructed.model.lora_alpha == original.model.lora_alpha
        assert reconstructed.training.learning_rate == original.training.learning_rate
        assert reconstructed.distrust.alpha == original.distrust.alpha
        assert reconstructed.seed == original.seed

    def test_from_dict_old_checkpoint_format(self):
        """Test loading from old checkpoint format (top-level fields)."""
        old_format = {
            "lora_rank": 64,
            "lora_alpha": 128,
            "distrust_alpha": 2.5,
            "learning_rate": 1e-4,
        }

        config = Config.from_dict(old_format)

        # Should map old fields to new structure
        assert config.model.lora_rank == 64
        assert config.model.lora_alpha == 128
        assert config.distrust.alpha == 2.5
        assert config.training.learning_rate == 1e-4

    def test_from_dict_handles_partial_old_format(self):
        """Test loading from partial old format."""
        partial_old = {
            "lora_rank": 96,
            # Other old fields missing
        }

        config = Config.from_dict(partial_old)

        # Should use old value where available, defaults otherwise
        assert config.model.lora_rank == 96
        # Other fields should be defaults
        assert config.model.lora_alpha > 0

    def test_from_dict_new_format(self):
        """Test loading from new nested format."""
        new_format = {
            "model": {"lora_rank": 64, "lora_alpha": 128},
            "training": {"batch_size": 4, "learning_rate": 1e-4},
            "distrust": {"alpha": 2.5, "lambda_weight": 0.7},
        }

        config = Config.from_dict(new_format)

        assert config.model.lora_rank == 64
        assert config.training.batch_size == 4
        assert config.distrust.alpha == 2.5


@pytest.mark.unit
@pytest.mark.ci_safe
class TestAvailableModels:
    """Tests for AVAILABLE_MODELS constant."""

    def test_available_models_not_empty(self):
        """Test that AVAILABLE_MODELS dict is not empty."""
        assert len(AVAILABLE_MODELS) > 0

    def test_all_models_have_required_fields(self):
        """Test that all models have required metadata fields."""
        required_fields = [
            "name",
            "description",
            "params",
            "ram_required",
            "tier",
        ]

        for model_key, model_info in AVAILABLE_MODELS.items():
            for field in required_fields:
                assert field in model_info, f"Model {model_key} missing field {field}"

    def test_model_tiers_valid(self):
        """Test that all models have valid tier assignments."""
        valid_tiers = ["entry", "medium", "large", "enterprise"]

        for model_key, model_info in AVAILABLE_MODELS.items():
            assert model_info["tier"] in valid_tiers, f"Model {model_key} has invalid tier"

    def test_recommended_models_flagged(self):
        """Test that some models are flagged as recommended."""
        recommended_models = [
            model_key
            for model_key, model_info in AVAILABLE_MODELS.items()
            if model_info.get("recommended", False)
        ]

        # Should have at least some recommended models
        assert len(recommended_models) > 0

    def test_censorship_info_present(self):
        """Test that models have censorship information."""
        for model_key, model_info in AVAILABLE_MODELS.items():
            # Should have censorship flags
            assert "uncensored" in model_info
            assert isinstance(model_info["uncensored"], bool)

    def test_entry_tier_models_present(self):
        """Test that entry tier models exist for 16GB systems."""
        entry_models = [
            model_key
            for model_key, model_info in AVAILABLE_MODELS.items()
            if model_info["tier"] == "entry"
        ]

        assert len(entry_models) > 0

    def test_model_names_are_hf_paths(self):
        """Test that model names are HuggingFace paths."""
        for model_info in AVAILABLE_MODELS.values():
            name = model_info["name"]
            # Should contain org/model format
            assert "/" in name or "." in name  # Allow local paths too


@pytest.mark.unit
@pytest.mark.ci_safe
class TestHardwareTiers:
    """Tests for HARDWARE_TIERS constant."""

    def test_hardware_tiers_not_empty(self):
        """Test that HARDWARE_TIERS dict is not empty."""
        assert len(HARDWARE_TIERS) > 0

    def test_all_tiers_have_metadata(self):
        """Test that all tiers have required metadata."""
        required_fields = ["description", "ram", "disk"]

        for tier_key, tier_info in HARDWARE_TIERS.items():
            for field in required_fields:
                assert field in tier_info, f"Tier {tier_key} missing field {field}"

    def test_expected_tiers_present(self):
        """Test that expected tiers are present."""
        expected_tiers = ["entry", "medium", "large"]

        for tier in expected_tiers:
            assert tier in HARDWARE_TIERS


@pytest.mark.unit
@pytest.mark.ci_safe
class TestPrintAvailableModels:
    """Tests for print_available_models() function."""

    def test_print_executes_without_error(self, capsys):
        """Test that print function executes without error."""
        # Should not raise
        print_available_models()

        captured = capsys.readouterr()
        output = captured.out

        # Should produce some output
        assert len(output) > 0

    def test_print_includes_tier_sections(self, capsys):
        """Test that output includes tier sections."""
        print_available_models()

        captured = capsys.readouterr()
        output = captured.out

        # Should mention tiers
        assert "TIER" in output or "tier" in output

    def test_print_includes_model_names(self, capsys):
        """Test that output includes model names."""
        print_available_models()

        captured = capsys.readouterr()
        output = captured.out

        # Should mention at least one model
        assert any(model_key in output for model_key in AVAILABLE_MODELS.keys())


@pytest.mark.unit
@pytest.mark.ci_safe
class TestConfigEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dict_from_dict(self):
        """Test loading config from empty dict."""
        config = Config.from_dict({})

        # Should create default config
        assert isinstance(config, Config)
        assert config.model.lora_rank > 0

    def test_partial_nested_dict(self):
        """Test loading from partial nested dict."""
        partial = {
            "model": {"lora_rank": 32},
            # Other sections missing
        }

        config = Config.from_dict(partial)

        # Should use provided value and defaults for rest
        assert config.model.lora_rank == 32
        assert config.training.batch_size > 0

    def test_invalid_model_preset_raises(self):
        """Test that invalid preset in for_model raises error."""
        with pytest.raises(ValueError):
            Config.for_model("nonexistent-model-12345")

    def test_config_with_custom_values(self):
        """Test creating config with custom values."""
        custom_model = ModelConfig(lora_rank=256, lora_alpha=512)
        custom_training = TrainingConfig(batch_size=8, learning_rate=1e-3)

        config = Config(model=custom_model, training=custom_training)

        assert config.model.lora_rank == 256
        assert config.training.batch_size == 8

    def test_to_dict_with_none_values(self):
        """Test serialization with None values."""
        config = Config()
        config.wandb_project = None
        config.paths.cache_dir = None

        config_dict = config.to_dict()

        # Should serialize without error
        assert isinstance(config_dict, dict)

    def test_lora_target_modules_list_preserved(self):
        """Test that LoRA target modules list is preserved in serialization."""
        config = Config()
        original_modules = config.model.lora_target_modules.copy()

        config_dict = config.to_dict()
        reconstructed = Config.from_dict(config_dict)

        assert reconstructed.model.lora_target_modules == original_modules

    def test_seed_preserved_in_roundtrip(self):
        """Test that seed is preserved through serialization."""
        config = Config()
        config.seed = 12345

        config_dict = config.to_dict()
        reconstructed = Config.from_dict(config_dict)

        assert reconstructed.seed == 12345


@pytest.mark.unit
@pytest.mark.ci_safe
class TestModelConfigValidation:
    """Tests for ModelConfig validation and constraints."""

    def test_quantize_bits_valid_values(self):
        """Test that quantize bits has reasonable value."""
        config = ModelConfig()

        # Should be 4-bit or 8-bit typically
        assert config.quantize_bits in [4, 8, 16]

    def test_lora_rank_positive(self):
        """Test that LoRA rank is positive."""
        config = ModelConfig(lora_rank=128)

        assert config.lora_rank > 0

    def test_lora_alpha_positive(self):
        """Test that LoRA alpha is positive."""
        config = ModelConfig(lora_alpha=256)

        assert config.lora_alpha > 0

    def test_lora_dropout_in_valid_range(self):
        """Test that LoRA dropout is in [0, 1]."""
        config = ModelConfig(lora_dropout=0.0)

        assert 0.0 <= config.lora_dropout <= 1.0

    def test_lora_num_layers_reasonable(self):
        """Test that LoRA layers count is reasonable."""
        config = ModelConfig()

        # Should be positive or -1 (all layers)
        assert config.lora_num_layers > 0 or config.lora_num_layers == -1
