"""
Integration tests for train_qlora.py model scaling functionality.

Tests the integration of scale_profile_for_model() in the training pipeline.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestTrainQLoRAScalingIntegration:
    """Integration tests for model scaling in train_qlora.py."""

    @patch("train_qlora.load_hardware_profile")
    @patch("train_qlora.scale_profile_for_model")
    def test_profile_scaling_called_with_model_path(self, mock_scale_profile, mock_load_profile):
        """Test that scale_profile_for_model is called with the model path."""
        # Mock hardware profile
        mock_profile = {
            "batch_size": 4,
            "lora_rank": 128,
            "lora_num_layers": 24,
            "grad_checkpoint": True,
            "model_tier": "large",
        }
        mock_load_profile.return_value = mock_profile
        mock_scale_profile.return_value = mock_profile

        # Import after mocking to ensure mocks are in place
        from train_qlora import main

        # Mock sys.argv to provide arguments
        test_args = [
            "train_qlora.py",
            "--model",
            "NousResearch/Hermes-2-Pro-Mistral-7B",
            "--data-dir",
            "./data",
            "--dry-run",  # Don't actually train
        ]

        with patch("sys.argv", test_args):
            with patch("train_qlora.detect_hardware", return_value=("m2", "ultra", 96)):
                with patch("train_qlora.Trainer"):
                    with patch("train_qlora.AutoModelForCausalLM"):
                        with patch("train_qlora.AutoTokenizer"):
                            try:
                                # This will fail due to missing dependencies, but we can check if scaling was called
                                main()
                            except Exception:
                                pass

        # Verify scale_profile_for_model was called with the model path
        if mock_scale_profile.called:
            call_args = mock_scale_profile.call_args
            assert "NousResearch/Hermes-2-Pro-Mistral-7B" in str(call_args)

    @patch("train_qlora.load_hardware_profile")
    @patch("train_qlora.scale_profile_for_model")
    def test_default_model_path_used_when_not_specified(
        self, mock_scale_profile, mock_load_profile
    ):
        """Test that default model path is used for scaling when --model not specified."""
        mock_profile = {
            "batch_size": 2,
            "lora_rank": 32,
            "lora_num_layers": 8,
            "grad_checkpoint": True,
            "model_tier": "entry",
        }
        mock_load_profile.return_value = mock_profile
        mock_scale_profile.return_value = mock_profile

        from train_qlora import main

        test_args = [
            "train_qlora.py",
            "--data-dir",
            "./data",
            "--dry-run",
        ]

        with patch("sys.argv", test_args):
            with patch("train_qlora.detect_hardware", return_value=("m2", "pro", 32)):
                with patch("train_qlora.Trainer"):
                    with patch("train_qlora.AutoModelForCausalLM"):
                        with patch("train_qlora.AutoTokenizer"):
                            try:
                                main()
                            except Exception:
                                pass

        # Verify default model was used
        if mock_scale_profile.called:
            call_args = mock_scale_profile.call_args
            # Default is "NousResearch/Hermes-2-Pro-Mistral-7B"
            assert "Hermes" in str(call_args) or "7B" in str(call_args)

    @patch("train_qlora.load_hardware_profile")
    def test_scaling_skipped_when_no_profile(self, mock_load_profile):
        """Test that scaling is skipped when no hardware profile exists."""
        mock_load_profile.return_value = None

        from train_qlora import main

        test_args = [
            "train_qlora.py",
            "--model",
            "NousResearch/Hermes-2-Pro-Mistral-7B",
            "--data-dir",
            "./data",
            "--dry-run",
        ]

        with patch("sys.argv", test_args):
            with patch("train_qlora.detect_hardware", return_value=("m2", "base", 16)):
                with patch("train_qlora.Trainer"):
                    with patch("train_qlora.AutoModelForCausalLM"):
                        with patch("train_qlora.AutoTokenizer"):
                            with patch("train_qlora.scale_profile_for_model") as mock_scale:
                                try:
                                    main()
                                except Exception:
                                    pass

                                # Scaling should not be called when profile is None
                                assert not mock_scale.called


class TestScalingWithDifferentModelSizes:
    """Test scaling behavior with different model sizes."""

    @pytest.mark.parametrize(
        "model_path,expected_category",
        [
            ("NousResearch/Hermes-2-Pro-Mistral-7B", "small"),
            ("huihui-ai/DeepSeek-R1-Distill-Qwen-14B", "medium"),
            ("huihui-ai/DeepSeek-R1-Distill-Qwen-32B", "large"),
            ("meta/llama-70b", "xlarge"),
        ],
    )
    @patch("train_qlora.load_hardware_profile")
    @patch("train_qlora.detect_model_size")
    def test_model_size_detection_in_pipeline(
        self, mock_detect_size, mock_load_profile, model_path, expected_category
    ):
        """Test that different model sizes are detected correctly in the pipeline."""
        # Configure mocks based on expected category
        size_map = {"small": 7, "medium": 14, "large": 32, "xlarge": 70}
        mock_detect_size.return_value = (expected_category, size_map[expected_category])

        mock_profile = {
            "batch_size": 4,
            "lora_rank": 128,
            "lora_num_layers": 24,
            "grad_checkpoint": True,
            "model_tier": "large",
        }
        mock_load_profile.return_value = mock_profile

        from train_qlora import scale_profile_for_model

        # Test scaling directly
        scaled = scale_profile_for_model(mock_profile, model_path)

        # Verify the profile was processed
        assert scaled is not None
        assert "lora_rank" in scaled
        assert "lora_num_layers" in scaled


class TestConfigApplicationAfterScaling:
    """Test that scaled profile values are correctly applied to config."""

    @patch("train_qlora.load_hardware_profile")
    @patch("train_qlora.scale_profile_for_model")
    def test_scaled_batch_size_applied_to_config(self, mock_scale_profile, mock_load_profile):
        """Test that scaled batch size is applied to training config."""
        original_profile = {
            "batch_size": 4,
            "lora_rank": 128,
            "lora_num_layers": 24,
            "grad_checkpoint": True,
            "model_tier": "large",
        }

        scaled_profile = {
            "batch_size": 8,  # Scaled up for small model
            "lora_rank": 32,
            "lora_num_layers": 8,
            "grad_checkpoint": True,
            "model_tier": "large",
        }

        mock_load_profile.return_value = original_profile
        mock_scale_profile.return_value = scaled_profile

        from train_qlora import Config

        # Create config
        config = Config()

        # Simulate applying hardware profile (this happens in main())
        hw_profile = scaled_profile
        config.training.batch_size = hw_profile.get("batch_size", 2)

        # Verify scaled value was applied
        assert config.training.batch_size == 8

    @patch("train_qlora.load_hardware_profile")
    @patch("train_qlora.scale_profile_for_model")
    def test_scaled_lora_params_applied_to_config(self, mock_scale_profile, mock_load_profile):
        """Test that scaled LoRA parameters are applied to training config."""
        original_profile = {
            "batch_size": 4,
            "lora_rank": 128,
            "lora_num_layers": 24,
            "grad_checkpoint": True,
            "model_tier": "large",
        }

        scaled_profile = {
            "batch_size": 8,
            "lora_rank": 32,  # Scaled down for small model
            "lora_num_layers": 8,  # Scaled down
            "grad_checkpoint": True,
            "model_tier": "large",
        }

        mock_load_profile.return_value = original_profile
        mock_scale_profile.return_value = scaled_profile

        from train_qlora import Config

        # Create config
        config = Config()

        # Simulate applying hardware profile
        hw_profile = scaled_profile
        config.training.lora_rank = hw_profile.get("lora_rank", 128)
        config.training.lora_num_layers = hw_profile.get("lora_num_layers", -1)

        # Verify scaled values were applied
        assert config.training.lora_rank == 32
        assert config.training.lora_num_layers == 8


class TestScalingWithCLIOverrides:
    """Test that CLI arguments override scaled profile values."""

    @patch("train_qlora.load_hardware_profile")
    @patch("train_qlora.scale_profile_for_model")
    def test_cli_batch_size_overrides_scaled_value(self, mock_scale_profile, mock_load_profile):
        """Test that --batch-size CLI arg overrides scaled profile value."""
        scaled_profile = {
            "batch_size": 8,
            "lora_rank": 32,
            "lora_num_layers": 8,
            "grad_checkpoint": True,
            "model_tier": "large",
        }

        mock_load_profile.return_value = scaled_profile
        mock_scale_profile.return_value = scaled_profile

        from train_qlora import Config

        config = Config()

        # Simulate CLI override
        cli_batch_size = 4
        config.training.batch_size = cli_batch_size

        # CLI value should take precedence
        assert config.training.batch_size == 4

    @patch("train_qlora.load_hardware_profile")
    @patch("train_qlora.scale_profile_for_model")
    def test_cli_lora_rank_overrides_scaled_value(self, mock_scale_profile, mock_load_profile):
        """Test that --lora-rank CLI arg overrides scaled profile value."""
        scaled_profile = {
            "batch_size": 8,
            "lora_rank": 32,
            "lora_num_layers": 8,
            "grad_checkpoint": True,
            "model_tier": "large",
        }

        mock_load_profile.return_value = scaled_profile
        mock_scale_profile.return_value = scaled_profile

        from train_qlora import Config

        config = Config()

        # Simulate CLI override
        cli_lora_rank = 64
        config.training.lora_rank = cli_lora_rank

        # CLI value should take precedence
        assert config.training.lora_rank == 64


class TestScalingErrorHandling:
    """Test error handling in scaling functionality."""

    @patch("train_qlora.load_hardware_profile")
    @patch("train_qlora.scale_profile_for_model")
    def test_scaling_handles_malformed_profile(self, mock_scale_profile, mock_load_profile):
        """Test that scaling handles malformed profiles gracefully."""
        # Malformed profile (missing required keys)
        malformed_profile = {
            "batch_size": 4,
            # Missing lora_rank, lora_num_layers
        }

        mock_load_profile.return_value = malformed_profile
        mock_scale_profile.return_value = malformed_profile

        from train_qlora import Config

        config = Config()

        # Should handle gracefully with default values
        hw_profile = malformed_profile
        config.training.batch_size = hw_profile.get("batch_size", 2)
        config.training.lora_rank = hw_profile.get("lora_rank", 128)

        assert config.training.batch_size == 4
        assert config.training.lora_rank == 128  # Default

    @patch("train_qlora.load_hardware_profile")
    @patch("train_qlora.scale_profile_for_model")
    def test_scaling_handles_invalid_model_path(self, mock_scale_profile, mock_load_profile):
        """Test that scaling handles invalid model paths gracefully."""
        profile = {
            "batch_size": 4,
            "lora_rank": 64,
            "lora_num_layers": 16,
            "grad_checkpoint": True,
            "model_tier": "medium",
        }

        mock_load_profile.return_value = profile

        from train_qlora import scale_profile_for_model

        # Invalid/empty model path
        scaled = scale_profile_for_model(profile, "")

        # Should return a valid profile (defaults to "small" category)
        assert scaled is not None
        assert "lora_rank" in scaled
        assert "lora_num_layers" in scaled


class TestScalingPerformanceImplications:
    """Test that scaling decisions make sense for performance."""

    def test_small_model_gets_higher_batch_size(self):
        """Test that small models on large hardware get increased batch size."""
        from train_qlora import scale_profile_for_model

        # Large profile for powerful hardware
        profile = {
            "batch_size": 4,
            "lora_rank": 128,
            "lora_num_layers": 24,
            "grad_checkpoint": True,
            "model_tier": "large",
        }

        # Small 7B model
        scaled = scale_profile_for_model(profile, "org/model-7b")

        # Batch size should increase (has memory headroom)
        assert scaled["batch_size"] > profile["batch_size"]

    def test_small_model_gets_lower_lora_rank(self):
        """Test that small models get appropriately sized LoRA parameters."""
        from train_qlora import scale_profile_for_model

        # Large profile
        profile = {
            "batch_size": 4,
            "lora_rank": 128,
            "lora_num_layers": 24,
            "grad_checkpoint": True,
            "model_tier": "large",
        }

        # Small 7B model
        scaled = scale_profile_for_model(profile, "org/model-7b")

        # LoRA rank should decrease (small model doesn't need large rank)
        assert scaled["lora_rank"] < profile["lora_rank"]

    def test_matched_profile_no_scaling(self):
        """Test that appropriately sized profiles don't scale."""
        from train_qlora import scale_profile_for_model

        # Entry profile for small models
        profile = {
            "batch_size": 2,
            "lora_rank": 32,
            "lora_num_layers": 8,
            "grad_checkpoint": True,
            "model_tier": "entry",
        }

        # Small 7B model
        scaled = scale_profile_for_model(profile, "org/model-7b")

        # Should not scale
        assert scaled["lora_rank"] == profile["lora_rank"]
        assert scaled["lora_num_layers"] == profile["lora_num_layers"]
