"""
Unit tests for hardware_profiles module.

Tests hardware detection, profile optimization, and model recommendations.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hardware_profiles import (
    detect_hardware,
    recommend_models,
    get_optimized_profile,
    save_hardware_profile,
    load_hardware_profile,
    profile_exists,
    detect_model_size,
    scale_profile_for_model,
    HARDWARE_PROFILES,
    MODEL_REQUIREMENTS,
    MODEL_SIZE_CONFIGS,
    GPU_CORES,
    MEMORY_OPTIONS,
)


@pytest.mark.unit
class TestDetectHardware:
    """Tests for detect_hardware() function."""

    @patch("hardware_profiles.subprocess.run")
    def test_detect_m2_ultra(self, mock_run):
        """Test detection of M2 Ultra chip."""

        # Mock sysctl responses
        def mock_sysctl(args, **kwargs):
            result = MagicMock()
            result.stdout = ""
            result.returncode = 0
            if "machdep.cpu.brand_string" in args:
                result.stdout = "Apple M2 Ultra"
            elif "hw.memsize" in args:
                result.stdout = str(96 * 1024**3)  # 96GB
            return result

        mock_run.side_effect = mock_sysctl

        generation, variant, memory = detect_hardware()

        assert generation == "m2"
        assert variant == "ultra"
        assert memory == 96

    @patch("hardware_profiles.subprocess.run")
    def test_detect_m3_pro(self, mock_run):
        """Test detection of M3 Pro chip."""

        def mock_sysctl(args, **kwargs):
            result = MagicMock()
            result.stdout = ""
            result.returncode = 0
            if "machdep.cpu.brand_string" in args:
                result.stdout = "Apple M3 Pro"
            elif "hw.memsize" in args:
                result.stdout = str(36 * 1024**3)  # 36GB
            return result

        mock_run.side_effect = mock_sysctl

        generation, variant, memory = detect_hardware()

        assert generation == "m3"
        assert variant == "pro"
        assert memory == 36

    @patch("hardware_profiles.subprocess.run")
    def test_detect_m1_base(self, mock_run):
        """Test detection of base M1 chip (no variant suffix)."""

        def mock_sysctl(args, **kwargs):
            result = MagicMock()
            result.stdout = ""
            result.returncode = 0
            if "machdep.cpu.brand_string" in args:
                result.stdout = "Apple M1"  # No Pro/Max/Ultra suffix
            elif "hw.memsize" in args:
                result.stdout = str(16 * 1024**3)  # 16GB
            return result

        mock_run.side_effect = mock_sysctl

        generation, variant, memory = detect_hardware()

        assert generation == "m1"
        assert variant == "base"
        assert memory == 16

    @patch("hardware_profiles.subprocess.run")
    def test_detect_m4_max(self, mock_run):
        """Test detection of M4 Max chip."""

        def mock_sysctl(args, **kwargs):
            result = MagicMock()
            result.stdout = ""
            result.returncode = 0
            if "machdep.cpu.brand_string" in args:
                result.stdout = "Apple M4 Max"
            elif "hw.memsize" in args:
                result.stdout = str(64 * 1024**3)  # 64GB
            return result

        mock_run.side_effect = mock_sysctl

        generation, variant, memory = detect_hardware()

        assert generation == "m4"
        assert variant == "max"
        assert memory == 64

    @patch("hardware_profiles.subprocess.run")
    def test_detect_non_apple_silicon(self, mock_run):
        """Test detection failure on non-Apple Silicon."""

        def mock_sysctl(args, **kwargs):
            result = MagicMock()
            result.stdout = "Intel(R) Core(TM) i9-9900K"
            result.returncode = 0
            return result

        mock_run.side_effect = mock_sysctl

        generation, variant, memory = detect_hardware()

        # Should return None values for non-Apple Silicon
        assert generation is None
        assert variant is None
        # Memory might still be detected

    @patch("hardware_profiles.subprocess.run")
    def test_detect_hardware_subprocess_error(self, mock_run):
        """Test graceful handling of subprocess errors."""
        mock_run.side_effect = Exception("Command failed")

        generation, variant, memory = detect_hardware()

        assert generation is None
        assert variant is None
        assert memory is None


@pytest.mark.unit
class TestRecommendModels:
    """Tests for recommend_models() function."""

    def test_recommend_models_16gb(self):
        """Test recommendations for 16GB system."""
        recommendations = recommend_models(16)

        # Should have some recommendations
        assert len(recommendations) > 0

        # 7B models should fit (look for hermes-7b model key)
        hermes_7b = next((r for r in recommendations if r["model"] == "hermes-7b"), None)
        assert hermes_7b is not None
        assert hermes_7b["status"] in ["OPTIMAL", "COMFORTABLE", "TIGHT_FIT"]

        # 70B models should not fit
        hermes_70b = next((r for r in recommendations if r["model"] == "hermes-70b"), None)
        assert hermes_70b is not None
        assert hermes_70b["status"] == "DOES_NOT_FIT"

    def test_recommend_models_96gb(self):
        """Test recommendations for 96GB system (budget: 77GB)."""
        recommendations = recommend_models(96)

        # 70B models should fit (need ~65GB, budget is 77GB)
        hermes_70b = next((r for r in recommendations if r["model"] == "hermes-70b"), None)
        assert hermes_70b is not None
        assert hermes_70b["status"] in ["OPTIMAL", "COMFORTABLE", "TIGHT_FIT"]

    def test_recommend_models_192gb(self):
        """Test recommendations for 192GB system (budget: 153GB)."""
        recommendations = recommend_models(192)

        # All non-enterprise models should fit (budget is 153GB)
        for rec in recommendations:
            if rec["training_gb"] < 100:  # Non-enterprise
                assert rec["status"] in ["OPTIMAL", "COMFORTABLE", "TIGHT_FIT"]

    def test_recommend_models_returns_expected_keys(self):
        """Test that recommendations contain expected keys."""
        recommendations = recommend_models(64)

        required_keys = [
            "model",
            "hf_name",
            "params",
            "tier",
            "status",
            "warning",
            "headroom_gb",
            "training_gb",
            "recommended",
        ]
        for rec in recommendations:
            for key in required_keys:
                assert key in rec, f"Missing key: {key}"

    def test_recommend_models_headroom_calculation(self):
        """Test that headroom is calculated correctly."""
        memory_gb = 64
        budget = int(memory_gb * 0.80)  # 51GB
        recommendations = recommend_models(memory_gb)

        for rec in recommendations:
            expected_headroom = budget - rec["training_gb"]
            assert rec["headroom_gb"] == expected_headroom


@pytest.mark.unit
class TestGetOptimizedProfile:
    """Tests for get_optimized_profile() function."""

    def test_profile_ultra_192gb(self):
        """Test profile generation for Ultra 192GB."""
        profile = get_optimized_profile("m2", "ultra", 192)

        assert profile["generation"] == "m2"
        assert profile["variant"] == "ultra"
        assert profile["memory_gb"] == 192
        assert profile["training_budget_gb"] == 153  # 80% of 192
        assert profile["batch_size"] >= 4
        assert profile["lora_rank"] >= 128
        assert profile["model_tier"] == "large"

    def test_profile_base_16gb(self):
        """Test profile generation for Base 16GB."""
        profile = get_optimized_profile("m2", "base", 16)

        assert profile["generation"] == "m2"
        assert profile["variant"] == "base"
        assert profile["memory_gb"] == 16
        assert profile["training_budget_gb"] == 12  # 80% of 16
        assert profile["batch_size"] == 1  # Limited memory
        assert profile["grad_checkpoint"] is True  # Should be enabled for low memory

    def test_profile_unknown_memory_uses_fallback(self):
        """Test that unknown memory configs use sensible fallbacks."""
        # 200GB is not in HARDWARE_PROFILES but should still work
        profile = get_optimized_profile("m2", "ultra", 200)

        assert profile["memory_gb"] == 200
        assert profile["training_budget_gb"] == 160  # 80% of 200
        # Should use closest profile (192GB ultra) as base
        assert profile["batch_size"] >= 4

    def test_profile_gpu_cores_adjustment(self):
        """Test that GPU cores affect batch size."""
        # Ultra has 64+ cores, should get batch_size >= 4
        profile = get_optimized_profile("m2", "ultra", 96)
        assert profile["batch_size"] >= 4

    def test_profile_contains_required_keys(self):
        """Test that all required keys are present."""
        profile = get_optimized_profile("m2", "max", 64)

        required_keys = [
            "generation",
            "variant",
            "memory_gb",
            "gpu_cores",
            "training_budget_gb",
            "batch_size",
            "lora_rank",
            "lora_num_layers",
            "grad_checkpoint",
            "model_tier",
        ]
        for key in required_keys:
            assert key in profile, f"Missing key: {key}"


@pytest.mark.unit
class TestProfileStorage:
    """Tests for save_hardware_profile() and load_hardware_profile()."""

    @patch("hardware_profiles.PROFILE_PATH")
    def test_save_and_load_profile(self, mock_path, tmp_path):
        """Test saving and loading a profile."""
        test_profile_path = tmp_path / ".your_ai" / "hardware_profile.json"
        mock_path.__truediv__ = lambda self, x: test_profile_path.parent / x
        mock_path.parent = test_profile_path.parent

        # Create a mock Path that behaves correctly
        with patch("hardware_profiles.PROFILE_PATH", test_profile_path):
            profile = {
                "generation": "m2",
                "variant": "ultra",
                "memory_gb": 96,
                "batch_size": 4,
            }

            # Ensure parent directory exists
            test_profile_path.parent.mkdir(parents=True, exist_ok=True)

            # Save
            saved_path = save_hardware_profile(profile)
            assert saved_path.exists()

            # Load
            loaded = load_hardware_profile()
            assert loaded["generation"] == "m2"
            assert loaded["variant"] == "ultra"
            assert loaded["memory_gb"] == 96

    @patch("hardware_profiles.PROFILE_PATH")
    def test_profile_exists_when_present(self, mock_path, tmp_path):
        """Test profile_exists() returns True when file exists."""
        test_profile_path = tmp_path / "hardware_profile.json"
        test_profile_path.write_text('{"generation": "m2"}')

        with patch("hardware_profiles.PROFILE_PATH", test_profile_path):
            assert profile_exists() is True

    @patch("hardware_profiles.PROFILE_PATH")
    def test_profile_exists_when_absent(self, mock_path, tmp_path):
        """Test profile_exists() returns False when file doesn't exist."""
        test_profile_path = tmp_path / "nonexistent.json"

        with patch("hardware_profiles.PROFILE_PATH", test_profile_path):
            assert profile_exists() is False

    @patch("hardware_profiles.PROFILE_PATH")
    def test_load_profile_returns_none_when_missing(self, mock_path, tmp_path):
        """Test load_hardware_profile() returns None when file doesn't exist."""
        test_profile_path = tmp_path / "nonexistent.json"

        with patch("hardware_profiles.PROFILE_PATH", test_profile_path):
            assert load_hardware_profile() is None


@pytest.mark.unit
class TestHardwareConstants:
    """Tests for hardware configuration constants."""

    def test_gpu_cores_all_generations(self):
        """Test GPU cores are defined for all generations."""
        generations = ["m1", "m2", "m3", "m4"]
        variants = ["base", "pro", "max", "ultra"]

        for gen in generations:
            assert gen in GPU_CORES, f"Missing GPU cores for generation {gen}"
            for var in variants:
                assert var in GPU_CORES[gen], f"Missing GPU cores for {gen} {var}"
                assert GPU_CORES[gen][var] > 0

    def test_memory_options_reasonable(self):
        """Test memory options are within reasonable bounds."""
        for variant, options in MEMORY_OPTIONS.items():
            for mem in options:
                assert 8 <= mem <= 512, f"Unreasonable memory {mem}GB for {variant}"

    def test_hardware_profiles_have_required_keys(self):
        """Test all hardware profiles have required settings."""
        required_keys = ["batch_size", "lora_rank", "lora_num_layers", "grad_checkpoint"]

        for key, profile in HARDWARE_PROFILES.items():
            for req in required_keys:
                assert req in profile, f"Profile {key} missing {req}"

    def test_model_requirements_have_required_keys(self):
        """Test all model requirements have required keys."""
        required_keys = ["hf_name", "inference_gb", "training_gb", "params", "tier"]

        for name, req in MODEL_REQUIREMENTS.items():
            for key in required_keys:
                assert key in req, f"Model {name} missing {key}"


@pytest.mark.unit
class TestInputValidation:
    """Tests for input validation in hardware profiles."""

    def test_memory_bounds_in_profile(self):
        """Test that profile generation handles edge case memory values."""
        # Very low memory (should still work with minimums)
        profile = get_optimized_profile("m1", "base", 8)
        assert profile["batch_size"] >= 1

        # High memory
        profile = get_optimized_profile("m2", "ultra", 512)
        assert profile["training_budget_gb"] == 409  # 80% of 512

    def test_all_variants_valid(self):
        """Test that all variant strings produce valid profiles."""
        variants = ["base", "pro", "max", "ultra"]
        for var in variants:
            profile = get_optimized_profile("m2", var, 32)
            assert profile["variant"] == var

    def test_all_generations_valid(self):
        """Test that all generation strings produce valid profiles."""
        generations = ["m1", "m2", "m3", "m4"]
        for gen in generations:
            profile = get_optimized_profile(gen, "pro", 32)
            assert profile["generation"] == gen


@pytest.mark.unit
class TestDetectModelSize:
    """Tests for detect_model_size() function."""

    def test_detect_7b_model(self):
        """Test detection of 7B model."""
        category, params = detect_model_size("NousResearch/Hermes-2-Pro-Mistral-7B")
        assert category == "small"
        assert params == 7

    def test_detect_8b_model(self):
        """Test detection of 8B model."""
        category, params = detect_model_size("mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")
        assert category == "small"
        assert params == 8

    def test_detect_14b_model(self):
        """Test detection of 14B model."""
        category, params = detect_model_size(
            "huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2"
        )
        assert category == "medium"
        assert params == 14

    def test_detect_32b_model(self):
        """Test detection of 32B model."""
        category, params = detect_model_size("huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated")
        assert category == "large"
        assert params == 32

    def test_detect_70b_model(self):
        """Test detection of 70B model."""
        category, params = detect_model_size("huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated")
        assert category == "xlarge"
        assert params == 70

    def test_detect_dolphin_8b(self):
        """Test detection of Dolphin 8B model."""
        category, params = detect_model_size("cognitivecomputations/dolphin-2.9-llama3-8b")
        assert category == "small"
        assert params == 8

    def test_detect_hermes_70b(self):
        """Test detection of Hermes 70B model."""
        category, params = detect_model_size("NousResearch/Hermes-3-Llama-3.1-70B")
        assert category == "xlarge"
        assert params == 70

    def test_detect_unknown_model_fallback(self):
        """Test fallback for unknown model without size in name."""
        category, params = detect_model_size("some-org/unknown-model")
        # Should default to "small" for safety
        assert category == "small"
        assert params == 0

    def test_detect_mistral_fallback(self):
        """Test fallback detection for Mistral models without explicit size."""
        category, params = detect_model_size("mistralai/Mistral-v0.1")
        # Should use known pattern fallback
        assert category == "small"
        assert params == 7


@pytest.mark.unit
class TestScaleProfileForModel:
    """Tests for scale_profile_for_model() function."""

    def test_scale_large_profile_for_small_model(self):
        """Test scaling a large-tier profile for a 7B model - preserves LoRA, scales batch."""
        # Profile designed for 70B model
        profile = {
            "batch_size": 4,
            "lora_rank": 128,
            "lora_num_layers": 24,
            "grad_checkpoint": True,
            "model_tier": "large",
        }

        scaled = scale_profile_for_model(profile, "NousResearch/Hermes-2-Pro-Mistral-7B")

        # LoRA settings should be PRESERVED (use max hardware resources)
        assert scaled["lora_rank"] == 128  # Preserved
        assert scaled["lora_num_layers"] == 24  # Preserved
        # Batch size should increase since we have headroom (4 * 2.0 = 8)
        assert scaled["batch_size"] == 8
        # Grad checkpoint should stay enabled (saves activation memory)
        assert scaled["grad_checkpoint"] is True

    def test_scale_entry_profile_stays_same(self):
        """Test that entry-tier profile scales batch size for small model."""
        # Profile already designed for small models
        profile = {
            "batch_size": 2,
            "lora_rank": 32,
            "lora_num_layers": 8,
            "grad_checkpoint": True,
            "model_tier": "entry",
        }

        scaled = scale_profile_for_model(profile, "NousResearch/Hermes-2-Pro-Mistral-7B")

        # LoRA settings preserved
        assert scaled["lora_rank"] == 32
        assert scaled["lora_num_layers"] == 8
        # Batch size scaled up (2 * 2.0 = 4)
        assert scaled["batch_size"] == 4

    def test_scale_medium_profile_for_14b_model(self):
        """Test that medium-tier profile scales batch for 14B model."""
        profile = {
            "batch_size": 2,
            "lora_rank": 64,
            "lora_num_layers": 16,
            "grad_checkpoint": True,
            "model_tier": "medium",
        }

        scaled = scale_profile_for_model(
            profile, "huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2"
        )

        # LoRA settings preserved
        assert scaled["lora_rank"] == 64
        assert scaled["lora_num_layers"] == 16
        # Batch size scaled up (2 * 1.5 = 3)
        assert scaled["batch_size"] == 3

    def test_scale_large_profile_for_32b_model(self):
        """Test scaling a large-tier profile for a 32B model."""
        profile = {
            "batch_size": 4,
            "lora_rank": 128,
            "lora_num_layers": 24,
            "grad_checkpoint": True,
            "model_tier": "large",
        }

        scaled = scale_profile_for_model(
            profile, "huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated"
        )

        # LoRA settings preserved
        assert scaled["lora_rank"] == 128
        assert scaled["lora_num_layers"] == 24
        # Batch size scaled up (4 * 1.25 = 5)
        assert scaled["batch_size"] == 5

    def test_scale_does_not_modify_original(self):
        """Test that scaling creates a new profile, not modifying original."""
        original = {
            "batch_size": 4,
            "lora_rank": 128,
            "lora_num_layers": 24,
            "grad_checkpoint": True,
            "model_tier": "large",
        }
        original_copy = original.copy()

        scale_profile_for_model(original, "NousResearch/Hermes-2-Pro-Mistral-7B")

        # Original should be unchanged
        assert original == original_copy

    def test_scale_batch_size_capped(self):
        """Test that batch size scaling is capped at reasonable maximum."""
        profile = {
            "batch_size": 8,
            "lora_rank": 128,
            "lora_num_layers": 24,
            "grad_checkpoint": True,
            "model_tier": "large",
        }

        scaled = scale_profile_for_model(profile, "NousResearch/Hermes-2-Pro-Mistral-7B")

        # Batch size should be increased but capped at 16 (8 * 2.0 = 16)
        assert scaled["batch_size"] == 16


@pytest.mark.unit
class TestModelSizeConfigs:
    """Tests for MODEL_SIZE_CONFIGS constant."""

    def test_all_size_categories_defined(self):
        """Test that all size categories have configs."""
        expected_categories = ["small", "medium", "large", "xlarge"]
        for cat in expected_categories:
            assert cat in MODEL_SIZE_CONFIGS, f"Missing config for category {cat}"

    def test_configs_have_batch_multiplier(self):
        """Test that all configs have batch_size_multiplier key."""
        for cat, config in MODEL_SIZE_CONFIGS.items():
            assert "batch_size_multiplier" in config, f"Config {cat} missing batch_size_multiplier"

    def test_batch_multipliers_decrease_with_size(self):
        """Test that smaller models get larger batch multipliers (more memory headroom)."""
        # Smaller models should have higher batch multipliers
        assert (
            MODEL_SIZE_CONFIGS["small"]["batch_size_multiplier"]
            > MODEL_SIZE_CONFIGS["xlarge"]["batch_size_multiplier"]
        )


@pytest.mark.unit
class TestDetectModelSizeEdgeCases:
    """Additional edge case tests for detect_model_size() function."""

    def test_detect_very_large_model_100b(self):
        """Test detection of 100B+ models."""
        category, params = detect_model_size("meta/llama-3-100b-instruct")
        assert category == "xlarge"
        assert params == 100

    def test_detect_very_large_model_405b(self):
        """Test detection of extremely large models like Llama 405B."""
        category, params = detect_model_size("meta-llama/Meta-Llama-3.1-405B-Instruct")
        assert category == "xlarge"
        assert params == 405

    def test_detect_model_with_decimal_version(self):
        """Test detection with decimal version numbers (e.g., 3.1)."""
        category, params = detect_model_size("meta/llama-3.1-8b")
        assert category == "small"
        assert params == 8

    def test_detect_model_underscore_separator(self):
        """Test detection with underscore separator."""
        category, params = detect_model_size("org/model_7b_instruct")
        assert category == "small"
        assert params == 7

    def test_detect_model_hyphen_separator(self):
        """Test detection with hyphen separator."""
        category, params = detect_model_size("org/model-14b-chat")
        assert category == "medium"
        assert params == 14

    def test_detect_model_billion_spelled_out(self):
        """Test detection with 'billion' spelled out."""
        category, params = detect_model_size("org/7billion-parameter-model")
        assert category == "small"
        assert params == 7

    def test_detect_phi_2_model(self):
        """Test detection of Phi-2 3B model via fallback."""
        category, params = detect_model_size("microsoft/phi-2")
        assert category == "small"
        assert params == 3

    def test_detect_phi2_no_hyphen(self):
        """Test detection of phi2 without hyphen."""
        category, params = detect_model_size("microsoft/phi2")
        assert category == "small"
        assert params == 3

    def test_detect_qwen_default(self):
        """Test detection of Qwen models via fallback."""
        category, params = detect_model_size("Qwen/Qwen-Chat")
        assert category == "small"
        assert params == 7

    def test_detect_llama3_no_hyphen(self):
        """Test detection of llama3 models."""
        category, params = detect_model_size("meta/llama3-instruct")
        assert category == "small"
        assert params == 8

    def test_detect_local_path(self):
        """Test detection from local path."""
        category, params = detect_model_size("./models/mistral-7b")
        assert category == "small"
        assert params == 7

    def test_detect_model_uppercase_b(self):
        """Test detection with uppercase B."""
        category, params = detect_model_size("org/MODEL-70B-INSTRUCT")
        assert category == "xlarge"
        assert params == 70

    def test_detect_model_mixed_case(self):
        """Test detection with mixed case."""
        category, params = detect_model_size("Org/Model-32B-Chat")
        assert category == "large"
        assert params == 32

    def test_detect_empty_string(self):
        """Test detection with empty string."""
        category, params = detect_model_size("")
        assert category == "small"
        assert params == 0

    def test_detect_model_with_multiple_numbers(self):
        """Test model with multiple numbers (should pick first match)."""
        category, params = detect_model_size("llama-3-8b-v2")
        # Should detect 3 or 8, both would be small
        assert category == "small"
        assert params in [3, 8]  # Either is valid depending on regex order

    def test_detect_model_b_at_end(self):
        """Test pattern with 'b' at the end with trailing chars."""
        category, params = detect_model_size("org/model-7b-")
        assert category == "small"
        assert params == 7

    def test_detect_model_b_with_underscore_after(self):
        """Test pattern with 'b_' pattern."""
        category, params = detect_model_size("org/model-7b_chat")
        assert category == "small"
        assert params == 7

    def test_detect_boundary_10b_is_small(self):
        """Test that 10B is categorized as small."""
        category, params = detect_model_size("org/model-10b")
        assert category == "small"
        assert params == 10

    def test_detect_boundary_11b_is_medium(self):
        """Test that 11B is categorized as medium."""
        category, params = detect_model_size("org/model-11b")
        assert category == "medium"
        assert params == 11

    def test_detect_boundary_20b_is_medium(self):
        """Test that 20B is categorized as medium."""
        category, params = detect_model_size("org/model-20b")
        assert category == "medium"
        assert params == 20

    def test_detect_boundary_21b_is_large(self):
        """Test that 21B is categorized as large."""
        category, params = detect_model_size("org/model-21b")
        assert category == "large"
        assert params == 21

    def test_detect_boundary_50b_is_large(self):
        """Test that 50B is categorized as large."""
        category, params = detect_model_size("org/model-50b")
        assert category == "large"
        assert params == 50

    def test_detect_boundary_51b_is_xlarge(self):
        """Test that 51B is categorized as xlarge."""
        category, params = detect_model_size("org/model-51b")
        assert category == "xlarge"
        assert params == 51

    def test_detect_single_b_no_number(self):
        """Test handling of 'b' without preceding number."""
        category, params = detect_model_size("org/bert-base")
        assert category == "small"
        assert params == 0  # No size detected

    def test_detect_model_with_only_org_name(self):
        """Test with only organization name, no model."""
        category, params = detect_model_size("organization")
        assert category == "small"
        assert params == 0


@pytest.mark.unit
class TestScaleProfileForModelAdvanced:
    """Advanced tests for scale_profile_for_model() function."""

    def test_scale_profile_without_model_tier(self):
        """Test scaling with profile missing model_tier key."""
        profile = {
            "batch_size": 4,
            "lora_rank": 64,
            "lora_num_layers": 16,
            "grad_checkpoint": True,
        }

        scaled = scale_profile_for_model(profile, "NousResearch/Hermes-2-Pro-Mistral-7B")

        # LoRA settings preserved, batch scaled up
        assert scaled["lora_rank"] == 64
        assert scaled["lora_num_layers"] == 16
        assert scaled["batch_size"] == 8  # 4 * 2.0

    def test_scale_profile_with_invalid_tier(self):
        """Test scaling with invalid tier value."""
        profile = {
            "batch_size": 2,
            "lora_rank": 64,
            "lora_num_layers": 16,
            "grad_checkpoint": True,
            "model_tier": "invalid_tier",
        }

        scaled = scale_profile_for_model(profile, "NousResearch/Hermes-2-Pro-Mistral-7B")

        # Should handle gracefully - LoRA preserved, batch scaled
        assert scaled is not None
        assert scaled["lora_rank"] == 64
        assert scaled["batch_size"] == 4  # 2 * 2.0

    def test_scale_preserves_all_keys(self):
        """Test that scaling preserves all profile keys."""
        profile = {
            "batch_size": 4,
            "lora_rank": 128,
            "lora_num_layers": 24,
            "grad_checkpoint": True,
            "model_tier": "large",
            "custom_key": "custom_value",
            "another_key": 42,
        }

        scaled = scale_profile_for_model(profile, "NousResearch/Hermes-2-Pro-Mistral-7B")

        # All keys should be preserved
        assert "custom_key" in scaled
        assert scaled["custom_key"] == "custom_value"
        assert "another_key" in scaled
        assert scaled["another_key"] == 42
        # LoRA preserved
        assert scaled["lora_rank"] == 128
        assert scaled["lora_num_layers"] == 24

    def test_scale_medium_profile_for_small_model(self):
        """Test scaling a medium-tier profile for a 7B model - preserves LoRA."""
        profile = {
            "batch_size": 3,
            "lora_rank": 48,
            "lora_num_layers": 12,
            "grad_checkpoint": True,
            "model_tier": "medium",
        }

        scaled = scale_profile_for_model(profile, "NousResearch/Hermes-2-Pro-Mistral-7B")

        # LoRA settings preserved (use max hardware resources)
        assert scaled["lora_rank"] == 48
        assert scaled["lora_num_layers"] == 12
        # Batch scaled up (3 * 2.0 = 6)
        assert scaled["batch_size"] == 6

    def test_scale_with_large_initial_batch_size(self):
        """Test that very large initial batch sizes are still capped."""
        profile = {
            "batch_size": 6,
            "lora_rank": 128,
            "lora_num_layers": 24,
            "grad_checkpoint": True,
            "model_tier": "large",
        }

        scaled = scale_profile_for_model(profile, "NousResearch/Hermes-2-Pro-Mistral-7B")

        # 6 * 2.0 = 12
        assert scaled["batch_size"] == 12

    def test_scale_with_missing_batch_size(self):
        """Test scaling with profile missing batch_size."""
        profile = {
            "lora_rank": 128,
            "lora_num_layers": 24,
            "grad_checkpoint": True,
            "model_tier": "large",
        }

        scaled = scale_profile_for_model(profile, "NousResearch/Hermes-2-Pro-Mistral-7B")

        # Should handle missing batch_size gracefully (defaults to 2)
        assert scaled["lora_rank"] == 128  # Preserved
        assert scaled["batch_size"] == 4  # 2 * 2.0

    def test_scale_xlarge_model_on_large_profile(self):
        """Test that xlarge model gets no batch scaling (multiplier 1.0)."""
        profile = {
            "batch_size": 2,
            "lora_rank": 128,
            "lora_num_layers": 24,
            "grad_checkpoint": True,
            "model_tier": "large",
        }

        scaled = scale_profile_for_model(profile, "meta/llama-70b")

        # 70B is xlarge, multiplier is 1.0
        assert scaled["lora_rank"] == 128
        assert scaled["lora_num_layers"] == 24
        assert scaled["batch_size"] == 2  # 2 * 1.0

    def test_scale_with_very_small_3b_model(self):
        """Test scaling for very small 3B model - max batch scaling."""
        profile = {
            "batch_size": 2,
            "lora_rank": 64,
            "lora_num_layers": 16,
            "grad_checkpoint": True,
            "model_tier": "medium",
        }

        scaled = scale_profile_for_model(profile, "microsoft/phi-2")

        # 3B is "small" - LoRA preserved, batch scaled
        assert scaled["lora_rank"] == 64
        assert scaled["lora_num_layers"] == 16
        assert scaled["batch_size"] == 4  # 2 * 2.0

    def test_scale_14b_on_large_profile(self):
        """Test scaling 14B model on large-tier profile."""
        profile = {
            "batch_size": 4,
            "lora_rank": 128,
            "lora_num_layers": 24,
            "grad_checkpoint": True,
            "model_tier": "large",
        }

        scaled = scale_profile_for_model(profile, "org/model-14b")

        # 14B is medium - LoRA preserved, batch scaled by 1.5
        assert scaled["lora_rank"] == 128
        assert scaled["lora_num_layers"] == 24
        assert scaled["batch_size"] == 6  # 4 * 1.5

    def test_scale_maintains_gradient_checkpointing(self):
        """Test that gradient checkpointing is always preserved."""
        profile = {
            "batch_size": 4,
            "lora_rank": 128,
            "lora_num_layers": 24,
            "grad_checkpoint": False,  # Explicitly disabled
            "model_tier": "large",
        }

        scaled = scale_profile_for_model(profile, "NousResearch/Hermes-2-Pro-Mistral-7B")

        # Should preserve the grad_checkpoint setting
        assert "grad_checkpoint" in scaled
        assert scaled["grad_checkpoint"] is False

    def test_scale_unknown_size_category(self):
        """Test scaling with unknown model size category."""
        profile = {
            "batch_size": 4,
            "lora_rank": 64,
            "lora_num_layers": 16,
            "grad_checkpoint": True,
            "model_tier": "medium",
        }

        # Unknown model should default to "small" category
        scaled = scale_profile_for_model(profile, "unknown/model")

        # LoRA preserved, batch scaled by 2.0
        assert scaled["lora_rank"] == 64
        assert scaled["lora_num_layers"] == 16
        assert scaled["batch_size"] == 8  # 4 * 2.0


@pytest.mark.unit
class TestModelSizeConfigsComprehensive:
    """Comprehensive tests for MODEL_SIZE_CONFIGS constant."""

    def test_batch_size_multipliers_valid(self):
        """Test that batch size multipliers are reasonable values."""
        for cat, config in MODEL_SIZE_CONFIGS.items():
            multiplier = config["batch_size_multiplier"]
            assert multiplier > 0, f"{cat} has non-positive multiplier"
            assert multiplier <= 5.0, f"{cat} multiplier too large"

    def test_batch_multiplier_decreasing(self):
        """Test that batch multiplier decreases with model size (smaller models get more)."""
        categories = ["small", "medium", "large", "xlarge"]
        prev_mult = float("inf")
        for cat in categories:
            mult = MODEL_SIZE_CONFIGS[cat]["batch_size_multiplier"]
            assert mult <= prev_mult, f"{cat} multiplier not <= previous"
            prev_mult = mult

    def test_small_config_values(self):
        """Test specific values for small model config."""
        small = MODEL_SIZE_CONFIGS["small"]
        assert small["batch_size_multiplier"] == 2.0

    def test_medium_config_values(self):
        """Test specific values for medium model config."""
        medium = MODEL_SIZE_CONFIGS["medium"]
        assert medium["batch_size_multiplier"] == 1.5

    def test_large_config_values(self):
        """Test specific values for large model config."""
        large = MODEL_SIZE_CONFIGS["large"]
        assert large["batch_size_multiplier"] == 1.25

    def test_xlarge_config_values(self):
        """Test specific values for xlarge model config."""
        xlarge = MODEL_SIZE_CONFIGS["xlarge"]
        assert xlarge["batch_size_multiplier"] == 1.0

    def test_configs_immutable(self):
        """Test that configs are not accidentally modified."""
        # Get reference to config
        small_config = MODEL_SIZE_CONFIGS["small"]
        original_mult = small_config["batch_size_multiplier"]

        # Try to get config multiple times
        for _ in range(3):
            config = MODEL_SIZE_CONFIGS["small"]
            assert config["batch_size_multiplier"] == original_mult


@pytest.mark.unit
class TestModelSizeDetectionIntegration:
    """Integration tests for model size detection with real-world patterns."""

    def test_detect_multiple_deepseek_variants(self):
        """Test detection of various DeepSeek model variants."""
        test_cases = [
            ("huihui-ai/DeepSeek-R1-Distill-Qwen-7B", "small", 7),
            ("huihui-ai/DeepSeek-R1-Distill-Qwen-14B", "medium", 14),
            ("huihui-ai/DeepSeek-R1-Distill-Qwen-32B", "large", 32),
            ("huihui-ai/DeepSeek-R1-Distill-Llama-70B", "xlarge", 70),
        ]

        for model_path, expected_cat, expected_params in test_cases:
            category, params = detect_model_size(model_path)
            assert category == expected_cat, f"Failed for {model_path}"
            assert params == expected_params, f"Failed for {model_path}"

    def test_detect_multiple_llama_variants(self):
        """Test detection of various Llama model variants."""
        test_cases = [
            ("meta-llama/Llama-2-7b-hf", "small", 7),
            ("meta-llama/Llama-2-13b-chat-hf", "medium", 13),
            ("meta-llama/Meta-Llama-3-8B", "small", 8),
            ("meta-llama/Meta-Llama-3.1-70B", "xlarge", 70),
        ]

        for model_path, expected_cat, expected_params in test_cases:
            category, params = detect_model_size(model_path)
            assert category == expected_cat, f"Failed for {model_path}"
            assert params == expected_params, f"Failed for {model_path}"

    def test_detect_multiple_mistral_variants(self):
        """Test detection of various Mistral model variants."""
        test_cases = [
            ("mistralai/Mistral-7B-v0.1", "small", 7),
            ("mistralai/Mistral-7B-Instruct-v0.2", "small", 7),
            ("NousResearch/Hermes-2-Pro-Mistral-7B", "small", 7),
        ]

        for model_path, expected_cat, expected_params in test_cases:
            category, params = detect_model_size(model_path)
            assert category == expected_cat, f"Failed for {model_path}"
            assert params == expected_params, f"Failed for {model_path}"

    def test_detect_community_fine_tunes(self):
        """Test detection of community fine-tuned models."""
        test_cases = [
            ("cognitivecomputations/dolphin-2.9-llama3-8b", "small", 8),
            ("TheBloke/Llama-2-70B-GGUF", "xlarge", 70),
            ("NousResearch/Hermes-3-Llama-3.1-70B", "xlarge", 70),
        ]

        for model_path, expected_cat, expected_params in test_cases:
            category, params = detect_model_size(model_path)
            assert category == expected_cat, f"Failed for {model_path}"
            assert params == expected_params, f"Failed for {model_path}"


@pytest.mark.unit
class TestScaleProfileEndToEnd:
    """End-to-end tests simulating real usage scenarios."""

    def test_typical_m2_ultra_with_7b_model(self):
        """Test typical M2 Ultra profile with 7B model - max resources used."""
        # Typical M2 Ultra profile (designed for up to 70B)
        profile = {
            "batch_size": 4,
            "lora_rank": 128,
            "lora_num_layers": 24,
            "grad_checkpoint": True,
            "model_tier": "large",
        }

        scaled = scale_profile_for_model(profile, "NousResearch/Hermes-2-Pro-Mistral-7B")

        # LoRA settings preserved (use max hardware resources)
        assert scaled["lora_rank"] == 128
        assert scaled["lora_num_layers"] == 24
        assert scaled["batch_size"] == 8  # 4 * 2.0 multiplier
        assert scaled["grad_checkpoint"] is True

    def test_typical_m3_pro_with_14b_model(self):
        """Test typical M3 Pro profile with 14B model."""
        # Typical M3 Pro profile (designed for 14B-32B)
        profile = {
            "batch_size": 2,
            "lora_rank": 64,
            "lora_num_layers": 16,
            "grad_checkpoint": True,
            "model_tier": "medium",
        }

        scaled = scale_profile_for_model(profile, "huihui-ai/DeepSeek-R1-Distill-Qwen-14B")

        # LoRA preserved, batch scaled by 1.5
        assert scaled["lora_rank"] == 64
        assert scaled["lora_num_layers"] == 16
        assert scaled["batch_size"] == 3  # 2 * 1.5

    def test_oversized_profile_for_tiny_model(self):
        """Test large profile with tiny 3B model - max batch scaling."""
        profile = {
            "batch_size": 4,
            "lora_rank": 128,
            "lora_num_layers": 24,
            "grad_checkpoint": True,
            "model_tier": "large",
        }

        scaled = scale_profile_for_model(profile, "microsoft/phi-2")

        # LoRA preserved, batch scaled
        assert scaled["lora_rank"] == 128
        assert scaled["lora_num_layers"] == 24
        assert scaled["batch_size"] == 8  # 4 * 2.0

    def test_entry_level_profile_scales_batch(self):
        """Test entry-level profile scales batch for different model sizes."""
        profile = {
            "batch_size": 2,
            "lora_rank": 32,
            "lora_num_layers": 8,
            "grad_checkpoint": True,
            "model_tier": "entry",
        }

        # LoRA always preserved, batch scaled based on model size
        expected_batch = {
            "org/model-7b": 4,  # 2 * 2.0
            "org/model-14b": 3,  # 2 * 1.5
            "org/model-32b": 2,  # 2 * 1.25 = 2.5 -> 2
            "org/model-70b": 2,  # 2 * 1.0
        }

        for model_path, expected in expected_batch.items():
            scaled = scale_profile_for_model(profile, model_path)
            assert scaled["lora_rank"] == 32  # Always preserved
            assert scaled["lora_num_layers"] == 8  # Always preserved
            assert scaled["batch_size"] == expected, f"Failed for {model_path}"
