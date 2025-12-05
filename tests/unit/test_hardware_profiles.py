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
    HARDWARE_PROFILES,
    MODEL_REQUIREMENTS,
    GPU_CORES,
    MEMORY_OPTIONS,
)


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
