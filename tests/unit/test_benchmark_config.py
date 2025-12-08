"""
Unit tests for benchmark configuration module.

Tests configuration retrieval, priority filtering, and category mappings.
"""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from benchmark_config import (
    BenchmarkConfig,
    BENCHMARK_REGISTRY,
    TRUTHFULQA_CONFIG,
    CENSORBENCH_CONFIG,
    CATEGORY_MAPPING,
    get_benchmark_config,
    get_priority_benchmarks,
    get_all_benchmarks,
    get_mapped_benchmarks,
)


class TestBenchmarkConfig:
    """Test BenchmarkConfig dataclass and constants."""

    def test_truthfulqa_config(self):
        """Test TruthfulQA configuration is properly defined."""
        assert TRUTHFULQA_CONFIG.name == "truthfulqa"
        assert TRUTHFULQA_CONFIG.display_name == "TruthfulQA"
        assert TRUTHFULQA_CONFIG.dataset_id == "truthfulqa/truthful_qa"
        assert TRUTHFULQA_CONFIG.total_questions == 817
        assert TRUTHFULQA_CONFIG.license == "MIT"
        assert TRUTHFULQA_CONFIG.alignment_score == "high"
        assert TRUTHFULQA_CONFIG.min_pass_threshold == 0.50
        assert len(TRUTHFULQA_CONFIG.categories) > 0

    def test_censorbench_config(self):
        """Test CensorBench configuration is properly defined."""
        assert CENSORBENCH_CONFIG.name == "censorbench"
        assert CENSORBENCH_CONFIG.display_name == "CensorBench"
        assert CENSORBENCH_CONFIG.dataset_id is None  # Not on HuggingFace yet
        assert isinstance(CENSORBENCH_CONFIG.dataset_path, Path)
        assert CENSORBENCH_CONFIG.total_questions == 500
        assert CENSORBENCH_CONFIG.alignment_score == "high"
        assert CENSORBENCH_CONFIG.min_pass_threshold == 0.75

    def test_benchmark_registry_completeness(self):
        """Test that all expected benchmarks are in registry."""
        expected_benchmarks = [
            "truthfulqa",
            "censorbench",
            "safetybench",
            "forbidden_science",
            "toxigen",
        ]
        for name in expected_benchmarks:
            assert name in BENCHMARK_REGISTRY
            assert isinstance(BENCHMARK_REGISTRY[name], BenchmarkConfig)

    def test_benchmark_configs_have_required_fields(self):
        """Test that all benchmark configs have required fields."""
        required_fields = [
            "name",
            "display_name",
            "categories",
            "total_questions",
            "license",
            "description",
            "min_pass_threshold",
            "alignment_score",
        ]

        for config in BENCHMARK_REGISTRY.values():
            for field in required_fields:
                assert hasattr(config, field)
                assert getattr(config, field) is not None


class TestGetBenchmarkConfig:
    """Test get_benchmark_config() function."""

    def test_get_valid_benchmark(self):
        """Test retrieving a valid benchmark configuration."""
        config = get_benchmark_config("truthfulqa")
        assert config.name == "truthfulqa"
        assert isinstance(config, BenchmarkConfig)

    def test_get_all_valid_benchmarks(self):
        """Test retrieving all registered benchmarks."""
        for name in BENCHMARK_REGISTRY.keys():
            config = get_benchmark_config(name)
            assert config.name == name

    def test_get_invalid_benchmark_raises_error(self):
        """Test that invalid benchmark name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_benchmark_config("nonexistent_benchmark")

    def test_error_message_includes_available_benchmarks(self):
        """Test that error message lists available benchmarks."""
        with pytest.raises(ValueError) as exc_info:
            get_benchmark_config("invalid")
        assert "Available:" in str(exc_info.value)


class TestGetPriorityBenchmarks:
    """Test get_priority_benchmarks() filtering."""

    def test_get_high_priority_benchmarks(self):
        """Test retrieving high-priority benchmarks."""
        benchmarks = get_priority_benchmarks("high")
        assert len(benchmarks) == 2
        names = [b.name for b in benchmarks]
        assert "truthfulqa" in names
        assert "censorbench" in names

    def test_get_medium_priority_benchmarks(self):
        """Test retrieving medium-priority benchmarks."""
        benchmarks = get_priority_benchmarks("medium")
        assert len(benchmarks) == 2
        names = [b.name for b in benchmarks]
        assert "forbidden_science" in names
        assert "safetybench" in names

    def test_get_low_priority_benchmarks(self):
        """Test retrieving low-priority benchmarks."""
        benchmarks = get_priority_benchmarks("low")
        assert len(benchmarks) == 1
        assert benchmarks[0].name == "toxigen"

    def test_default_priority_is_high(self):
        """Test that default priority level is 'high'."""
        default = get_priority_benchmarks()
        explicit_high = get_priority_benchmarks("high")
        assert len(default) == len(explicit_high)
        assert [b.name for b in default] == [b.name for b in explicit_high]

    def test_invalid_priority_raises_error(self):
        """Test that invalid priority raises ValueError."""
        with pytest.raises(ValueError, match="Invalid priority"):
            get_priority_benchmarks("invalid")

    def test_all_priority_benchmarks_are_configs(self):
        """Test that all returned items are BenchmarkConfig objects."""
        for priority in ["high", "medium", "low"]:
            benchmarks = get_priority_benchmarks(priority)
            for benchmark in benchmarks:
                assert isinstance(benchmark, BenchmarkConfig)


class TestGetAllBenchmarks:
    """Test get_all_benchmarks() function."""

    def test_returns_all_benchmarks(self):
        """Test that all benchmarks are returned."""
        all_benchmarks = get_all_benchmarks()
        assert len(all_benchmarks) == len(BENCHMARK_REGISTRY)

    def test_returns_config_objects(self):
        """Test that returned items are BenchmarkConfig objects."""
        all_benchmarks = get_all_benchmarks()
        for benchmark in all_benchmarks:
            assert isinstance(benchmark, BenchmarkConfig)

    def test_contains_all_registry_entries(self):
        """Test that all registry entries are included."""
        all_benchmarks = get_all_benchmarks()
        benchmark_names = {b.name for b in all_benchmarks}
        registry_names = set(BENCHMARK_REGISTRY.keys())
        assert benchmark_names == registry_names


class TestCategoryMapping:
    """Test CATEGORY_MAPPING lookups."""

    def test_category_mapping_structure(self):
        """Test that category mapping has expected structure."""
        assert isinstance(CATEGORY_MAPPING, dict)
        for category, mappings in CATEGORY_MAPPING.items():
            assert isinstance(category, str)
            assert isinstance(mappings, list)
            for mapping in mappings:
                assert isinstance(mapping, tuple)
                assert len(mapping) == 2
                benchmark_name, benchmark_category = mapping
                assert isinstance(benchmark_name, str)
                assert isinstance(benchmark_category, str)

    def test_ccp_censorship_mapping(self):
        """Test CCP censorship category mapping."""
        mappings = CATEGORY_MAPPING["ccp_censorship"]
        assert len(mappings) >= 1
        assert ("censorbench", "Political_Sensitivity_CCP") in mappings

    def test_western_censorship_mapping(self):
        """Test Western censorship category mapping."""
        mappings = CATEGORY_MAPPING["western_censorship"]
        assert len(mappings) >= 1
        assert ("censorbench", "Political_Sensitivity_Western") in mappings

    def test_authority_bias_mapping(self):
        """Test authority bias category mapping."""
        mappings = CATEGORY_MAPPING["authority_bias"]
        assert len(mappings) >= 1
        # Should map to TruthfulQA categories
        benchmark_names = [m[0] for m in mappings]
        assert "truthfulqa" in benchmark_names

    def test_all_mapped_benchmarks_exist(self):
        """Test that all mapped benchmark names exist in registry."""
        for category, mappings in CATEGORY_MAPPING.items():
            for benchmark_name, benchmark_category in mappings:
                assert benchmark_name in BENCHMARK_REGISTRY, (
                    f"Benchmark {benchmark_name} in {category} mapping not in registry"
                )


class TestGetMappedBenchmarks:
    """Test get_mapped_benchmarks() function."""

    def test_get_valid_category_mapping(self):
        """Test retrieving mappings for a valid category."""
        mappings = get_mapped_benchmarks("authority_bias")
        assert isinstance(mappings, list)
        assert len(mappings) > 0

    def test_get_invalid_category_returns_empty(self):
        """Test that invalid category returns empty list."""
        mappings = get_mapped_benchmarks("nonexistent_category")
        assert mappings == []

    def test_all_categories_have_mappings(self):
        """Test that all defined categories return mappings."""
        for category in CATEGORY_MAPPING.keys():
            mappings = get_mapped_benchmarks(category)
            assert len(mappings) > 0

    def test_mapping_format(self):
        """Test that returned mappings have correct format."""
        mappings = get_mapped_benchmarks("authority_bias")
        for mapping in mappings:
            assert isinstance(mapping, tuple)
            assert len(mapping) == 2
            benchmark_name, benchmark_category = mapping
            assert isinstance(benchmark_name, str)
            assert isinstance(benchmark_category, str)
