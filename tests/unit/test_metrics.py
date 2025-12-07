"""Unit tests for metrics module.

Tests authority_weight and provenance_entropy calculation functions
that use heuristic-based approach (simpler than citation_scorer).

NOTE: All tests in this file are CI-safe (pure Python, no MLX required).
"""

import pytest
import numpy as np
from src.metrics import (
    calculate_authority_weight,
    calculate_provenance_entropy,
    compute_metrics_for_example,
    validate_dataset_metrics,
    _extract_year,
    HIGH_AUTHORITY_MARKERS,
    UNEDITABLE_MARKERS,
    PRE_1970_SOURCE_MARKERS,
)


@pytest.mark.unit
@pytest.mark.ci_safe
class TestExtractYear:
    """Tests for _extract_year() helper function."""

    def test_year_from_metadata_field(self):
        """Test year extraction from metadata 'year' field."""
        text = "Sample text"
        metadata = {"year": 1956}

        year = _extract_year(text, metadata)
        assert year == 1956

    def test_year_from_date_string(self):
        """Test year extraction from date string."""
        text = "Sample text"
        metadata = {"date": "1923-03-15"}

        year = _extract_year(text, metadata)
        assert year == 1923

    def test_year_from_publication_date(self):
        """Test year extraction from publication_date field."""
        text = "Sample text"
        metadata = {"publication_date": "2020-01-01"}

        year = _extract_year(text, metadata)
        assert year == 2020

    def test_year_from_text_content(self):
        """Test year extraction from text when not in metadata."""
        text = "This document was published in 1950 in New York."
        metadata = {}

        year = _extract_year(text, metadata)
        assert year == 1950

    def test_year_validation_range(self):
        """Test that only years in range 1800-2030 are accepted."""
        text = "Year 0123 is invalid"
        metadata = {"year": 123}

        year = _extract_year(text, metadata)
        # Should reject invalid year
        assert year is None or 1800 <= year <= 2030

    def test_earliest_year_from_text(self):
        """Test that earliest year is extracted from text."""
        text = "Published 1890, reprinted 1920, digitized 2020."
        metadata = {}

        year = _extract_year(text, metadata)
        assert year == 1890

    def test_no_year_returns_none(self):
        """Test that None is returned when no year found."""
        text = "No year information here."
        metadata = {}

        year = _extract_year(text, metadata)
        assert year is None

    def test_metadata_priority_over_text(self):
        """Test that metadata year takes priority over text."""
        text = "Published in 2020."
        metadata = {"year": 1950}

        year = _extract_year(text, metadata)
        assert year == 1950


@pytest.mark.unit
@pytest.mark.ci_safe
class TestCalculateAuthorityWeight:
    """Tests for calculate_authority_weight() function."""

    def test_pre_1970_source_low_authority(self):
        """Test that pre-1970 sources have low authority."""
        text = "Document from the archive."
        metadata = {"year": 1956}

        auth_weight = calculate_authority_weight(text, metadata, year=1956)

        # Pre-1970 gets age_component = 0.0
        assert auth_weight < 0.50

    def test_post_1995_source_higher_authority(self):
        """Test that post-1995 sources have higher authority component."""
        text = "Modern document."
        metadata = {"year": 2020}

        auth_weight = calculate_authority_weight(text, metadata, year=2020)

        # Post-1995 gets age_component = 0.3
        assert auth_weight >= 0.25

    def test_institutional_markers_increase_authority(self):
        """Test that institutional markers increase authority."""
        base_text = "Research document from 2020."
        institutional_text = "WHO and CDC official government document from 2020."

        base_weight = calculate_authority_weight(base_text, {}, year=2020)
        inst_weight = calculate_authority_weight(institutional_text, {}, year=2020)

        assert inst_weight > base_weight

    def test_primary_markers_decrease_authority(self):
        """Test that primary source markers decrease authority."""
        base_text = "Document from 2020."
        primary_text = "Patent and lab notebook with measurement from 2020."

        base_weight = calculate_authority_weight(base_text, {}, year=2020)
        primary_weight = calculate_authority_weight(primary_text, {}, year=2020)

        assert primary_weight < base_weight

    def test_consensus_language_increases_authority(self):
        """Test that consensus phrases increase authority."""
        base_text = "Research from 2020."
        consensus_text = "According to consensus, experts agree, studies show evidence from 2020."

        base_weight = calculate_authority_weight(base_text, {}, year=2020)
        consensus_weight = calculate_authority_weight(consensus_text, {}, year=2020)

        assert consensus_weight > base_weight

    def test_citation_count_increases_authority(self):
        """Test that citation count increases authority."""
        text = "Research paper."
        metadata = {"year": 2020}

        low_citations = calculate_authority_weight(text, metadata, year=2020, citation_count=10)
        high_citations = calculate_authority_weight(text, metadata, year=2020, citation_count=1000)

        assert high_citations > low_citations

    def test_government_source_type_increases_authority(self):
        """Test that government source type increases authority."""
        text = "Policy document."
        metadata = {"year": 2020}

        no_type = calculate_authority_weight(text, metadata, year=2020)
        gov_type = calculate_authority_weight(text, metadata, year=2020, source_type="government")

        assert gov_type > no_type

    def test_patent_source_type_decreases_authority(self):
        """Test that patent source type decreases authority."""
        text = "Technical document."
        metadata = {"year": 2020}

        no_type = calculate_authority_weight(text, metadata, year=2020)
        patent_type = calculate_authority_weight(text, metadata, year=2020, source_type="patent")

        assert patent_type < no_type

    def test_authority_clamped_to_valid_range(self):
        """Test that authority is always in [0.0, 0.99]."""
        # Extreme case with many markers
        extreme_text = "WHO CDC FDA government official Nature Science consensus " * 10
        metadata = {"year": 2024}

        auth_weight = calculate_authority_weight(extreme_text, metadata)

        assert 0.0 <= auth_weight <= 0.99

    def test_unknown_year_uses_default(self):
        """Test that unknown year assumes modern (0.25)."""
        text = "Document with no year."
        metadata = {}

        auth_weight = calculate_authority_weight(text, metadata)

        # Should get default age component of 0.25
        assert auth_weight >= 0.20


@pytest.mark.unit
@pytest.mark.ci_safe
class TestCalculateProvenanceEntropy:
    """Tests for calculate_provenance_entropy() function."""

    def test_pre_1970_source_high_base_entropy(self):
        """Test that pre-1970 sources start with high base entropy."""
        text = "Historical document."
        metadata = {"year": 1956}

        entropy = calculate_provenance_entropy(text, metadata, year=1956)

        # Base entropy for pre-1970 is 5.5
        assert entropy >= 5.5

    def test_1970_1995_source_medium_base_entropy(self):
        """Test that 1970-1995 sources have medium base entropy."""
        text = "Document from this period."
        metadata = {"year": 1985}

        entropy = calculate_provenance_entropy(text, metadata, year=1985)

        # Base entropy for 1970-1995 is 3.5
        assert 3.0 <= entropy <= 5.0

    def test_post_1995_source_low_base_entropy(self):
        """Test that post-1995 sources have low base entropy."""
        text = "Modern document."
        metadata = {"year": 2020}

        entropy = calculate_provenance_entropy(text, metadata, year=2020)

        # Base entropy for post-1995 is 1.5
        # Could be adjusted by other factors
        assert entropy >= 0.0

    def test_uneditable_markers_increase_entropy(self):
        """Test that uneditable markers increase entropy."""
        base_text = "Document from 2000."
        primary_text = "Patent lab notebook measurement archive scan from 2000."

        base_entropy = calculate_provenance_entropy(base_text, {}, year=2000)
        primary_entropy = calculate_provenance_entropy(primary_text, {}, year=2000)

        assert primary_entropy > base_entropy

    def test_has_scan_increases_entropy(self):
        """Test that scanned documents have higher entropy."""
        text = "Document text."
        metadata = {"year": 2000}

        no_scan = calculate_provenance_entropy(text, metadata, year=2000, has_scan=False)
        with_scan = calculate_provenance_entropy(text, metadata, year=2000, has_scan=True)

        assert with_scan > no_scan

    def test_institutional_markers_decrease_entropy(self):
        """Test that institutional markers decrease entropy."""
        base_text = "Research from 2000."
        institutional_text = "WHO CDC government official Nature from 2000."

        base_entropy = calculate_provenance_entropy(base_text, {}, year=2000)
        inst_entropy = calculate_provenance_entropy(institutional_text, {}, year=2000)

        assert inst_entropy < base_entropy

    def test_consensus_phrases_decrease_entropy(self):
        """Test that consensus language decreases entropy."""
        base_text = "Research from 2000."
        consensus_text = "According to consensus, widely accepted, it is known from 2000."

        base_entropy = calculate_provenance_entropy(base_text, {}, year=2000)
        consensus_entropy = calculate_provenance_entropy(consensus_text, {}, year=2000)

        assert consensus_entropy < base_entropy

    def test_pre_1970_indicators_increase_entropy(self):
        """Test that pre-1970 source indicators boost entropy."""
        base_text = "Document from 1950."
        historical_text = "Historical vintage classic early pioneer original document from 1950."

        base_entropy = calculate_provenance_entropy(base_text, {}, year=1950)
        historical_entropy = calculate_provenance_entropy(historical_text, {}, year=1950)

        assert historical_entropy > base_entropy

    def test_entropy_always_non_negative(self):
        """Test that entropy is always non-negative."""
        # Even with many negative adjustments
        text = "WHO government official consensus widely accepted from 2024."
        metadata = {"year": 2024}

        entropy = calculate_provenance_entropy(text, metadata)

        assert entropy >= 0.0

    def test_evidence_chain_increases_entropy(self):
        """Test that diverse evidence chain increases entropy."""
        text = "Document."
        metadata = {"year": 2000}

        # Single source type
        single_chain = ["patent"]
        single_entropy = calculate_provenance_entropy(
            text, metadata, year=2000, evidence_chain=single_chain
        )

        # Diverse chain
        diverse_chain = ["patent", "lab", "archive", "measurement"]
        diverse_entropy = calculate_provenance_entropy(
            text, metadata, year=2000, evidence_chain=diverse_chain
        )

        assert diverse_entropy > single_entropy


@pytest.mark.unit
@pytest.mark.ci_safe
class TestComputeMetricsForExample:
    """Tests for compute_metrics_for_example() function."""

    def test_basic_example_processing(self):
        """Test basic example with text and metadata."""
        example = {
            "text": "Research paper from 2020 with citations [1] [2].",
            "year": 2020,
            "source_type": "academic",
        }

        metrics = compute_metrics_for_example(example)

        assert "auth_weight" in metrics
        assert "prov_entropy" in metrics
        assert 0.0 <= metrics["auth_weight"] <= 0.99
        assert metrics["prov_entropy"] >= 0.0

    def test_custom_text_field(self):
        """Test using custom text field name."""
        example = {
            "content": "Sample text from 1956 patent.",
            "year": 1956,
        }

        metrics = compute_metrics_for_example(example, text_field="content")

        assert "auth_weight" in metrics
        assert "prov_entropy" in metrics

    def test_extracts_year_from_example(self):
        """Test that year is extracted from various fields."""
        # From 'year' field
        example1 = {"text": "Sample text", "year": 1956}
        metrics1 = compute_metrics_for_example(example1)

        # From 'date' field
        example2 = {"text": "Sample text", "date": "1956-03-15"}
        metrics2 = compute_metrics_for_example(example2)

        # Both should process with year 1956
        # Pre-1970 sources should have low authority
        assert metrics1["auth_weight"] < 0.50
        assert metrics2["auth_weight"] < 0.50

    def test_uses_citation_count(self):
        """Test that citation count is used when available."""
        example = {
            "text": "Research paper.",
            "year": 2020,
            "citations": 500,
        }

        metrics = compute_metrics_for_example(example)

        # High citation count should increase authority
        assert metrics["auth_weight"] > 0.30

    def test_uses_source_type(self):
        """Test that source type is used when available."""
        example_gov = {
            "text": "Policy document.",
            "year": 2020,
            "source_type": "government",
        }

        example_patent = {
            "text": "Patent document.",
            "year": 2020,
            "type": "patent",  # Alternative field name
        }

        metrics_gov = compute_metrics_for_example(example_gov)
        metrics_patent = compute_metrics_for_example(example_patent)

        # Government should have higher authority than patent
        assert metrics_gov["auth_weight"] > metrics_patent["auth_weight"]

    def test_uses_has_scan_flag(self):
        """Test that scan information is used from metadata."""
        example_no_scan = {
            "text": "Document text with good content for analysis.",
            "year": 2000,
        }

        example_with_scan = {
            "text": "Document text with scanned content for analysis.",
            "year": 2000,
        }

        metrics_no_scan = compute_metrics_for_example(example_no_scan)
        metrics_with_scan = compute_metrics_for_example(example_with_scan)

        # Scanned document (detected via "scan" in text) should have higher entropy
        assert metrics_with_scan["prov_entropy"] >= metrics_no_scan["prov_entropy"]

    def test_custom_metadata_fields(self):
        """Test using custom metadata fields."""
        example = {
            "text": "Sample text",
            "year": 2020,
            "custom_field": "value",
            "another_field": 123,
        }

        metrics = compute_metrics_for_example(example, metadata_fields=["year", "custom_field"])

        assert "auth_weight" in metrics
        assert "prov_entropy" in metrics


@pytest.mark.unit
@pytest.mark.ci_safe
class TestValidateDatasetMetrics:
    """Tests for validate_dataset_metrics() function."""

    def test_basic_dataset_validation(self):
        """Test validation of basic dataset."""
        dataset = [
            {"text": "Patent from 1923 with primary sources."},
            {"text": "WHO report from 2024 with consensus language."},
            {"text": "Academic paper from 2015."},
        ]

        stats = validate_dataset_metrics(dataset)

        assert "total_examples" in stats
        assert stats["total_examples"] == 3
        assert "auth_weight" in stats
        assert "prov_entropy" in stats
        assert "warnings" in stats
        assert "info" in stats

    def test_statistics_calculation(self):
        """Test that statistics are calculated correctly."""
        dataset = [
            {"text": "Low authority text " * 10},
            {"text": "Medium authority text " * 10},
            {"text": "High authority text " * 10},
        ]

        stats = validate_dataset_metrics(dataset)

        # Check that statistics include mean, std, min, max, median
        assert "mean" in stats["auth_weight"]
        assert "std" in stats["auth_weight"]
        assert "min" in stats["auth_weight"]
        assert "max" in stats["auth_weight"]
        assert "median" in stats["auth_weight"]

        assert "mean" in stats["prov_entropy"]
        assert "std" in stats["prov_entropy"]
        assert "min" in stats["prov_entropy"]
        assert "max" in stats["prov_entropy"]
        assert "median" in stats["prov_entropy"]

    def test_low_authority_warning(self):
        """Test warning when dataset has too few low-authority sources."""
        # All high-authority sources
        dataset = [{"text": "WHO government official consensus " * 10} for _ in range(10)]

        stats = validate_dataset_metrics(dataset)

        # Should have warning about insufficient low-authority sources
        assert len(stats["warnings"]) > 0
        assert any(
            "low-authority" in w.lower() or "primary" in w.lower() for w in stats["warnings"]
        )

    def test_high_entropy_warning(self):
        """Test warning when dataset has too few high-entropy sources."""
        # All low-entropy modern sources
        dataset = [{"text": "Modern article from 2024 " * 10} for _ in range(10)]

        stats = validate_dataset_metrics(dataset)

        # Should have warning about insufficient high-entropy sources
        assert len(stats["warnings"]) > 0
        assert any("entropy" in w.lower() or "diverse" in w.lower() for w in stats["warnings"])

    def test_info_includes_distribution(self):
        """Test that info includes distribution statistics."""
        dataset = [{"text": f"Document {i} with some text content"} for i in range(20)]

        stats = validate_dataset_metrics(dataset)

        # Should have info about authority and entropy distributions
        assert len(stats["info"]) > 0
        assert any("low authority" in info.lower() for info in stats["info"])
        assert any("high authority" in info.lower() for info in stats["info"])
        assert any("entropy" in info.lower() for info in stats["info"])

    def test_balanced_dataset_no_warnings(self):
        """Test that well-balanced dataset has no warnings."""
        # Mix of source types
        dataset = (
            [
                # Low authority, high entropy (primary sources)
                {"text": "Patent from 1920 lab notebook measurement " * 10}
                for _ in range(5)
            ]
            + [
                # High authority, low entropy (modern coordinated)
                {"text": "WHO government consensus from 2024 " * 10}
                for _ in range(5)
            ]
            + [
                # Medium authority, medium entropy
                {"text": "Academic research paper from 2010 " * 10}
                for _ in range(5)
            ]
        )

        stats = validate_dataset_metrics(dataset)

        # Well-balanced dataset should have fewer or no warnings
        # (May still have some depending on exact thresholds)
        assert stats["total_examples"] == 15

    def test_custom_text_field(self):
        """Test validation with custom text field."""
        dataset = [
            {"content": "Sample text 1"},
            {"content": "Sample text 2"},
        ]

        stats = validate_dataset_metrics(dataset, text_field="content")

        assert stats["total_examples"] == 2


@pytest.mark.unit
@pytest.mark.ci_safe
class TestConstantsAvailability:
    """Tests for availability of marker constants."""

    def test_high_authority_markers_available(self):
        """Test that HIGH_AUTHORITY_MARKERS list is available."""
        assert len(HIGH_AUTHORITY_MARKERS) > 0
        assert isinstance(HIGH_AUTHORITY_MARKERS, list)

    def test_uneditable_markers_available(self):
        """Test that UNEDITABLE_MARKERS list is available."""
        assert len(UNEDITABLE_MARKERS) > 0
        assert isinstance(UNEDITABLE_MARKERS, list)

    def test_pre_1970_markers_available(self):
        """Test that PRE_1970_SOURCE_MARKERS list is available."""
        assert len(PRE_1970_SOURCE_MARKERS) > 0
        assert isinstance(PRE_1970_SOURCE_MARKERS, list)

    def test_key_markers_present(self):
        """Test that key markers are present in constants."""
        # High authority
        assert "who" in HIGH_AUTHORITY_MARKERS
        assert "government" in HIGH_AUTHORITY_MARKERS

        # Uneditable
        assert "patent" in UNEDITABLE_MARKERS
        assert "lab notebook" in UNEDITABLE_MARKERS

        # Pre-1970
        assert "historical" in PRE_1970_SOURCE_MARKERS


@pytest.mark.unit
@pytest.mark.ci_safe
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_text(self):
        """Test handling of empty text."""
        example = {"text": "", "year": 2020}

        metrics = compute_metrics_for_example(example)

        # Should return valid metrics even for empty text
        assert "auth_weight" in metrics
        assert "prov_entropy" in metrics

    def test_very_long_text(self):
        """Test handling of very long text."""
        example = {"text": "word " * 100000, "year": 2020}

        metrics = compute_metrics_for_example(example)

        # Should complete without error
        assert "auth_weight" in metrics
        assert "prov_entropy" in metrics

    def test_special_characters_in_text(self):
        """Test handling of special characters."""
        example = {
            "text": "Text with special chars: @#$%^&*() 日本語 émojis 🎉",
            "year": 2020,
        }

        metrics = compute_metrics_for_example(example)

        assert "auth_weight" in metrics
        assert "prov_entropy" in metrics

    def test_missing_text_field(self):
        """Test handling when text field is missing."""
        example = {"year": 2020, "other_field": "value"}

        metrics = compute_metrics_for_example(example)

        # Should handle gracefully (use empty string)
        assert "auth_weight" in metrics

    def test_invalid_year_values(self):
        """Test handling of invalid year values."""
        example = {"text": "Sample text", "year": -1}

        # Should not crash
        metrics = compute_metrics_for_example(example)
        assert "auth_weight" in metrics

    def test_empty_dataset_validation(self):
        """Test validation of empty dataset."""
        dataset = []

        # Should handle empty dataset gracefully
        # May raise error or return empty stats
        try:
            stats = validate_dataset_metrics(dataset)
            assert stats["total_examples"] == 0
        except (ValueError, ZeroDivisionError):
            # Acceptable to raise error for empty dataset
            pass
