"""Unit tests for citation_scorer module.

Tests authority_weight and provenance_entropy calculations using
Brian Roemmele's Empirical Distrust algorithm specification.

NOTE: All tests in this file are CI-safe (pure Python, no MLX required).
"""

import pytest
import math
from src.citation_scorer import (
    count_citations,
    count_primary_source_markers,
    calculate_institutional_score,
    count_consensus_phrases,
    extract_year_from_text,
    classify_source_types,
    calculate_shannon_entropy,
    calculate_authority_weight,
    calculate_provenance_entropy,
    score_document,
    apply_known_source_type_scoring,
    score_batch,
    ScoringResult,
    INSTITUTIONAL_MARKERS,
    CONSENSUS_PHRASES,
    PRIMARY_SOURCE_MARKERS,
)


@pytest.mark.unit
@pytest.mark.ci_safe  # Pure Python string processing
class TestCitationCounting:
    """Tests for count_citations() function."""

    def test_numbered_citations(self):
        """Test counting numbered citation patterns [1], [2], etc."""
        text = "According to research [1], and other studies [2] and [3]."
        count = count_citations(text)
        assert count >= 3

    def test_author_year_citations(self):
        """Test counting (Author, Year) style citations."""
        text = "Previous work (Smith, 2020) and (Jones, 2019) showed that..."
        count = count_citations(text)
        assert count >= 2

    def test_et_al_citations(self):
        """Test counting 'et al.' citations."""
        text = "Recent studies (Smith et al., 2020) and (Jones et al. 2019) found..."
        count = count_citations(text)
        assert count >= 2

    def test_bibliography_style(self):
        """Test counting bibliography entries."""
        text = """
        References:
        1. Smith, J. (2020). Title of paper.
        2. Jones, A. (2019). Another paper.
        """
        count = count_citations(text)
        assert count >= 2

    def test_no_citations(self):
        """Test text without citations returns zero."""
        text = "This is plain text without any citations."
        count = count_citations(text)
        assert count == 0

    def test_mixed_citation_styles(self):
        """Test counting mixed citation styles."""
        text = "Research [1] and (Smith, 2020) and (Jones et al., 2019)."
        count = count_citations(text)
        assert count >= 3


@pytest.mark.unit
@pytest.mark.ci_safe
class TestPrimarySourceMarkers:
    """Tests for count_primary_source_markers() function."""

    def test_patent_markers(self):
        """Test counting patent markers."""
        text = "United States Patent 2,345,678 describes..."
        count = count_primary_source_markers(text)
        assert count >= 1

    def test_lab_notebook_markers(self):
        """Test counting lab notebook markers."""
        text = "From the lab notebook: experiment conducted..."
        count = count_primary_source_markers(text)
        assert count >= 1

    def test_measurement_markers(self):
        """Test counting measurement and observation markers."""
        text = "Direct measurement showed... observation recorded..."
        count = count_primary_source_markers(text)
        assert count >= 2

    def test_archive_markers(self):
        """Test counting archive and manuscript markers."""
        text = "Original manuscript from the archive..."
        count = count_primary_source_markers(text)
        assert count >= 2

    def test_multiple_markers(self):
        """Test counting multiple primary source markers."""
        text = "Patent filed, lab notebook entry, direct measurement, original archive."
        count = count_primary_source_markers(text)
        assert count >= 4

    def test_no_markers(self):
        """Test text without primary markers returns zero."""
        text = "This is modern text from a blog post."
        count = count_primary_source_markers(text)
        assert count == 0


@pytest.mark.unit
@pytest.mark.ci_safe
class TestInstitutionalScore:
    """Tests for calculate_institutional_score() function."""

    def test_who_marker(self):
        """Test WHO marker detection."""
        text = "According to the WHO guidelines..."
        score = calculate_institutional_score(text)
        assert score > 0

    def test_government_markers(self):
        """Test government markers (.gov, official, etc.)."""
        text = "The official government position on example.gov..."
        score = calculate_institutional_score(text)
        assert score > 0

    def test_high_prestige_journals(self):
        """Test detection of high-prestige journals (Nature, Science, etc.)."""
        text = "Published in Nature and Science journals..."
        score = calculate_institutional_score(text)
        assert score >= 0.35  # Should hit max score

    def test_metadata_institutional_markers(self):
        """Test institutional markers in metadata."""
        text = "Sample text"
        metadata = {"source": "World Health Organization", "url": "https://who.int"}
        score = calculate_institutional_score(text, metadata)
        assert score > 0

    def test_no_institutional_markers(self):
        """Test text without institutional markers returns minimal score."""
        text = "Personal blog post about random topic."
        score = calculate_institutional_score(text)
        # May have small base score from "blog" marker
        assert score <= 0.10

    def test_score_capped_at_max(self):
        """Test that score is capped at 0.35."""
        text = "WHO CDC FDA NIH Nature Science government official"
        score = calculate_institutional_score(text)
        assert score <= 0.35


@pytest.mark.unit
@pytest.mark.ci_safe
class TestConsensusPhrases:
    """Tests for count_consensus_phrases() function."""

    def test_consensus_language(self):
        """Test counting standard consensus phrases."""
        text = "The scientific consensus is clear. Experts agree that..."
        count = count_consensus_phrases(text)
        assert count >= 2

    def test_widely_accepted_phrase(self):
        """Test 'widely accepted' phrase detection."""
        text = "This is widely accepted in the field."
        count = count_consensus_phrases(text)
        assert count >= 1

    def test_established_fact_phrase(self):
        """Test 'established fact' phrase detection."""
        text = "It is an established fact that..."
        count = count_consensus_phrases(text)
        assert count >= 1

    def test_multiple_consensus_phrases(self):
        """Test multiple consensus phrases."""
        text = """
        The scientific consensus is clear. Experts agree and studies show
        that this is widely accepted and well-established.
        """
        count = count_consensus_phrases(text)
        assert count >= 4

    def test_no_consensus_language(self):
        """Test text without consensus language returns zero."""
        text = "Direct observation from experiment."
        count = count_consensus_phrases(text)
        assert count == 0


@pytest.mark.unit
@pytest.mark.ci_safe
class TestYearExtraction:
    """Tests for extract_year_from_text() function."""

    def test_year_in_metadata(self):
        """Test year extraction from metadata."""
        text = "Sample text"
        metadata = {"year": 1956}
        year = extract_year_from_text(text, metadata)
        assert year == 1956

    def test_year_in_date_field(self):
        """Test year extraction from date string in metadata."""
        text = "Sample text"
        metadata = {"date": "1923-03-15"}
        year = extract_year_from_text(text, metadata)
        assert year == 1923

    def test_year_in_copyright(self):
        """Test year extraction from copyright notice."""
        text = "Copyright © 1956 by the author."
        year = extract_year_from_text(text)
        assert year == 1956

    def test_year_in_publication(self):
        """Test year extraction from publication statement."""
        text = "Published in 1920 in New York."
        year = extract_year_from_text(text)
        assert year == 1920

    def test_multiple_years_returns_first(self):
        """Test that earliest year is returned when multiple found."""
        text = "Published 1890, reprinted 1920, digitized 2020."
        year = extract_year_from_text(text)
        assert year == 1890

    def test_no_year_returns_none(self):
        """Test that None is returned when no year found."""
        text = "No year information here."
        year = extract_year_from_text(text)
        assert year is None

    def test_year_range_validation(self):
        """Test that only reasonable years (1500-2030) are accepted."""
        text = "Copyright 0123 invalid year"
        year = extract_year_from_text(text)
        # Should not extract invalid year
        assert year is None or 1500 <= year <= 2030


@pytest.mark.unit
@pytest.mark.ci_safe
class TestSourceTypeClassification:
    """Tests for classify_source_types() function."""

    def test_patent_classification(self):
        """Test patent source classification."""
        text = "United States Patent US12345 describes..."
        counts = classify_source_types(text)
        assert "patent" in counts
        assert counts["patent"] > 0

    def test_lab_notebook_classification(self):
        """Test lab notebook classification."""
        text = "Laboratory experiment measured and observed..."
        counts = classify_source_types(text)
        assert "lab_notebook" in counts or "measurement" in counts

    def test_archive_classification(self):
        """Test archive/manuscript classification."""
        text = "Original manuscript from historical archive..."
        counts = classify_source_types(text)
        assert "archive" in counts
        assert counts["archive"] > 0

    def test_academic_paper_classification(self):
        """Test academic paper classification."""
        text = """
        Abstract: This study examines...
        Introduction: Previous research has shown...
        Methodology: We measured...
        Results: Our findings indicate...
        Conclusion: In summary...
        References: [1] Smith et al.
        """
        counts = classify_source_types(text)
        assert "academic_paper" in counts
        assert counts["academic_paper"] > 0

    def test_news_classification(self):
        """Test news article classification."""
        text = "Journalist reported that press release announced..."
        counts = classify_source_types(text)
        assert "news" in counts
        assert counts["news"] > 0

    def test_wiki_classification(self):
        """Test Wikipedia classification."""
        text = "According to Wikipedia, the encyclopedia states..."
        counts = classify_source_types(text)
        assert "wiki" in counts
        assert counts["wiki"] > 0

    def test_government_classification(self):
        """Test government source classification."""
        text = "Official government policy regulation from agency..."
        counts = classify_source_types(text)
        assert "government" in counts
        assert counts["government"] > 0

    def test_metadata_classification(self):
        """Test classification using metadata."""
        text = "Sample text"
        metadata = {"source_type": "patent"}
        counts = classify_source_types(text, metadata)
        assert "patent" in counts
        assert counts["patent"] >= 2  # Metadata adds +2


@pytest.mark.unit
@pytest.mark.ci_safe
class TestShannonEntropy:
    """Tests for calculate_shannon_entropy() function."""

    def test_entropy_single_source(self):
        """Test entropy of single source type is zero."""
        counts = {"patent": 10}
        entropy = calculate_shannon_entropy(counts)
        assert entropy == 0.0

    def test_entropy_two_equal_sources(self):
        """Test entropy of two equally weighted sources."""
        counts = {"patent": 5, "lab_notebook": 5}
        entropy = calculate_shannon_entropy(counts)
        # H = -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0
        assert abs(entropy - 1.0) < 0.01

    def test_entropy_diverse_sources(self):
        """Test entropy increases with source diversity."""
        # More diverse distribution
        diverse = {"a": 1, "b": 1, "c": 1, "d": 1, "e": 1}
        diverse_entropy = calculate_shannon_entropy(diverse)

        # Less diverse distribution
        skewed = {"a": 5, "b": 1}
        skewed_entropy = calculate_shannon_entropy(skewed)

        assert diverse_entropy > skewed_entropy

    def test_entropy_empty_counts(self):
        """Test entropy with empty counts returns zero."""
        counts = {}
        entropy = calculate_shannon_entropy(counts)
        assert entropy == 0.0

    def test_entropy_mathematical_correctness(self):
        """Test Shannon entropy formula: H = -Σ p_i * log2(p_i)."""
        counts = {"a": 2, "b": 3, "c": 5}
        total = 10

        entropy = calculate_shannon_entropy(counts)

        # Manual calculation
        expected = 0.0
        for count in counts.values():
            p = count / total
            expected -= p * math.log2(p)

        assert abs(entropy - expected) < 0.001


@pytest.mark.unit
@pytest.mark.ci_safe
class TestAuthorityWeightCalculation:
    """Tests for calculate_authority_weight() function."""

    def test_pre_1970_patent_low_authority(self):
        """Test that pre-1970 patents have low authority weight."""
        text = "United States Patent 2,345,678 filed March 1923."
        metadata = {}

        auth_weight, breakdown = calculate_authority_weight(text, metadata)

        # Should be in primary source range (0.0-0.30)
        assert 0.0 <= auth_weight <= 0.30
        assert breakdown["year"] == 1923

    def test_modern_wikipedia_high_authority(self):
        """Test that modern Wikipedia has high authority weight."""
        text = """
        According to Wikipedia, the scientific consensus is clear.
        Experts agree that this is widely accepted.
        Published 2024.
        """
        metadata = {"source": "Wikipedia"}
        
        auth_weight, breakdown = calculate_authority_weight(text, metadata)
        
        # Should be in high authority range (modern coordinated)
        assert 0.70 <= auth_weight <= 0.99

    def test_1956_lab_notebook_very_low_authority(self):
        """Test that 1956 lab notebook has very low authority weight."""
        text = """
        Laboratory notebook entry from 1956.
        Direct measurement and observation recorded.
        Original experimental log.
        """
        metadata = {}

        auth_weight, breakdown = calculate_authority_weight(text, metadata)

        # Should be very low (near 0)
        assert auth_weight < 0.10
        assert breakdown["year"] == 1956
        assert breakdown["primary_count"] > 0

    def test_academic_paper_medium_authority(self):
        """Test that academic papers have medium authority weight."""
        text = """
        Published in peer-reviewed journal, 2015.
        References: [1] Previous work (2010).
        """
        metadata = {}

        auth_weight, breakdown = calculate_authority_weight(text, metadata)

        # Should be in medium range
        assert 0.30 <= auth_weight <= 0.80

    def test_consensus_language_increases_authority(self):
        """Test that consensus language increases authority weight."""
        base_text = "Research from 2020 shows results."
        consensus_text = "Widely accepted scientific consensus from 2020 shows results."

        base_weight, _ = calculate_authority_weight(base_text, {})
        consensus_weight, _ = calculate_authority_weight(consensus_text, {})

        assert consensus_weight > base_weight

    def test_primary_markers_decrease_authority(self):
        """Test that primary source markers decrease authority weight."""
        base_text = "Research from 2020."
        primary_text = "Patent and lab notebook measurement from 2020."

        base_weight, _ = calculate_authority_weight(base_text, {})
        primary_weight, _ = calculate_authority_weight(primary_text, {})

        assert primary_weight < base_weight

    def test_authority_clamped_to_valid_range(self):
        """Test that authority weight is always in [0.0, 0.99]."""
        # Extreme primary source
        extreme_primary = "patent lab measurement archive scan diary letter " * 10

        auth_weight, _ = calculate_authority_weight(extreme_primary, {})

        assert 0.0 <= auth_weight <= 0.99

    def test_breakdown_includes_all_components(self):
        """Test that breakdown dict includes all expected components."""
        text = "Sample text from 1956 with patent reference."
        auth_weight, breakdown = calculate_authority_weight(text, {})

        expected_keys = [
            "citation_count",
            "citation_score",
            "institutional_score",
            "consensus_count",
            "consensus_score",
            "year",
            "age_adjustment",
            "primary_count",
            "primary_adjustment",
        ]

        for key in expected_keys:
            assert key in breakdown


@pytest.mark.unit
@pytest.mark.ci_safe
class TestProvenanceEntropyCalculation:
    """Tests for calculate_provenance_entropy() function."""

    def test_pre_1970_source_high_entropy(self):
        """Test that pre-1970 sources have high entropy base."""
        text = "Historical document from 1956 archive."
        metadata = {}

        prov_entropy, breakdown = calculate_provenance_entropy(text, metadata)

        # Should start with base of 5.5 bits for pre-1970
        assert prov_entropy >= 5.5
        assert breakdown["year"] == 1956
        assert breakdown["base_entropy"] == 5.5

    def test_post_1995_source_low_entropy(self):
        """Test that post-1995 sources have low entropy base."""
        text = "Modern article from 2020."
        metadata = {}

        prov_entropy, breakdown = calculate_provenance_entropy(text, metadata)

        # Should start with base of 1.5 bits for post-1995
        assert breakdown["base_entropy"] == 1.5

    def test_primary_markers_increase_entropy(self):
        """Test that primary source markers increase entropy."""
        base_text = "Document from 2000."
        primary_text = "Patent and lab notebook with measurement from 2000."

        base_entropy, _ = calculate_provenance_entropy(base_text, {})
        primary_entropy, _ = calculate_provenance_entropy(primary_text, {})

        assert primary_entropy > base_entropy

    def test_institutional_markers_decrease_entropy(self):
        """Test that institutional markers decrease entropy."""
        base_text = "Document from 2000."
        institutional_text = "WHO and government official document from 2000."

        base_entropy, _ = calculate_provenance_entropy(base_text, {})
        institutional_entropy, _ = calculate_provenance_entropy(institutional_text, {})

        assert institutional_entropy < base_entropy

    def test_consensus_phrases_decrease_entropy(self):
        """Test that consensus language decreases entropy."""
        base_text = "Research from 2000."
        consensus_text = "Widely accepted scientific consensus from 2000."

        base_entropy, _ = calculate_provenance_entropy(base_text, {})
        consensus_entropy, _ = calculate_provenance_entropy(consensus_text, {})

        assert consensus_entropy < base_entropy

    def test_scanned_document_increases_entropy(self):
        """Test that scanned documents increase entropy."""
        text = "Sample text"
        metadata_no_scan = {}
        metadata_with_scan = {"scan": "yes"}  # Detected via metadata
        
        entropy_no_scan, _ = calculate_provenance_entropy(text, metadata_no_scan)
        entropy_with_scan, _ = calculate_provenance_entropy(text, metadata_with_scan)
        
        # Having "scan" in metadata should increase entropy
        assert entropy_with_scan >= entropy_no_scan

    def test_entropy_non_negative(self):
        """Test that entropy is always non-negative."""
        # Even with many negative adjustments
        text = "WHO government official consensus widely accepted from 2024."
        metadata = {}

        prov_entropy, _ = calculate_provenance_entropy(text, metadata)

        assert prov_entropy >= 0.0

    def test_breakdown_includes_all_components(self):
        """Test that breakdown includes all expected components."""
        text = "Patent from 1956 archive."
        prov_entropy, breakdown = calculate_provenance_entropy(text, {})

        expected_keys = [
            "year",
            "base_entropy",
            "distribution_entropy",
            "source_counts",
            "primary_count",
            "primary_bonus",
            "variety_count",
            "variety_bonus",
            "institutional_penalty",
            "consensus_penalty",
        ]

        for key in expected_keys:
            assert key in breakdown


@pytest.mark.unit
@pytest.mark.ci_safe
class TestScoreDocument:
    """Tests for score_document() function."""

    def test_score_1923_patent_example(self):
        """Test scoring of 1923 patent (from docs example)."""
        text = """
        United States Patent 2,345,678
        Filed: March 15, 1923
        Inventor: Thomas Edison

        This patent describes an improved method for the measurement of
        electrical resistance in laboratory conditions. The experiment
        was conducted using primary measurement apparatus...
        """

        result = score_document(text)

        # Should have low authority (primary source)
        assert result.authority_weight < 0.30

        # Should have high entropy (pre-1970 primary source)
        assert result.provenance_entropy > 5.0

        # Should detect primary source markers
        assert result.primary_source_count > 0

    def test_score_2024_who_example(self):
        """Test scoring of 2024 WHO press release (from docs example)."""
        text = """
        According to Wikipedia and the World Health Organization (WHO),
        the scientific consensus is clear. Experts agree that this is
        a well-established fact supported by government guidelines.
        Studies show overwhelming evidence...
        """

        result = score_document(text)

        # Should have high authority (coordinated modern source)
        assert result.authority_weight >= 0.70

        # Should have low entropy (single coordinated source)
        assert result.provenance_entropy < 3.0

        # Should have high institutional score
        assert result.institutional_score > 0.20

        # Should detect consensus language
        assert result.consensus_score > 0

    def test_score_1956_lab_notebook_example(self):
        """Test scoring of 1956 lab notebook."""
        text = """
        Laboratory Notebook - June 1956
        Experiment: Measurement of reaction rates
        Observations recorded in field notes.
        Direct measurement apparatus used.
        Original research conducted.
        """

        result = score_document(text)

        # Should have very low authority
        assert result.authority_weight < 0.15

        # Should have very high entropy
        assert result.provenance_entropy > 7.0

        # Should detect many primary markers
        assert result.primary_source_count >= 3

    def test_result_is_scoring_result_dataclass(self):
        """Test that result is a ScoringResult dataclass."""
        text = "Sample text from 2020."
        result = score_document(text)

        assert isinstance(result, ScoringResult)
        assert hasattr(result, "authority_weight")
        assert hasattr(result, "provenance_entropy")
        assert hasattr(result, "citation_count")
        assert hasattr(result, "primary_source_count")
        assert hasattr(result, "institutional_score")
        assert hasattr(result, "consensus_score")
        assert hasattr(result, "source_type_distribution")


@pytest.mark.unit
@pytest.mark.ci_safe
class TestKnownSourceTypeScoring:
    """Tests for apply_known_source_type_scoring() function."""

    def test_known_patent_pre1970_scoring(self):
        """Test scoring with known source type 'patent_pre1970'."""
        text = "Patent document text."

        result = apply_known_source_type_scoring(text, "patent_pre1970")

        # Prior is (0.05, 7.0)
        # Should be close to prior (70% weight)
        assert result.authority_weight < 0.20
        assert result.provenance_entropy > 5.0

    def test_known_wiki_scoring(self):
        """Test scoring with known source type 'wiki'."""
        text = "Wikipedia article text."

        result = apply_known_source_type_scoring(text, "wiki")

        # Prior is (0.90, 1.0)
        # Should be close to prior (70% weight)
        assert result.authority_weight > 0.70
        assert result.provenance_entropy < 3.0

    def test_known_source_blending_ratio(self):
        """Test that blending is 70% prior, 30% dynamic."""
        text = "Sample text"
        source_type = "patent_pre1970"

        # Get dynamic score
        dynamic_result = score_document(text)

        # Get blended score
        blended_result = apply_known_source_type_scoring(text, source_type)

        # Prior for patent_pre1970 is (0.05, 7.0)
        # Verify blending (approximate check)
        # auth = 0.7 * 0.05 + 0.3 * dynamic_auth
        # If dynamic is different, blended should be weighted
        expected_auth_approx = 0.7 * 0.05 + 0.3 * dynamic_result.authority_weight
        expected_entropy_approx = 0.7 * 7.0 + 0.3 * dynamic_result.provenance_entropy

        # Allow some tolerance
        assert abs(blended_result.authority_weight - expected_auth_approx) < 0.15
        assert abs(blended_result.provenance_entropy - expected_entropy_approx) < 1.5

    def test_unknown_source_type_uses_default(self):
        """Test that unknown source type uses default prior."""
        text = "Sample text"

        result = apply_known_source_type_scoring(text, "unknown_type")

        # Default is (0.50, 3.0)
        # Should be close to default
        assert 0.30 <= result.authority_weight <= 0.70
        assert 2.0 <= result.provenance_entropy <= 4.5


@pytest.mark.unit
@pytest.mark.ci_safe
class TestScoreBatch:
    """Tests for score_batch() function."""

    def test_batch_processing_basic(self):
        """Test basic batch processing."""
        documents = [
            {"text": "Patent from 1923"},
            {"text": "WHO report from 2024"},
            {"text": "Lab notebook from 1956"},
        ]

        results = score_batch(documents)

        assert len(results) == 3
        for doc in results:
            assert "auth_weight" in doc
            assert "prov_entropy" in doc
            assert "scoring_method" in doc

    def test_batch_with_known_source_types(self):
        """Test batch processing with known source types."""
        documents = [
            {
                "text": "This is a patent document with sufficient text content for proper analysis and scoring to avoid default fallback behavior.",
                "source_type": "patent_pre1970"
            },
            {
                "text": "This is a Wikipedia article with sufficient text content for proper analysis and scoring to avoid default fallback behavior.",
                "source_type": "wiki"
            },
        ]
        
        results = score_batch(documents, use_known_source_type=True)
        
        # First should use hybrid scoring (patent)
        assert results[0]["scoring_method"] == "hybrid"
        assert results[0]["auth_weight"] < 0.30
        
        # Second should use hybrid scoring (wiki)
        assert results[1]["scoring_method"] == "hybrid"
        assert results[1]["auth_weight"] >= 0.60

    def test_batch_without_known_source_types(self):
        """Test batch processing without using known source types."""
        documents = [
            {
                "text": "This is sufficient text for dynamic scoring analysis without using the known source type field even though present.",
                "source_type": "patent_pre1970"
            },
            {
                "text": "This is sufficient text for dynamic scoring analysis without using the known source type field even though present.",
                "source_type": "wiki"
            },
        ]
        
        results = score_batch(documents, use_known_source_type=False)
        
        # Should use dynamic scoring even though source_type present
        assert all(doc["scoring_method"] == "dynamic" for doc in results)

    def test_batch_handles_short_documents(self):
        """Test that very short documents get default scores."""
        documents = [
            {"text": "Short"},  # Too short
            {"text": "This is a longer document with sufficient text for analysis."},
        ]

        results = score_batch(documents)

        # Short doc should get defaults
        assert results[0]["scoring_method"] == "default"
        assert results[0]["auth_weight"] == 0.50
        assert results[0]["prov_entropy"] == 3.0

        # Long doc should be scored
        assert results[1]["scoring_method"] in ["dynamic", "hybrid"]

    def test_batch_custom_text_field(self):
        """Test batch processing with custom text field name."""
        documents = [
            {"content": "Sample text for testing", "id": 1},
        ]

        results = score_batch(documents, text_field="content")

        assert len(results) == 1
        assert "auth_weight" in results[0]

    def test_batch_preserves_original_fields(self):
        """Test that batch processing preserves original document fields."""
        documents = [
            {"text": "Sample text", "id": 123, "category": "test"},
        ]

        results = score_batch(documents)

        assert results[0]["id"] == 123
        assert results[0]["category"] == "test"
        assert "auth_weight" in results[0]
        assert "prov_entropy" in results[0]


@pytest.mark.unit
@pytest.mark.ci_safe
class TestConstantsCompleteness:
    """Tests for completeness of marker and phrase constants."""

    def test_institutional_markers_not_empty(self):
        """Test that INSTITUTIONAL_MARKERS dict is not empty."""
        assert len(INSTITUTIONAL_MARKERS) > 0

    def test_consensus_phrases_not_empty(self):
        """Test that CONSENSUS_PHRASES list is not empty."""
        assert len(CONSENSUS_PHRASES) > 0

    def test_primary_source_markers_not_empty(self):
        """Test that PRIMARY_SOURCE_MARKERS list is not empty."""
        assert len(PRIMARY_SOURCE_MARKERS) > 0

    def test_institutional_markers_have_scores(self):
        """Test that all institutional markers have numeric scores."""
        for marker, score in INSTITUTIONAL_MARKERS.items():
            assert isinstance(marker, str)
            assert isinstance(score, (int, float))
            assert 0.0 <= score <= 1.0

    def test_high_authority_institutions_present(self):
        """Test that key high-authority institutions are in markers."""
        expected_institutions = ["who", "cdc", "nature", "science", ".gov"]

        for inst in expected_institutions:
            assert inst in INSTITUTIONAL_MARKERS

    def test_consensus_phrases_coverage(self):
        """Test that key consensus phrases are present."""
        expected_phrases = [
            "widely accepted",
            "experts agree",
            "scientific consensus",
            "established fact",
        ]

        for phrase in expected_phrases:
            assert phrase in CONSENSUS_PHRASES

    def test_primary_markers_coverage(self):
        """Test that key primary source markers are present."""
        expected_markers = [
            "patent",
            "lab notebook",
            "measurement",
            "observation",
            "archive",
            "primary source",
        ]

        for marker in expected_markers:
            assert marker in PRIMARY_SOURCE_MARKERS

