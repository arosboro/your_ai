"""Integration tests for data preparation pipeline.

Tests the data preparation workflow including JSONL parsing,
authority/entropy assignment, and train/val split logic.
"""

import pytest
import json
import tempfile
from pathlib import Path
from src.citation_scorer import score_batch
from src.metrics import compute_metrics_for_example, validate_dataset_metrics


@pytest.fixture
def temp_raw_data_dir():
    """Create temporary directory with REAL examples from documentation.

    Uses actual example texts from:
    - citation_scorer.py docstring examples
    - docs/ALGORITHM.md examples
    - Real dataset samples
    """
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    # REAL example 1: From citation_scorer.py test section (lines 620-628)
    # Low authority example (patent)
    patent_data = [
        {
            "text": """United States Patent 2,345,678
Filed: March 15, 1923
Inventor: Thomas Edison

This patent describes an improved method for the measurement of
electrical resistance in laboratory conditions. The experiment
was conducted using primary measurement apparatus...""",
            "source_type": "patent_pre1970",
            "year": 1923,
            "identifier": "patent_1923_edison",
        },
        {
            "text": """United States Patent 3,456,789
Filed: June 12, 1956
Laboratory Notebook Entry

Direct measurement of electromagnetic properties using original
experimental apparatus. Field notes and observations recorded.""",
            "source_type": "patent_pre1970",
            "year": 1956,
            "identifier": "patent_1956_lab",
        },
    ]

    # REAL example 2: From citation_scorer.py test section (lines 629-635)
    # High authority example (modern consensus)
    wiki_data = [
        {
            "text": """According to Wikipedia and the World Health Organization (WHO),
the scientific consensus is clear. Experts agree that this is
a well-established fact supported by government guidelines.
Studies show overwhelming evidence...""",
            "source_type": "wiki",
            "year": 2024,
            "identifier": "wiki_2024_who",
        },
        {
            "text": """Wikipedia article on the topic. The widely accepted view
among researchers is that this represents scientific consensus.
Government agencies and Nature journal support this position.""",
            "source_type": "wiki",
            "year": 2023,
            "identifier": "wiki_2023_nature",
        },
    ]

    # REAL example 3: From citation_scorer.py test section (lines 636-641)
    # Medium authority example (academic)
    academic_data = [
        {
            "text": """Abstract: This paper presents results from our experimental study.
Introduction: We measured various parameters under controlled conditions.
Methodology: Samples were collected and analyzed using standard protocols.
Results: Our findings indicate significant correlations.
References: [1] Smith et al., 2019. [2] Jones, 2020.""",
            "source_type": "academic_paper",
            "year": 2015,
            "identifier": "academic_2015_experimental",
        },
    ]

    # Write files
    with open(temp_path / "patents.jsonl", "w") as f:
        for item in patent_data:
            f.write(json.dumps(item) + "\n")

    with open(temp_path / "wiki.jsonl", "w") as f:
        for item in wiki_data:
            f.write(json.dumps(item) + "\n")

    with open(temp_path / "academic.jsonl", "w") as f:
        for item in academic_data:
            f.write(json.dumps(item) + "\n")

    yield temp_path

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for output files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)

    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.ci_safe  # Pure Python file I/O
class TestJSONLParsing:
    """Tests for JSONL file parsing."""

    def test_parse_single_jsonl_file(self, temp_raw_data_dir):
        """Test parsing a single JSONL file."""
        patent_file = temp_raw_data_dir / "patents.jsonl"

        documents = []
        with open(patent_file, "r") as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc)

        assert len(documents) == 2
        assert all("text" in doc for doc in documents)
        assert all("identifier" in doc for doc in documents)

    def test_parse_multiple_jsonl_files(self, temp_raw_data_dir):
        """Test parsing multiple JSONL files."""
        files = list(temp_raw_data_dir.glob("*.jsonl"))

        all_documents = []
        for file in files:
            with open(file, "r") as f:
                for line in f:
                    doc = json.loads(line)
                    all_documents.append(doc)

        # Should have 5 total documents (2 + 2 + 1)
        assert len(all_documents) == 5

    def test_validate_required_fields(self, temp_raw_data_dir):
        """Test that all documents have required fields."""
        files = list(temp_raw_data_dir.glob("*.jsonl"))

        required_fields = ["text", "identifier"]

        for file in files:
            with open(file, "r") as f:
                for line in f:
                    doc = json.loads(line)
                    for field in required_fields:
                        assert field in doc, f"Missing field {field} in {file}"


@pytest.mark.integration
@pytest.mark.ci_safe
class TestAuthorityEntropyAssignment:
    """Tests for authority/entropy assignment to samples."""

    def test_assign_scores_to_documents(self, temp_raw_data_dir):
        """Test assigning authority and entropy scores to documents."""
        patent_file = temp_raw_data_dir / "patents.jsonl"

        documents = []
        with open(patent_file, "r") as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc)

        # Assign scores using citation_scorer
        scored_docs = score_batch(documents, use_known_source_type=True)

        assert len(scored_docs) == len(documents)
        for doc in scored_docs:
            assert "auth_weight" in doc
            assert "prov_entropy" in doc
            assert 0.0 <= doc["auth_weight"] <= 0.99
            assert doc["prov_entropy"] >= 0.0

    def test_primary_sources_low_authority(self, temp_raw_data_dir):
        """Test that primary sources get low authority scores."""
        patent_file = temp_raw_data_dir / "patents.jsonl"

        documents = []
        with open(patent_file, "r") as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc)

        scored_docs = score_batch(documents, use_known_source_type=True)

        # Patents should have low authority
        for doc in scored_docs:
            assert doc["auth_weight"] < 0.50

    def test_wiki_sources_high_authority(self, temp_raw_data_dir):
        """Test that Wikipedia sources get high authority scores."""
        wiki_file = temp_raw_data_dir / "wiki.jsonl"

        documents = []
        with open(wiki_file, "r") as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc)

        scored_docs = score_batch(documents, use_known_source_type=True)

        # Wiki should have high authority
        for doc in scored_docs:
            assert doc["auth_weight"] > 0.60

    def test_primary_sources_high_entropy(self, temp_raw_data_dir):
        """Test that primary sources get high entropy scores."""
        patent_file = temp_raw_data_dir / "patents.jsonl"

        documents = []
        with open(patent_file, "r") as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc)

        scored_docs = score_batch(documents, use_known_source_type=True)

        # Pre-1970 patents should have high entropy
        for doc in scored_docs:
            assert doc["prov_entropy"] > 4.0

    def test_scoring_preserves_original_fields(self, temp_raw_data_dir):
        """Test that scoring preserves original document fields."""
        patent_file = temp_raw_data_dir / "patents.jsonl"

        documents = []
        with open(patent_file, "r") as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc)

        scored_docs = score_batch(documents)

        # Original fields should be preserved
        for i, doc in enumerate(scored_docs):
            assert doc["text"] == documents[i]["text"]
            assert doc["identifier"] == documents[i]["identifier"]
            assert doc["year"] == documents[i]["year"]


@pytest.mark.integration
@pytest.mark.ci_safe
class TestTrainValSplit:
    """Tests for train/val split logic."""

    def test_split_maintains_total_count(self, temp_raw_data_dir):
        """Test that split maintains total document count."""
        files = list(temp_raw_data_dir.glob("*.jsonl"))

        all_documents = []
        for file in files:
            with open(file, "r") as f:
                for line in f:
                    doc = json.loads(line)
                    all_documents.append(doc)

        total = len(all_documents)

        # Split 80/20
        train_size = int(total * 0.8)

        import random

        random.seed(42)
        random.shuffle(all_documents)

        train_docs = all_documents[:train_size]
        val_docs = all_documents[train_size:]

        assert len(train_docs) + len(val_docs) == total

    def test_split_no_overlap(self, temp_raw_data_dir):
        """Test that train and val sets don't overlap."""
        files = list(temp_raw_data_dir.glob("*.jsonl"))

        all_documents = []
        for file in files:
            with open(file, "r") as f:
                for line in f:
                    doc = json.loads(line)
                    all_documents.append(doc)

        # Split
        import random

        random.seed(42)
        random.shuffle(all_documents)

        total = len(all_documents)
        train_size = int(total * 0.8)

        train_docs = all_documents[:train_size]
        val_docs = all_documents[train_size:]

        # Extract identifiers
        train_ids = {doc["identifier"] for doc in train_docs}
        val_ids = {doc["identifier"] for doc in val_docs}

        # No overlap
        assert len(train_ids & val_ids) == 0

    def test_split_preserves_distribution(self, temp_raw_data_dir):
        """Test that split roughly preserves source type distribution."""
        files = list(temp_raw_data_dir.glob("*.jsonl"))

        all_documents = []
        for file in files:
            with open(file, "r") as f:
                for line in f:
                    doc = json.loads(line)
                    all_documents.append(doc)

        # Count source types
        from collections import Counter

        total_types = Counter(doc["source_type"] for doc in all_documents)

        # Split with stratification (simplified)
        import random

        random.seed(42)
        random.shuffle(all_documents)

        train_size = int(len(all_documents) * 0.8)
        train_docs = all_documents[:train_size]

        train_types = Counter(doc["source_type"] for doc in train_docs)

        # Distribution should be roughly similar
        # (With only 5 samples, exact stratification not possible)
        # Ensure all source types are present in training split
        for source_type in total_types:
            assert source_type in train_types, (
                f"Source type '{source_type}' missing from training split"
            )
            assert train_types[source_type] > 0, (
                f"Source type '{source_type}' has 0 count in training split"
            )

            # Check proportion is preserved within tolerance
            total_prop = total_types[source_type] / len(all_documents)
            train_prop = train_types[source_type] / len(train_docs)
            tolerance = 0.3  # Allow 30% deviation given small sample size
            assert abs(train_prop - total_prop) < tolerance, (
                f"Source type '{source_type}' proportion mismatch: "
                f"total={total_prop:.2f}, train={train_prop:.2f}, diff={abs(train_prop - total_prop):.2f}"
            )


@pytest.mark.integration
@pytest.mark.ci_safe
class TestOutputFileFormat:
    """Tests for output file format."""

    def test_write_jsonl_format(self, temp_raw_data_dir, temp_output_dir):
        """Test writing documents to JSONL format."""
        patent_file = temp_raw_data_dir / "patents.jsonl"

        documents = []
        with open(patent_file, "r") as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc)

        # Score documents
        scored_docs = score_batch(documents, use_known_source_type=True)

        # Write to output
        output_file = temp_output_dir / "train.jsonl"
        with open(output_file, "w") as f:
            for doc in scored_docs:
                f.write(json.dumps(doc) + "\n")

        # Verify output
        assert output_file.exists()

        # Read back
        read_docs = []
        with open(output_file, "r") as f:
            for line in f:
                doc = json.loads(line)
                read_docs.append(doc)

        assert len(read_docs) == len(scored_docs)

    def test_output_has_required_fields(self, temp_raw_data_dir, temp_output_dir):
        """Test that output documents have all required fields."""
        patent_file = temp_raw_data_dir / "patents.jsonl"

        documents = []
        with open(patent_file, "r") as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc)

        scored_docs = score_batch(documents, use_known_source_type=True)

        # Write output
        output_file = temp_output_dir / "train.jsonl"
        with open(output_file, "w") as f:
            for doc in scored_docs:
                f.write(json.dumps(doc) + "\n")

        # Verify all required fields present
        required_fields = ["text", "auth_weight", "prov_entropy"]

        with open(output_file, "r") as f:
            for line in f:
                doc = json.loads(line)
                for field in required_fields:
                    assert field in doc, f"Missing required field: {field}"

    def test_separate_train_val_files(self, temp_raw_data_dir, temp_output_dir):
        """Test creating separate train and val files."""
        files = list(temp_raw_data_dir.glob("*.jsonl"))

        all_documents = []
        for file in files:
            with open(file, "r") as f:
                for line in f:
                    doc = json.loads(line)
                    all_documents.append(doc)

        # Score all documents
        scored_docs = score_batch(all_documents, use_known_source_type=True)

        # Split
        import random

        random.seed(42)
        random.shuffle(scored_docs)

        train_size = int(len(scored_docs) * 0.8)
        train_docs = scored_docs[:train_size]
        val_docs = scored_docs[train_size:]

        # Write separate files
        train_file = temp_output_dir / "train.jsonl"
        val_file = temp_output_dir / "val.jsonl"

        with open(train_file, "w") as f:
            for doc in train_docs:
                f.write(json.dumps(doc) + "\n")

        with open(val_file, "w") as f:
            for doc in val_docs:
                f.write(json.dumps(doc) + "\n")

        # Verify both exist
        assert train_file.exists()
        assert val_file.exists()

        # Verify counts
        train_count = sum(1 for _ in open(train_file))
        val_count = sum(1 for _ in open(val_file))

        assert train_count + val_count == len(scored_docs)


@pytest.mark.integration
@pytest.mark.ci_safe
class TestDatasetValidation:
    """Tests for dataset validation after preparation."""

    def test_validate_prepared_dataset(self, temp_raw_data_dir):
        """Test validating a prepared dataset."""
        files = list(temp_raw_data_dir.glob("*.jsonl"))

        all_documents = []
        for file in files:
            with open(file, "r") as f:
                for line in f:
                    doc = json.loads(line)
                    all_documents.append(doc)

        # Score documents
        scored_docs = score_batch(all_documents, use_known_source_type=True)

        # Validate
        stats = validate_dataset_metrics(scored_docs, text_field="text")

        assert "total_examples" in stats
        assert stats["total_examples"] == len(scored_docs)
        assert "auth_weight" in stats
        assert "prov_entropy" in stats

    def test_validation_detects_imbalance(self, temp_raw_data_dir):
        """Test that validation detects imbalanced datasets."""
        # Create imbalanced dataset (all high-authority)
        wiki_file = temp_raw_data_dir / "wiki.jsonl"

        documents = []
        with open(wiki_file, "r") as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc)

        # Duplicate to have more samples
        documents = documents * 10

        scored_docs = score_batch(documents, use_known_source_type=True)

        stats = validate_dataset_metrics(scored_docs, text_field="text")

        # Should have warnings about insufficient low-authority sources
        assert len(stats["warnings"]) > 0

    def test_validation_reports_statistics(self, temp_raw_data_dir):
        """Test that validation reports statistics."""
        files = list(temp_raw_data_dir.glob("*.jsonl"))

        all_documents = []
        for file in files:
            with open(file, "r") as f:
                for line in f:
                    doc = json.loads(line)
                    all_documents.append(doc)

        scored_docs = score_batch(all_documents, use_known_source_type=True)

        stats = validate_dataset_metrics(scored_docs, text_field="text")

        # Should have mean, std, min, max
        assert "mean" in stats["auth_weight"]
        assert "std" in stats["auth_weight"]
        assert "min" in stats["auth_weight"]
        assert "max" in stats["auth_weight"]


@pytest.mark.integration
@pytest.mark.ci_safe
class TestEndToEndDataPipeline:
    """End-to-end tests for complete data preparation pipeline."""

    def test_full_pipeline(self, temp_raw_data_dir, temp_output_dir):
        """Test complete pipeline from raw data to train/val files."""
        # 1. Load raw data
        files = list(temp_raw_data_dir.glob("*.jsonl"))
        all_documents = []

        for file in files:
            with open(file, "r") as f:
                for line in f:
                    doc = json.loads(line)
                    all_documents.append(doc)

        # 2. Score documents
        scored_docs = score_batch(all_documents, use_known_source_type=True)

        # 3. Shuffle and split
        import random

        random.seed(42)
        random.shuffle(scored_docs)

        train_size = int(len(scored_docs) * 0.8)
        train_docs = scored_docs[:train_size]
        val_docs = scored_docs[train_size:]

        # 4. Write output files
        train_file = temp_output_dir / "train.jsonl"
        val_file = temp_output_dir / "val.jsonl"

        with open(train_file, "w") as f:
            for doc in train_docs:
                f.write(json.dumps(doc) + "\n")

        with open(val_file, "w") as f:
            for doc in val_docs:
                f.write(json.dumps(doc) + "\n")

        # 5. Validate outputs
        assert train_file.exists()
        assert val_file.exists()

        # Verify all documents processed
        train_count = sum(1 for _ in open(train_file))
        val_count = sum(1 for _ in open(val_file))
        assert train_count + val_count == len(all_documents)

        # Verify required fields
        with open(train_file, "r") as f:
            for line in f:
                doc = json.loads(line)
                assert "text" in doc
                assert "auth_weight" in doc
                assert "prov_entropy" in doc

    def test_pipeline_handles_diverse_sources(self, temp_raw_data_dir, temp_output_dir):
        """Test pipeline with diverse source types."""
        files = list(temp_raw_data_dir.glob("*.jsonl"))
        all_documents = []

        for file in files:
            with open(file, "r") as f:
                for line in f:
                    doc = json.loads(line)
                    all_documents.append(doc)

        # Verify we have diverse source types
        source_types = {doc["source_type"] for doc in all_documents}
        assert len(source_types) >= 3  # patents, wiki, academic

        # Score and verify diversity preserved
        scored_docs = score_batch(all_documents, use_known_source_type=True)

        # Check authority range spans low to high
        auth_weights = [doc["auth_weight"] for doc in scored_docs]
        assert min(auth_weights) < 0.30  # Has primary sources
        assert max(auth_weights) > 0.70  # Has modern sources


@pytest.mark.integration
@pytest.mark.ci_safe
class TestErrorHandling:
    """Tests for error handling in data preparation."""

    def test_handle_malformed_json(self, temp_output_dir):
        """Test handling of malformed JSON lines."""
        # Create file with malformed JSON
        bad_file = temp_output_dir / "bad.jsonl"
        with open(bad_file, "w") as f:
            f.write('{"text": "valid json"}\n')
            f.write("{invalid json here\n")  # Malformed
            f.write('{"text": "another valid"}\n')

        # Try to parse
        documents = []
        errors = 0

        with open(bad_file, "r") as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    documents.append(doc)
                except json.JSONDecodeError:
                    errors += 1

        # Should skip malformed line
        assert len(documents) == 2
        assert errors == 1

    def test_handle_missing_fields(self, temp_output_dir):
        """Test handling of documents with missing fields."""
        # Create file with incomplete documents
        incomplete_file = temp_output_dir / "incomplete.jsonl"
        with open(incomplete_file, "w") as f:
            f.write('{"text": "complete doc", "year": 2020}\n')
            f.write('{"text": "missing year"}\n')  # No year field
            f.write('{"year": 2020}\n')  # No text field

        documents = []
        with open(incomplete_file, "r") as f:
            for line in f:
                doc = json.loads(line)
                # Only keep if has text field
                if "text" in doc and doc["text"]:
                    documents.append(doc)

        assert len(documents) == 2  # Skip the one without text

    def test_handle_empty_text(self, temp_output_dir):
        """Test handling of documents with empty text."""
        empty_file = temp_output_dir / "empty.jsonl"
        with open(empty_file, "w") as f:
            f.write('{"text": "valid text", "year": 2020}\n')
            f.write('{"text": "", "year": 2020}\n')  # Empty text
            f.write('{"text": "   ", "year": 2020}\n')  # Whitespace only

        documents = []
        with open(empty_file, "r") as f:
            for line in f:
                doc = json.loads(line)
                # Filter out empty/whitespace-only text
                if doc.get("text", "").strip():
                    documents.append(doc)

        assert len(documents) == 1
