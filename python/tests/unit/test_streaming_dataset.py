"""Unit tests for StreamingDataset."""

import json
import pytest
import tempfile
from pathlib import Path
from src.data.streaming_dataset import StreamingDataset


@pytest.fixture
def temp_jsonl_file():
    """Create a temporary JSONL file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        data = [
            {"id": 1, "text": "first sample"},
            {"id": 2, "text": "second sample"},
            {"id": 3, "text": "third sample"},
            {"id": 4, "text": "fourth sample"},
            {"id": 5, "text": "fifth sample"},
        ]
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_multi_jsonl_files():
    """Create multiple temporary JSONL files for testing."""
    files = []
    for file_num in range(2):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(3):
                sample_id = file_num * 3 + i + 1
                f.write(json.dumps({"id": sample_id, "text": f"sample {sample_id}"}) + "\n")
            files.append(f.name)

    yield files

    # Cleanup
    for f in files:
        Path(f).unlink(missing_ok=True)


@pytest.fixture
def temp_corrupted_jsonl_file():
    """Create JSONL file with some corrupted lines."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"id": 1, "text": "valid"}) + "\n")
        f.write("{invalid json\n")  # Corrupted line
        f.write(json.dumps({"id": 2, "text": "also valid"}) + "\n")
        f.write("not json at all\n")  # Corrupted line
        f.write(json.dumps({"id": 3, "text": "valid again"}) + "\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.mark.unit
def test_basic_iteration(temp_jsonl_file):
    """Test basic streaming iteration."""
    dataset = StreamingDataset([temp_jsonl_file], batch_size=2)

    batches = list(dataset)

    # Should have 3 batches: [2, 2, 1]
    assert len(batches) == 3
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 1

    # Check data integrity
    assert batches[0][0]["id"] == 1
    assert batches[0][1]["id"] == 2
    assert batches[2][0]["id"] == 5


@pytest.mark.unit
def test_shuffling_deterministic(temp_jsonl_file):
    """Test shuffling with seed produces same order."""
    dataset1 = StreamingDataset([temp_jsonl_file], batch_size=10, shuffle=True, seed=42)
    dataset2 = StreamingDataset([temp_jsonl_file], batch_size=10, shuffle=True, seed=42)

    batches1 = [sample for batch in dataset1 for sample in batch]
    batches2 = [sample for batch in dataset2 for sample in batch]

    # Same seed should produce same order
    assert batches1 == batches2

    # But order should be different from non-shuffled
    dataset3 = StreamingDataset([temp_jsonl_file], batch_size=10, shuffle=False)
    batches3 = [sample for batch in dataset3 for sample in batch]

    # Check that shuffling actually changed order (very unlikely to be same)
    ids1 = [s["id"] for s in batches1]
    ids3 = [s["id"] for s in batches3]
    assert ids1 != ids3


@pytest.mark.unit
def test_corrupted_lines_handled(temp_corrupted_jsonl_file, caplog):
    """Test that corrupted JSON lines are skipped with warning."""
    dataset = StreamingDataset([temp_corrupted_jsonl_file], batch_size=10)

    batches = [sample for batch in dataset for sample in batch]

    # Should get only 3 valid samples
    assert len(batches) == 3
    assert batches[0]["id"] == 1
    assert batches[1]["id"] == 2
    assert batches[2]["id"] == 3

    # Should have warnings in log
    assert "corrupted" in caplog.text.lower() or "invalid" in caplog.text.lower()


@pytest.mark.unit
def test_multi_file_streaming(temp_multi_jsonl_files):
    """Test streaming from multiple files."""
    dataset = StreamingDataset(temp_multi_jsonl_files, batch_size=2)

    batches = [sample for batch in dataset for sample in batch]

    # Should have 6 samples total (3 per file)
    assert len(batches) == 6

    # Check samples are from both files
    ids = [s["id"] for s in batches]
    assert set(ids) == {1, 2, 3, 4, 5, 6}


@pytest.mark.unit
def test_progress_tracking(temp_jsonl_file):
    """Test progress tracking functionality."""
    dataset = StreamingDataset([temp_jsonl_file], batch_size=2)

    # Initial progress
    progress = dataset.get_progress()
    assert progress["current_position"] == 0

    # After first batch
    next(dataset)
    progress = dataset.get_progress()
    assert progress["current_position"] == 2
    assert temp_jsonl_file in progress["current_file"]

    # Check progress percentage is calculated
    if progress["total_samples"]:
        assert 0 <= progress["progress_percent"] <= 100


@pytest.mark.unit
def test_estimate_total_samples(temp_jsonl_file):
    """Test total sample estimation."""
    dataset = StreamingDataset([temp_jsonl_file], batch_size=2)

    total = dataset.estimate_total_samples()

    # Should estimate close to actual (5 samples)
    assert total is not None
    assert 4 <= total <= 6  # Allow some estimation error


@pytest.mark.unit
def test_context_manager(temp_jsonl_file):
    """Test context manager protocol."""
    with StreamingDataset([temp_jsonl_file], batch_size=2) as dataset:
        batches = list(dataset)
        assert len(batches) > 0

    # File should be closed after exiting context


@pytest.mark.unit
def test_invalid_batch_size(temp_jsonl_file):
    """Test validation of batch_size parameter."""
    with pytest.raises(ValueError):
        StreamingDataset([temp_jsonl_file], batch_size=0)

    with pytest.raises(ValueError):
        StreamingDataset([temp_jsonl_file], batch_size=-1)


@pytest.mark.unit
def test_invalid_buffer_size(temp_jsonl_file):
    """Test validation of buffer_size parameter."""
    with pytest.raises(ValueError):
        StreamingDataset([temp_jsonl_file], batch_size=10, buffer_size=5)


@pytest.mark.unit
def test_nonexistent_file():
    """Test handling of missing files."""
    with pytest.raises(FileNotFoundError):
        StreamingDataset(["nonexistent_file.jsonl"], batch_size=10)


@pytest.mark.unit
def test_cycling_behavior(temp_jsonl_file):
    """Test that cycle=True loops indefinitely."""
    dataset = StreamingDataset([temp_jsonl_file], batch_size=2, cycle=True)

    # Iterate more than file contains
    batches = []
    for i, batch in enumerate(dataset):
        batches.append(batch)
        if i >= 10:  # More than 3 batches in file
            break

    # Should have gotten more batches than file contains
    assert len(batches) > 3


@pytest.mark.unit
def test_reset_functionality(temp_jsonl_file):
    """Test reset() returns iterator to start."""
    dataset = StreamingDataset([temp_jsonl_file], batch_size=2)

    # Get first batch
    first_batch = next(dataset)

    # Reset
    dataset.reset()

    # Get first batch again
    first_batch_after_reset = next(dataset)

    # Should be same data
    assert first_batch == first_batch_after_reset
