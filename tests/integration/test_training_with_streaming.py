"""Integration test for streaming-based training."""
import json
import pytest
import tempfile
import psutil
import os
from pathlib import Path
from src.data.streaming_dataset import StreamingDataset
from src.config import Config


@pytest.fixture
def large_jsonl_dataset():
    """Create a large JSONL dataset for memory testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        # Create 10k samples (should be manageable but test streaming)
        for i in range(10000):
            sample = {
                "id": i,
                "text": f"This is sample {i} with some text content to make it realistic.",
                "authority_weight": 0.5,
                "provenance_entropy": 2.0
            }
            f.write(json.dumps(sample) + '\n')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.mark.integration
@pytest.mark.slow
def test_streaming_memory_bounded(large_jsonl_dataset):
    """Test that streaming keeps memory usage bounded even with large dataset."""
    process = psutil.Process(os.getpid())
    
    # Measure baseline memory
    baseline_memory_mb = process.memory_info().rss / 1024 / 1024
    
    # Stream through large dataset
    config = Config()
    config.performance.use_streaming = True
    config.performance.streaming_buffer_size = 100
    
    dataset = StreamingDataset(
        [large_jsonl_dataset],
        batch_size=32,
        buffer_size=config.performance.streaming_buffer_size
    )
    
    max_memory_mb = baseline_memory_mb
    batch_count = 0
    
    for batch in dataset:
        batch_count += 1
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        max_memory_mb = max(max_memory_mb, current_memory_mb)
        
        # Process batch (simulate training)
        assert len(batch) > 0
        assert all("text" in sample for sample in batch)
    
    # Calculate memory increase
    memory_increase_mb = max_memory_mb - baseline_memory_mb
    
    # Memory increase should be bounded (< 100MB for streaming with small buffer)
    # If loading entire dataset, would be much higher
    assert memory_increase_mb < 100, f"Memory increased by {memory_increase_mb}MB, expected < 100MB"
    
    # Should have processed all batches
    expected_batches = 10000 // 32 + (1 if 10000 % 32 else 0)
    assert batch_count == expected_batches


@pytest.mark.integration
def test_streaming_with_config(large_jsonl_dataset):
    """Test streaming respects configuration settings."""
    config = Config()
    config.performance.use_streaming = True
    config.performance.streaming_buffer_size = 500
    
    dataset = StreamingDataset(
        [large_jsonl_dataset],
        batch_size=16,
        buffer_size=config.performance.streaming_buffer_size,
        shuffle=True,
        seed=config.seed
    )
    
    batches = []
    for i, batch in enumerate(dataset):
        batches.append(batch)
        if i >= 10:  # Just test first few batches
            break
    
    # Should have gotten batches
    assert len(batches) == 11
    assert all(len(batch) <= 16 for batch in batches)


@pytest.mark.integration
def test_streaming_progress_reporting(large_jsonl_dataset):
    """Test progress reporting during streaming."""
    dataset = StreamingDataset([large_jsonl_dataset], batch_size=100)
    
    progress_snapshots = []
    
    for i, batch in enumerate(dataset):
        if i % 10 == 0:  # Check progress every 10 batches
            progress = dataset.get_progress()
            progress_snapshots.append(progress)
        
        if i >= 50:  # Process 50 batches
            break
    
    # Progress should increase
    positions = [p["current_position"] for p in progress_snapshots]
    assert positions == sorted(positions)  # Monotonically increasing
    
    # Should have reasonable progress percentage
    if progress_snapshots[-1]["progress_percent"]:
        assert progress_snapshots[-1]["progress_percent"] > 0
