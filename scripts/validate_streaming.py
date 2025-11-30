#!/usr/bin/env python3
"""
Quick validation script for US1 - Streaming implementation.

This script validates that the streaming dataset works correctly
without requiring a full training run.
"""

import json
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.streaming_dataset import StreamingDataset
from data.batch_buffer import BatchBuffer
from config import Config


def create_test_dataset(num_samples=100):
    """Create a temporary JSONL test dataset."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(num_samples):
            sample = {
                "id": i,
                "text": f"Sample text {i} for testing streaming dataset functionality.",
                "auth_weight": 0.5 + (i % 10) * 0.05,
                "prov_entropy": 2.0 + (i % 5) * 0.3
            }
            f.write(json.dumps(sample) + '\n')
        return f.name


def test_streaming_basic():
    """Test basic streaming functionality."""
    print("=" * 60)
    print("TEST 1: Basic Streaming")
    print("=" * 60)
    
    test_file = create_test_dataset(100)
    
    try:
        dataset = StreamingDataset(
            file_paths=[test_file],
            batch_size=10,
            buffer_size=50,
            shuffle=False
        )
        
        batch_count = 0
        sample_count = 0
        
        for batch in dataset:
            batch_count += 1
            sample_count += len(batch)
            
            if batch_count == 1:
                print(f"✓ First batch received: {len(batch)} samples")
                print(f"  Sample IDs: {[s['id'] for s in batch[:3]]}...")
        
        print(f"✓ Total batches: {batch_count}")
        print(f"✓ Total samples: {sample_count}")
        print(f"✓ Expected: 10 batches, 100 samples")
        
        assert batch_count == 10, f"Expected 10 batches, got {batch_count}"
        assert sample_count == 100, f"Expected 100 samples, got {sample_count}"
        
        print("✅ PASSED: Basic streaming works correctly\n")
        return True
        
    finally:
        Path(test_file).unlink(missing_ok=True)


def test_streaming_shuffled():
    """Test shuffled streaming with deterministic seed."""
    print("=" * 60)
    print("TEST 2: Deterministic Shuffling")
    print("=" * 60)
    
    test_file = create_test_dataset(50)
    
    try:
        # Two datasets with same seed
        dataset1 = StreamingDataset([test_file], batch_size=50, shuffle=True, seed=42)
        dataset2 = StreamingDataset([test_file], batch_size=50, shuffle=True, seed=42)
        
        batch1 = next(iter(dataset1))
        batch2 = next(iter(dataset2))
        
        ids1 = [s['id'] for s in batch1]
        ids2 = [s['id'] for s in batch2]
        
        print(f"✓ Dataset 1 IDs: {ids1[:10]}...")
        print(f"✓ Dataset 2 IDs: {ids2[:10]}...")
        
        assert ids1 == ids2, "Same seed should produce same order"
        assert ids1 != list(range(50)), "Shuffled order should differ from original"
        
        print("✅ PASSED: Deterministic shuffling works correctly\n")
        return True
        
    finally:
        Path(test_file).unlink(missing_ok=True)


def test_streaming_progress():
    """Test progress tracking."""
    print("=" * 60)
    print("TEST 3: Progress Tracking")
    print("=" * 60)
    
    test_file = create_test_dataset(100)
    
    try:
        dataset = StreamingDataset([test_file], batch_size=10)
        
        # Get initial progress
        progress = dataset.get_progress()
        print(f"✓ Initial position: {progress['current_position']}")
        
        # Process first batch
        next(iter(dataset))
        progress = dataset.get_progress()
        print(f"✓ After 1 batch: position={progress['current_position']}")
        
        if progress['progress_percent']:
            print(f"✓ Progress: {progress['progress_percent']:.1f}%")
        
        assert progress['current_position'] == 10, "Should have processed 10 samples"
        
        print("✅ PASSED: Progress tracking works correctly\n")
        return True
        
    finally:
        Path(test_file).unlink(missing_ok=True)


def test_batch_buffer():
    """Test BatchBuffer allocation."""
    print("=" * 60)
    print("TEST 4: BatchBuffer")
    print("=" * 60)
    
    try:
        import mlx.core as mx
        
        buffer = BatchBuffer(batch_size=4, max_seq_length=128)
        
        print(f"✓ Buffer shape: {buffer.input_ids.shape}")
        print(f"✓ Buffer dtype: {buffer.input_ids.dtype}")
        
        assert buffer.input_ids.shape == (4, 128), "Buffer shape incorrect"
        
        # Test view
        view = buffer.get_view(2)
        print(f"✓ View shape: {view['input_ids'].shape}")
        
        assert view['input_ids'].shape == (2, 128), "View shape incorrect"
        
        print("✅ PASSED: BatchBuffer works correctly\n")
        return True
        
    except ImportError:
        print("⚠️  SKIPPED: MLX not available (expected on non-Mac systems)\n")
        return True


def test_config():
    """Test PerformanceConfig."""
    print("=" * 60)
    print("TEST 5: Configuration")
    print("=" * 60)
    
    config = Config()
    
    print(f"✓ Streaming enabled: {config.performance.use_streaming}")
    print(f"✓ Buffer size: {config.performance.streaming_buffer_size}")
    print(f"✓ Parallel workers: {config.performance.parallel_workers}")
    print(f"✓ Cache enabled: {config.performance.use_cache}")
    print(f"✓ Checkpoint enabled: {config.performance.checkpoint_enabled}")
    
    assert hasattr(config, 'performance'), "Config missing performance attribute"
    assert config.performance.use_streaming == True, "Streaming should be enabled by default"
    
    print("✅ PASSED: Configuration works correctly\n")
    return True


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("STREAMING IMPLEMENTATION VALIDATION")
    print("=" * 60 + "\n")
    
    tests = [
        ("Basic Streaming", test_streaming_basic),
        ("Deterministic Shuffling", test_streaming_shuffled),
        ("Progress Tracking", test_streaming_progress),
        ("BatchBuffer", test_batch_buffer),
        ("Configuration", test_config),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, "PASSED" if passed else "FAILED"))
        except Exception as e:
            print(f"❌ FAILED: {name}")
            print(f"   Error: {e}\n")
            results.append((name, "FAILED"))
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results:
        symbol = "✅" if status == "PASSED" else "❌"
        print(f"{symbol} {name}: {status}")
    
    passed_count = sum(1 for _, status in results if status == "PASSED")
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Streaming implementation is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
