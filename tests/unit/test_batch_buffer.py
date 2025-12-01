"""Unit tests for BatchBuffer."""

import pytest
import mlx.core as mx
from src.data.batch_buffer import BatchBuffer


@pytest.mark.unit
def test_batch_buffer_allocation():
    """Test BatchBuffer allocates tensors correctly."""
    buffer = BatchBuffer(batch_size=4, max_seq_length=128, dtype=mx.int32)

    # Should have pre-allocated buffer
    assert buffer.input_ids is not None
    assert buffer.input_ids.shape == (4, 128)
    assert buffer.input_ids.dtype == mx.int32


@pytest.mark.unit
def test_batch_buffer_reuse():
    """Test BatchBuffer can be reused across batches."""
    buffer = BatchBuffer(batch_size=4, max_seq_length=128)

    # First use
    buffer.fill([1, 2, 3, 4])
    first_ptr = id(buffer.input_ids)

    # Second use
    buffer.fill([5, 6, 7, 8])
    second_ptr = id(buffer.input_ids)

    # Should reuse same buffer (same memory location)
    assert first_ptr == second_ptr


@pytest.mark.unit
def test_batch_buffer_different_sizes():
    """Test BatchBuffer handles different sequence lengths."""
    buffer = BatchBuffer(batch_size=2, max_seq_length=256)

    # Fill with shorter sequences
    buffer.fill([10, 20], actual_length=64)

    # Should work without error
    assert buffer.input_ids.shape[1] == 256  # Still allocated at max


@pytest.mark.unit
def test_batch_buffer_clear():
    """Test BatchBuffer can be cleared."""
    buffer = BatchBuffer(batch_size=4, max_seq_length=128)

    buffer.fill([1, 2, 3, 4])
    buffer.clear()

    # Should be all zeros after clear
    assert mx.all(buffer.input_ids == 0)


@pytest.mark.unit
def test_batch_buffer_invalid_size():
    """Test validation of buffer parameters."""
    with pytest.raises(ValueError):
        BatchBuffer(batch_size=0, max_seq_length=128)

    with pytest.raises(ValueError):
        BatchBuffer(batch_size=4, max_seq_length=0)
