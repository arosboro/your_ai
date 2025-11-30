"""BatchBuffer for efficient batch tensor allocation."""
import mlx.core as mx
from typing import Optional, List


class BatchBuffer:
    """
    Pre-allocated tensor buffer for batch processing.
    
    Reduces memory allocation overhead by reusing the same buffer
    across multiple batches.
    """
    
    def __init__(
        self,
        batch_size: int,
        max_seq_length: int,
        dtype: mx.Dtype = mx.int32
    ):
        """
        Initialize batch buffer with pre-allocated tensors.
        
        Args:
            batch_size: Maximum number of samples in a batch
            max_seq_length: Maximum sequence length
            dtype: Data type for tensors
            
        Raises:
            ValueError: If batch_size or max_seq_length <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be > 0, got {max_seq_length}")
        
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        
        # Pre-allocate tensors
        self.input_ids = mx.zeros((batch_size, max_seq_length), dtype=dtype)
        self.attention_mask = mx.zeros((batch_size, max_seq_length), dtype=dtype)
    
    def fill(self, token_ids: List[int], actual_length: Optional[int] = None):
        """
        Fill buffer with token IDs.
        
        Args:
            token_ids: List of token IDs to fill buffer with
            actual_length: Actual length to use (if different from max)
        """
        raise NotImplementedError("fill() is not yet implemented")
    
    def clear(self):
        """Clear buffer by zeroing all values."""
        self.input_ids = mx.zeros((self.batch_size, self.max_seq_length), dtype=self.dtype)
        self.attention_mask = mx.zeros((self.batch_size, self.max_seq_length), dtype=self.dtype)
    
    def get_view(self, actual_batch_size: int):
        """
        Get view of buffer for actual batch size (no copy).
        
        Args:
            actual_batch_size: Actual number of samples in batch
            
        Returns:
            View into buffer for actual batch size
        """
        if actual_batch_size > self.batch_size:
            raise ValueError(f"actual_batch_size {actual_batch_size} > buffer size {self.batch_size}")
        
        return {
            "input_ids": self.input_ids[:actual_batch_size, :],
            "attention_mask": self.attention_mask[:actual_batch_size, :]
        }
