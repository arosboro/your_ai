"""StreamingDataset for lazy-loading JSONL files."""
import json
import logging
import random
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Optional


logger = logging.getLogger(__name__)


class StreamingDataset:
    """
    Lazy-loading iterator that streams JSONL data without loading entire file into memory.

    Supports multiple files, shuffling, and progress tracking.
    """
    
    def __init__(
        self,
        file_paths: List[str],
        batch_size: int,
        buffer_size: int = 1000,
        shuffle: bool = False,
        seed: Optional[int] = None,
        cycle: bool = False
    ):
        """
        Initialize streaming dataset.

        Args:
            file_paths: List of JSONL file paths to stream
            batch_size: Number of samples per batch
            buffer_size: Internal prefetch buffer size (must be >= batch_size)
            shuffle: Whether to shuffle samples (uses buffered shuffle)
            seed: Random seed for deterministic shuffling
            cycle: Whether to loop infinitely (for training epochs)

        Raises:
            ValueError: If batch_size <= 0 or buffer_size < batch_size
            FileNotFoundError: If any file_path doesn't exist
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if buffer_size < batch_size:
            raise ValueError(f"buffer_size ({buffer_size}) must be >= batch_size ({batch_size})")
        
        # Validate files exist
        for fp in file_paths:
            if not Path(fp).exists():
                raise FileNotFoundError(f"File not found: {fp}")
        
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.seed = seed
        self.cycle = cycle
        
        # State
        self.current_position = 0
        self.current_file_idx = 0
        self.current_file = None
        self.current_file_handle = None
        self._buffer = deque()
        self._total_samples_estimate = None
        self._rng = random.Random(seed) if shuffle else None
        self._iteration_started = False
    
    def __iter__(self):
        """Return iterator (self)."""
        self._iteration_started = True
        return self
    
    def __next__(self) -> List[Dict[str, Any]]:
        """
        Get next batch of samples.

        Returns:
            List of dicts, each containing sample data (length = batch_size,
            except possibly last batch)

        Raises:
            StopIteration: When all files exhausted (if not cycling)
        """
        batch = []
        
        while len(batch) < self.batch_size:
            # Refill buffer if needed
            if not self._buffer:
                self._fill_buffer()
                
                # If buffer still empty, we're done
                if not self._buffer:
                    if self.cycle and self.current_file_idx == 0:
                        # Check if dataset is truly empty (prevent infinite loop)
                        if self.current_position == 0:
                            raise StopIteration
                        # Restart from beginning
                        self.current_file_idx = 0
                        self.current_position = 0
                        self._close_current_file()
                        continue
                    elif len(batch) > 0:
                        # Return partial batch
                        return batch
                    else:
                        raise StopIteration
            
            # Take from buffer
            sample = self._buffer.popleft()
            batch.append(sample)
            self.current_position += 1
        
        return batch
    
    def _fill_buffer(self):
        """Fill internal buffer from current file."""
        # Open next file if needed
        if self.current_file_handle is None:
            if self.current_file_idx >= len(self.file_paths):
                return  # No more files
            
            self.current_file = self.file_paths[self.current_file_idx]
            try:
                self.current_file_handle = open(self.current_file, 'r')
            except Exception as e:
                logger.error(f"Failed to open {self.current_file}: {e}")
                self.current_file_idx += 1
                return
        
        # Read lines into buffer
        lines_read = 0
        while lines_read < self.buffer_size:
            line = self.current_file_handle.readline()
            
            if not line:
                # End of file
                self._close_current_file()
                self.current_file_idx += 1
                
                # Try next file
                if self.current_file_idx < len(self.file_paths):
                    self.current_file = self.file_paths[self.current_file_idx]
                    try:
                        self.current_file_handle = open(self.current_file, 'r')
                    except Exception as e:
                        logger.error(f"Failed to open {self.current_file}: {e}")
                        self.current_file_idx += 1
                        continue
                    continue
                elif self.cycle:
                    # Check if we've cycled too many times without reading data
                    if lines_read == 0 and hasattr(self, '_cycle_count'):
                        self._cycle_count += 1
                        if self._cycle_count >= len(self.file_paths) * 2:
                            logger.error("Infinite cycle detected: no valid data after multiple passes")
                            break
                    elif lines_read > 0:
                        self._cycle_count = 0
                    elif not hasattr(self, '_cycle_count'):
                        self._cycle_count = 0
                    
                    # Restart from beginning
                    self.current_file_idx = 0
                    self.current_file = self.file_paths[0]
                    try:
                        self.current_file_handle = open(self.current_file, 'r')
                    except Exception as e:
                        logger.error(f"Failed to open {self.current_file}: {e}")
                        break
                    continue
                else:
                    break
            
            # Parse JSON
            try:
                sample = json.loads(line.strip())
                self._buffer.append(sample)
                lines_read += 1
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping corrupted line in {self.current_file}: {e}")
                continue
        
        # Shuffle buffer if requested
        if self.shuffle and self._rng:
            buffer_list = list(self._buffer)
            self._rng.shuffle(buffer_list)
            self._buffer = deque(buffer_list)
    
    def _close_current_file(self):
        """Close current file handle."""
        if self.current_file_handle:
            self.current_file_handle.close()
            self.current_file_handle = None
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress information.

        Returns:
            Dict with keys:
                - current_position: int (current sample index)
                - total_samples: Optional[int] (estimated total if known)
                - current_file: str (current file being read)
                - progress_percent: Optional[float] (percent complete if total known)
        """
        total = self._total_samples_estimate or self.estimate_total_samples()
        progress_percent = None
        if total:
            progress_percent = (self.current_position / total) * 100
        
        return {
            "current_position": self.current_position,
            "total_samples": total,
            "current_file": self.current_file or self.file_paths[0] if self.file_paths else "",
            "progress_percent": progress_percent
        }
    
    def estimate_total_samples(self) -> Optional[int]:
        """
        Estimate total samples across all files by counting lines.
        
        Note: This method is cached - only counts lines on first call.

        Returns:
            Estimated total sample count, or None if estimation fails
        """
        if self._total_samples_estimate is not None:
            return self._total_samples_estimate
        
        try:
            total = 0
            for fp in self.file_paths:
                # Count lines in file
                with open(fp, 'r') as f:
                    count = sum(1 for _ in f)
                total += count
            
            self._total_samples_estimate = total
            return total
        except Exception as e:
            logger.warning(f"Failed to estimate total samples: {e}")
            return None
    
    def reset(self) -> None:
        """
        Reset iterator to beginning.

        Resets all internal state including file position and buffer.
        Safe to call at any time; mid-iteration reset will restart from beginning.
        """
        self._close_current_file()
        self.current_position = 0
        self.current_file_idx = 0
        self.current_file = None
        self._buffer = deque()
        self._iteration_started = False
        if self.shuffle and self.seed is not None:
            self._rng = random.Random(self.seed)
    
    def close(self) -> None:
        """
        Close any open file handles and cleanup resources.

        Should be called when done streaming or use as context manager.
        """
        self._close_current_file()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit, ensures cleanup."""
        self.close()
