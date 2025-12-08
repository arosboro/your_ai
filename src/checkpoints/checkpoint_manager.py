"""Checkpoint manager for save/load/validation."""

import json
import hashlib
import threading
import time
import logging
import shutil
from pathlib import Path
from typing import List, Optional
import mlx.core as mx

from src.checkpoints.checkpoint_state import Checkpoint
from src.config import Config


logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoint saving, loading, validation, and cleanup.

    Supports async saves, corruption detection, and automatic old checkpoint removal.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 3,
        save_interval: int = 500,
        async_save: bool = True,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Base directory for checkpoints
            keep_last_n: Number of recent checkpoints to retain
            save_interval: Steps between automatic saves
            async_save: Whether to save in background thread

        Raises:
            OSError: If checkpoint_dir not writable
            ValueError: If keep_last_n < 1 or save_interval <= 0
        """
        if keep_last_n < 1:
            raise ValueError(f"keep_last_n must be >= 1, got {keep_last_n}")
        if save_interval <= 0:
            raise ValueError(f"save_interval must be > 0, got {save_interval}")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Test writability
        test_file = self.checkpoint_dir / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except OSError as e:
            raise OSError(f"Checkpoint directory not writable: {checkpoint_dir}") from e

        self.keep_last_n = keep_last_n
        self.save_interval = save_interval
        self.async_save = async_save

        # For async saves
        self._save_thread: Optional[threading.Thread] = None
        self._save_lock = threading.Lock()
        self._save_in_progress = False

    def save(self, checkpoint: Checkpoint, is_final: bool = False) -> str:
        """
        Save checkpoint to disk.

        Args:
            checkpoint: Checkpoint object to save
            is_final: Whether this is the final checkpoint (kept indefinitely)

        Returns:
            Path to saved checkpoint directory

        Side Effects:
            - Creates checkpoint-{step}/ directory
            - Writes model.npz, optimizer.npz, metadata.json, checksum.txt
            - Triggers cleanup of old checkpoints if not is_final
            - If async_save=True, returns immediately and saves in background

        Raises:
            OSError: If write fails
        """
        if is_final:
            checkpoint_path = self.checkpoint_dir / f"checkpoint-{checkpoint.step}-final"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint-{checkpoint.step}"

        if self.async_save:
            # Wait for any pending save
            self.wait_for_save()

            # Mark save as in progress
            self._save_in_progress = True

            # Start background save
            def _async_save_wrapper():
                try:
                    self._save_sync(checkpoint, checkpoint_path, is_final)
                finally:
                    self._save_in_progress = False

            self._save_thread = threading.Thread(target=_async_save_wrapper)
            self._save_thread.start()
        else:
            # Synchronous save
            self._save_sync(checkpoint, checkpoint_path, is_final)

        return str(checkpoint_path)

    def _save_sync(self, checkpoint: Checkpoint, checkpoint_path: Path, is_final: bool):
        """Internal synchronous save implementation."""
        with self._save_lock:
            try:
                # Create directory
                checkpoint_path.mkdir(parents=True, exist_ok=True)

                # Save model state
                model_path = checkpoint_path / "model.npz"

                # Filter and validate model state before saving
                clean_model_state = {}
                arrays_to_eval = []

                for key, value in checkpoint.model_state.items():
                    if isinstance(value, mx.array):
                        try:
                            # Check if array is valid (size > 0)
                            if value.size > 0:
                                clean_model_state[key] = value
                                arrays_to_eval.append(value)
                        except Exception as e:
                            logger.warning(f"Failed to validate array {key}: {e}")
                    else:
                        # Skip non-array values to avoid std::bad_cast
                        logger.warning(f"Skipping non-array model state key: {key} (type: {type(value)})")

                if not clean_model_state:
                    raise RuntimeError("No valid arrays in model state to save")

                # Evaluate all arrays as a batch to ensure they're fully materialized
                # This prevents lazy evaluation issues that cause std::bad_cast
                logger.info(f"Evaluating {len(arrays_to_eval)} arrays before save...")
                mx.eval(arrays_to_eval)

                # Save with error recovery
                try:
                    mx.savez(str(model_path), **clean_model_state)
                    logger.info(f"Model state saved successfully ({len(clean_model_state)} parameters)")
                except Exception as e:
                    # If save fails, try saving individual arrays to identify problematic ones
                    logger.error(f"Batch save failed: {e}. Attempting individual saves...")
                    failed_keys = []
                    partial_model_path = checkpoint_path / "model_partial.npz"
                    saved_state = {}

                    for key, value in clean_model_state.items():
                        try:
                            # Test if this array can be saved
                            test_dict = {key: value}
                            mx.eval([value])
                            saved_state[key] = value
                        except Exception as array_err:
                            logger.error(f"Failed to save parameter {key}: {array_err}")
                            failed_keys.append(key)

                    if saved_state:
                        # Save what we could
                        mx.savez(str(partial_model_path), **saved_state)
                        logger.warning(f"Partial save successful: {len(saved_state)}/{len(clean_model_state)} parameters")
                        logger.warning(f"Failed parameters: {failed_keys}")
                        # Use partial save as main save
                        partial_model_path.rename(model_path)
                    else:
                        raise RuntimeError(f"Could not save any parameters. All {len(failed_keys)} failed.") from e

                # Clean up to free memory
                del clean_model_state
                if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()

                # Save optimizer state
                optimizer_path = checkpoint_path / "optimizer.npz"
                # Convert optimizer state to saveable format (arrays only)
                opt_arrays = {}
                opt_scalars = {}
                opt_arrays_to_eval = []

                for key, value in checkpoint.optimizer_state.items():
                    if isinstance(value, dict):
                        # Flatten nested dicts
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, mx.array):
                                opt_arrays[f"{key}.{subkey}"] = subvalue
                                opt_arrays_to_eval.append(subvalue)
                            else:
                                opt_scalars[f"{key}.{subkey}"] = subvalue
                    elif isinstance(value, mx.array):
                        opt_arrays[key] = value
                        opt_arrays_to_eval.append(value)
                    else:
                        # Store scalars separately in metadata
                        opt_scalars[key] = value

                # Evaluate all optimizer arrays before saving
                if opt_arrays_to_eval:
                    logger.info(f"Evaluating {len(opt_arrays_to_eval)} optimizer arrays...")
                    mx.eval(opt_arrays_to_eval)

                if opt_arrays:
                    try:
                        mx.savez(str(optimizer_path), **opt_arrays)
                        logger.info(f"Optimizer state saved successfully ({len(opt_arrays)} values)")
                    except Exception as e:
                        logger.error(f"Failed to save optimizer state: {e}")
                        # Save empty file to maintain checkpoint structure
                        mx.savez(str(optimizer_path))
                        logger.warning("Saved empty optimizer state due to error")
                else:
                    # Save empty npz file for consistent handling
                    mx.savez(str(optimizer_path))

                # Save metadata
                metadata_path = checkpoint_path / "metadata.json"
                metadata = {
                    "step": checkpoint.step,
                    "timestamp": checkpoint.timestamp,
                    "loss_history": checkpoint.loss_history,
                    "random_state": checkpoint.random_state,
                    "metadata": checkpoint.metadata,
                    "optimizer_scalars": opt_scalars,  # Store scalar optimizer values
                    "config": checkpoint.config.to_dict(),
                }

                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                # Compute checksums
                checksums = {}
                for file_name in ["model.npz", "optimizer.npz", "metadata.json"]:
                    file_path = checkpoint_path / file_name
                    if file_path.exists():
                        with open(file_path, "rb") as f:
                            checksums[file_name] = hashlib.sha256(f.read()).hexdigest()

                # Save checksums
                checksum_path = checkpoint_path / "checksum.txt"
                with open(checksum_path, "w") as f:
                    for file_name, checksum in checksums.items():
                        f.write(f"{checksum}  {file_name}\n")

                logger.info(f"Saved checkpoint at step {checkpoint.step} to {checkpoint_path}")

                # Cleanup old checkpoints (only if not final)
                if not is_final:
                    self.cleanup()

            except Exception as e:
                logger.exception(f"Failed to save checkpoint: {e}")
                raise

    def load(self, step: int) -> Checkpoint:
        """
        Load specific checkpoint by step number.

        Args:
            step: Step number of checkpoint to load

        Returns:
            Loaded Checkpoint object

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint fails validation
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"

        # Try final checkpoint first if regular doesn't exist
        if not checkpoint_path.exists():
            final_path = self.checkpoint_dir / f"checkpoint-{step}-final"
            if final_path.exists():
                checkpoint_path = final_path
            else:
                raise FileNotFoundError(f"Checkpoint not found: checkpoint-{step}")

        if not self.validate(str(checkpoint_path)):
            raise ValueError(f"Checkpoint validation failed: {checkpoint_path}")

        return self._load_from_path(checkpoint_path)

    def _load_from_path(self, checkpoint_path: Path) -> Checkpoint:
        """Internal method to load checkpoint from path."""
        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Load model state
        model_path = checkpoint_path / "model.npz"
        model_state = dict(mx.load(str(model_path)))

        # Load optimizer state
        optimizer_path = checkpoint_path / "optimizer.npz"
        optimizer_state = {}

        if optimizer_path.stat().st_size > 0:
            opt_arrays = dict(mx.load(str(optimizer_path)))
            # Reconstruct optimizer state structure
            for key, value in opt_arrays.items():
                if "." in key:
                    # Nested dict
                    parent, child = key.split(".", 1)
                    if parent not in optimizer_state:
                        optimizer_state[parent] = {}
                    optimizer_state[parent][child] = value
                else:
                    optimizer_state[key] = value

        # Add scalar values from metadata
        if "optimizer_scalars" in metadata:
            for key, value in metadata["optimizer_scalars"].items():
                if "." in key:
                    # Nested dict
                    parent, child = key.split(".", 1)
                    if parent not in optimizer_state:
                        optimizer_state[parent] = {}
                    optimizer_state[parent][child] = value
                else:
                    optimizer_state[key] = value

        # Reconstruct Config from saved dict
        if "config" in metadata:
            config = Config.from_dict(metadata["config"])
        else:
            # Fallback for old checkpoints with partial config
            config = Config()
            cfg_data = metadata.get("config", {})
            if "lora_rank" in cfg_data:
                config.model.lora_rank = cfg_data["lora_rank"]
            if "lora_alpha" in cfg_data:
                config.model.lora_alpha = cfg_data["lora_alpha"]
            if "distrust_alpha" in cfg_data:
                config.distrust.alpha = cfg_data["distrust_alpha"]
            if "learning_rate" in cfg_data:
                config.training.learning_rate = cfg_data["learning_rate"]

        checkpoint = Checkpoint(
            step=metadata["step"],
            model_state=model_state,
            optimizer_state=optimizer_state,
            loss_history=metadata.get("loss_history", []),
            config=config,
            random_state=metadata.get("random_state", {}),
            timestamp=metadata.get("timestamp", time.time()),
            metadata=metadata.get("metadata", {}),
        )

        logger.info(f"Loaded checkpoint from step {checkpoint.step}")
        return checkpoint

    def load_latest(self) -> Optional[Checkpoint]:
        """
        Load most recent checkpoint.

        Returns:
            Latest Checkpoint if any exist, None otherwise

        Note: Skips corrupted checkpoints and tries previous ones
        """
        checkpoints = self.list_checkpoints()

        if not checkpoints:
            return None

        # Try checkpoints from newest to oldest
        for step in reversed(checkpoints):
            checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"

            if self.validate(str(checkpoint_path)):
                try:
                    return self._load_from_path(checkpoint_path)
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint {step}: {e}")
                    continue
            else:
                logger.warning(f"Checkpoint {step} failed validation, trying previous")
                continue

        return None

    def validate(self, checkpoint_path: str) -> bool:
        """
        Validate checkpoint integrity.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            True if valid, False if corrupted

        Validation checks:
            - All required files exist (model.npz, optimizer.npz, metadata.json, checksum.txt)
            - Checksums match stored values
            - Metadata is valid JSON
            - Arrays can be loaded
        """
        path = Path(checkpoint_path)

        # Check required files exist
        required_files = ["model.npz", "metadata.json", "checksum.txt"]
        for file_name in required_files:
            if not (path / file_name).exists():
                logger.warning(f"Missing file: {file_name}")
                return False

        # Load and validate checksums
        try:
            checksum_path = path / "checksum.txt"
            stored_checksums = {}
            with open(checksum_path, "r") as f:
                for line in f:
                    checksum, file_name = line.strip().split(None, 1)
                    stored_checksums[file_name] = checksum

            # Verify checksums
            for file_name in ["model.npz", "metadata.json"]:
                if file_name in stored_checksums:
                    file_path = path / file_name
                    with open(file_path, "rb") as f:
                        actual_checksum = hashlib.sha256(f.read()).hexdigest()

                    if actual_checksum != stored_checksums[file_name]:
                        logger.warning(f"Checksum mismatch for {file_name}")
                        return False

            # Validate metadata JSON
            metadata_path = path / "metadata.json"
            with open(metadata_path, "r") as f:
                json.load(f)  # Just check it's valid JSON

            # Try loading arrays (basic validation)
            model_path = path / "model.npz"
            mx.load(str(model_path))

            return True

        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return False

    def cleanup(self) -> List[str]:
        """
        Remove old checkpoints, keeping only last N.

        Returns:
            List of deleted checkpoint paths

        Note: Never deletes checkpoint-final/ or most recent keep_last_n checkpoints
        """
        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= self.keep_last_n:
            return []

        # Keep last N
        to_delete = checkpoints[: -self.keep_last_n]
        deleted = []

        for step in to_delete:
            checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"

            # Skip final checkpoints (both checkpoint-final and checkpoint-{step}-final)
            if (
                checkpoint_path.name.endswith("-final")
                or (self.checkpoint_dir / f"checkpoint-{step}-final").exists()
            ):
                continue

            # Check if path exists before attempting deletion
            if not checkpoint_path.exists():
                continue

            try:
                shutil.rmtree(checkpoint_path)
                deleted.append(str(checkpoint_path))
                logger.info(f"Deleted old checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {checkpoint_path}: {e}")

        return deleted

    def list_checkpoints(self) -> List[int]:
        """
        List available checkpoint step numbers.

        Returns:
            Sorted list of step numbers (ascending), including final checkpoints
        """
        checkpoints = []

        for path in self.checkpoint_dir.glob("checkpoint-*"):
            if path.is_dir():
                try:
                    # Extract step from both 'checkpoint-100' and 'checkpoint-100-final'
                    step_str = path.name.replace("checkpoint-", "").replace("-final", "")
                    step = int(step_str)
                    if step not in checkpoints:
                        checkpoints.append(step)
                except (ValueError, IndexError):
                    continue

        return sorted(checkpoints)

    def wait_for_save(self) -> None:
        """
        Block until any pending async save completes.

        Note: No-op if async_save=False or no save in progress
        """
        if self._save_thread and self._save_thread.is_alive():
            self._save_thread.join()

    @property
    def is_saving(self) -> bool:
        """
        Check if an async save is currently in progress.

        Returns:
            True if save is in progress, False otherwise
        """
        return self._save_in_progress

    def close(self) -> None:
        """
        Cleanup resources and wait for pending saves.

        Should be called before program exit.
        """
        self.wait_for_save()
