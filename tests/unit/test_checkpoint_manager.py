"""Unit tests for Checkpoint and CheckpointManager."""

import pytest
import tempfile
import shutil
from pathlib import Path
import mlx.core as mx

from src.checkpoints.checkpoint_state import Checkpoint
from src.checkpoints.checkpoint_manager import CheckpointManager
from src.config import Config


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_checkpoint():
    """Create a sample checkpoint for testing."""
    config = Config()
    return Checkpoint(
        step=1000,
        model_state={"layer1.weight": mx.ones((10, 10)), "layer1.bias": mx.zeros(10)},
        optimizer_state={"step": 1000, "learning_rate": 0.0002},
        loss_history=[2.5, 2.3, 2.1, 1.9],
        config=config,
        random_state={"seed": 42, "state": "dummy"},
        timestamp=1234567890.0,
        metadata={"version": "1.0", "note": "test checkpoint"},
    )


@pytest.mark.unit
def test_checkpoint_save(temp_checkpoint_dir, sample_checkpoint):
    """Test saving checkpoint to disk."""
    manager = CheckpointManager(
        checkpoint_dir=temp_checkpoint_dir,
        keep_last_n=3,
        async_save=False,  # Synchronous for testing
    )

    checkpoint_path = manager.save(sample_checkpoint)

    # Verify directory created
    assert Path(checkpoint_path).exists()
    assert Path(checkpoint_path).is_dir()

    # Verify files exist
    assert (Path(checkpoint_path) / "model.npz").exists()
    assert (Path(checkpoint_path) / "optimizer.npz").exists()
    assert (Path(checkpoint_path) / "metadata.json").exists()
    assert (Path(checkpoint_path) / "checksum.txt").exists()


@pytest.mark.unit
def test_checkpoint_load(temp_checkpoint_dir, sample_checkpoint):
    """Test loading checkpoint from disk."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir, async_save=False)

    # Save first
    manager.save(sample_checkpoint)

    # Load
    loaded = manager.load(step=1000)

    # Verify data
    assert loaded.step == sample_checkpoint.step
    assert loaded.optimizer_state == sample_checkpoint.optimizer_state
    assert loaded.loss_history == sample_checkpoint.loss_history
    assert len(loaded.model_state) == len(sample_checkpoint.model_state)


@pytest.mark.unit
def test_checkpoint_validation_valid(temp_checkpoint_dir, sample_checkpoint):
    """Test validation of valid checkpoint."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir, async_save=False)

    checkpoint_path = manager.save(sample_checkpoint)

    # Should validate successfully
    assert manager.validate(checkpoint_path)


@pytest.mark.unit
def test_checkpoint_validation_corrupted(temp_checkpoint_dir, sample_checkpoint):
    """Test validation detects corrupted checkpoint."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir, async_save=False)

    checkpoint_path = manager.save(sample_checkpoint)

    # Corrupt the model file
    model_path = Path(checkpoint_path) / "model.npz"
    with open(model_path, "wb") as f:
        f.write(b"corrupted data")

    # Should fail validation
    assert not manager.validate(checkpoint_path)


@pytest.mark.unit
def test_checkpoint_validation_missing_file(temp_checkpoint_dir, sample_checkpoint):
    """Test validation detects missing files."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir, async_save=False)

    checkpoint_path = manager.save(sample_checkpoint)

    # Remove checksum file
    (Path(checkpoint_path) / "checksum.txt").unlink()

    # Should fail validation
    assert not manager.validate(checkpoint_path)


@pytest.mark.unit
def test_checkpoint_cleanup(temp_checkpoint_dir):
    """Test cleanup removes old checkpoints."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir, keep_last_n=2, async_save=False)

    config = Config()

    # Save 5 checkpoints
    for step in [100, 200, 300, 400, 500]:
        checkpoint = Checkpoint(
            step=step,
            model_state={"w": mx.ones((2, 2))},
            optimizer_state={"step": step},
            loss_history=[1.0],
            config=config,
            random_state={},
            timestamp=float(step),
            metadata={},
        )
        manager.save(checkpoint)

    # Automatic cleanup should have happened during saves
    # Manual cleanup should return empty list since already cleaned
    deleted = manager.cleanup()
    assert len(deleted) == 0  # Nothing left to delete

    # Should keep only last 2 (400, 500)
    remaining = manager.list_checkpoints()
    assert remaining == [400, 500]


@pytest.mark.unit
def test_checkpoint_load_latest(temp_checkpoint_dir):
    """Test loading most recent checkpoint."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir, async_save=False)

    config = Config()

    # Save multiple checkpoints
    for step in [100, 200, 300]:
        checkpoint = Checkpoint(
            step=step,
            model_state={"w": mx.ones((2, 2))},
            optimizer_state={"step": step},
            loss_history=[1.0],
            config=config,
            random_state={},
            timestamp=float(step),
            metadata={},
        )
        manager.save(checkpoint)

    # Load latest
    latest = manager.load_latest()

    assert latest is not None
    assert latest.step == 300


@pytest.mark.unit
def test_checkpoint_load_latest_skips_corrupted(temp_checkpoint_dir):
    """Test that load_latest skips corrupted checkpoints."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir, async_save=False)

    config = Config()

    # Save checkpoints at step 100 and 200
    for step in [100, 200]:
        checkpoint = Checkpoint(
            step=step,
            model_state={"w": mx.ones((2, 2))},
            optimizer_state={"step": step},
            loss_history=[1.0],
            config=config,
            random_state={},
            timestamp=float(step),
            metadata={},
        )
        manager.save(checkpoint)

    # Corrupt checkpoint 200
    checkpoint_200_path = Path(temp_checkpoint_dir) / "checkpoint-200"
    (checkpoint_200_path / "model.npz").unlink()

    # Load latest should return 100 (skip corrupted 200)
    latest = manager.load_latest()

    assert latest is not None
    assert latest.step == 100


@pytest.mark.unit
def test_checkpoint_list(temp_checkpoint_dir):
    """Test listing available checkpoints."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir, async_save=False)

    config = Config()

    # Save checkpoints
    for step in [100, 300, 200]:  # Out of order
        checkpoint = Checkpoint(
            step=step,
            model_state={"w": mx.ones((2, 2))},
            optimizer_state={"step": step},
            loss_history=[1.0],
            config=config,
            random_state={},
            timestamp=float(step),
            metadata={},
        )
        manager.save(checkpoint)

    # List should be sorted
    checkpoints = manager.list_checkpoints()
    assert checkpoints == [100, 200, 300]


@pytest.mark.unit
def test_checkpoint_final_not_deleted(temp_checkpoint_dir):
    """Test that final checkpoint is never deleted."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir, keep_last_n=1, async_save=False)

    config = Config()

    # Save regular checkpoints
    for step in [100, 200, 300]:
        checkpoint = Checkpoint(
            step=step,
            model_state={"w": mx.ones((2, 2))},
            optimizer_state={"step": step},
            loss_history=[1.0],
            config=config,
            random_state={},
            timestamp=float(step),
            metadata={},
        )
        manager.save(checkpoint, is_final=(step == 100))

    # Cleanup
    manager.cleanup()

    # Final checkpoint (100) should still exist
    remaining = manager.list_checkpoints()
    assert 100 in remaining  # Final checkpoint preserved
    assert 300 in remaining  # Most recent preserved


@pytest.mark.unit
def test_checkpoint_async_save(temp_checkpoint_dir, sample_checkpoint):
    """Test asynchronous checkpoint saving."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir, async_save=True)

    # Save should return immediately
    checkpoint_path = manager.save(sample_checkpoint)

    # Wait for completion
    manager.wait_for_save()

    # Verify saved
    assert Path(checkpoint_path).exists()
    assert manager.validate(checkpoint_path)


@pytest.mark.unit
def test_checkpoint_manager_invalid_params(temp_checkpoint_dir):
    """Test validation of manager parameters."""
    with pytest.raises(ValueError):
        CheckpointManager(checkpoint_dir=temp_checkpoint_dir, keep_last_n=0)

    with pytest.raises(ValueError):
        CheckpointManager(checkpoint_dir=temp_checkpoint_dir, save_interval=0)


@pytest.mark.unit
def test_checkpoint_load_nonexistent(temp_checkpoint_dir):
    """Test loading nonexistent checkpoint raises error."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir, async_save=False)

    with pytest.raises(FileNotFoundError):
        manager.load(step=999)
