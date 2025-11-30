"""Integration test for checkpoint recovery."""
import pytest
import tempfile
import shutil
import signal
import os
import subprocess
import time
from pathlib import Path

from src.checkpoints.checkpoint_manager import CheckpointManager
from src.config import Config


@pytest.fixture
def temp_training_dir():
    """Create temporary directory for training."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.slow
def test_checkpoint_recovery_after_interrupt(temp_training_dir):
    """Test that training can resume from checkpoint after interruption."""
    checkpoint_dir = Path(temp_training_dir) / "checkpoints"
    checkpoint_dir.mkdir()
    
    manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        keep_last_n=3,
        save_interval=10,
        async_save=False
    )
    
    config = Config()
    import mlx.core as mx
    from src.checkpoints.checkpoint_state import Checkpoint
    
    # Simulate training: save checkpoint at step 500
    checkpoint = Checkpoint(
        step=500,
        model_state={"layer1": mx.ones((10, 10))},
        optimizer_state={"step": 500, "lr": 0.0002},
        loss_history=[3.0, 2.8, 2.6, 2.4, 2.2],
        config=config,
        random_state={"seed": 42},
        timestamp=time.time(),
        metadata={"interrupted": True}
    )
    
    manager.save(checkpoint)
    
    # Simulate interruption (checkpoint exists on disk)
    # Now resume training
    
    # Load latest checkpoint
    loaded = manager.load_latest()
    
    assert loaded is not None
    assert loaded.step == 500
    assert loaded.optimizer_state["step"] == 500
    assert len(loaded.loss_history) == 5
    
    # Verify can continue training from this state
    assert loaded.model_state["layer1"].shape == (10, 10)
    
    # Simulate continuing training
    checkpoint_continued = Checkpoint(
        step=1000,
        model_state=loaded.model_state,
        optimizer_state={"step": 1000, "lr": 0.0001},
        loss_history=loaded.loss_history + [2.0, 1.8, 1.6],
        config=loaded.config,
        random_state=loaded.random_state,
        timestamp=time.time(),
        metadata={"resumed_from": 500}
    )
    
    manager.save(checkpoint_continued)
    
    # Verify both checkpoints exist
    checkpoints = manager.list_checkpoints()
    assert 500 in checkpoints
    assert 1000 in checkpoints


@pytest.mark.integration
def test_checkpoint_recovery_with_validation(temp_training_dir):
    """Test that corrupted checkpoints are skipped during recovery."""
    checkpoint_dir = Path(temp_training_dir) / "checkpoints"
    checkpoint_dir.mkdir()
    
    manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        async_save=False
    )
    
    config = Config()
    import mlx.core as mx
    from src.checkpoints.checkpoint_state import Checkpoint
    
    # Save checkpoints at steps 100, 200, 300
    for step in [100, 200, 300]:
        checkpoint = Checkpoint(
            step=step,
            model_state={"w": mx.ones((5, 5))},
            optimizer_state={"step": step},
            loss_history=[2.0],
            config=config,
            random_state={},
            timestamp=time.time(),
            metadata={}
        )
        manager.save(checkpoint)
    
    # Corrupt checkpoint 300
    checkpoint_300_path = checkpoint_dir / "checkpoint-300"
    (checkpoint_300_path / "checksum.txt").write_text("invalid_checksum")
    
    # Load latest should skip corrupted 300 and return 200
    loaded = manager.load_latest()
    
    assert loaded is not None
    assert loaded.step == 200


@pytest.mark.integration
def test_checkpoint_save_time_under_threshold(temp_training_dir):
    """Test that checkpoint save completes within time threshold."""
    import time
    checkpoint_dir = Path(temp_training_dir) / "checkpoints"
    checkpoint_dir.mkdir()
    
    manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        async_save=False
    )
    
    config = Config()
    import mlx.core as mx
    from src.checkpoints.checkpoint_state import Checkpoint
    
    # Create checkpoint with realistic data size
    checkpoint = Checkpoint(
        step=1000,
        model_state={
            f"layer_{i}": mx.ones((1000, 1000)) for i in range(5)  # ~40MB
        },
        optimizer_state={"step": 1000, "momentum": {f"layer_{i}": mx.zeros((1000, 1000)) for i in range(5)}},
        loss_history=list(range(1000)),
        config=config,
        random_state={"seed": 42},
        timestamp=time.time(),
        metadata={}
    )
    
    # Measure save time
    start_time = time.time()
    manager.save(checkpoint)
    save_duration = time.time() - start_time
    
    # Should complete in under 10 seconds (requirement from spec)
    assert save_duration < 10.0, f"Save took {save_duration:.2f}s, expected <10s"
    
    print(f"✓ Checkpoint save completed in {save_duration:.2f}s")


@pytest.mark.integration
def test_checkpoint_cleanup_preserves_recent(temp_training_dir):
    """Test that cleanup keeps recent checkpoints even with many saves."""
    checkpoint_dir = Path(temp_training_dir) / "checkpoints"
    checkpoint_dir.mkdir()
    
    manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        keep_last_n=3,
        async_save=False
    )
    
    config = Config()
    import mlx.core as mx
    from src.checkpoints.checkpoint_state import Checkpoint
    
    # Save 10 checkpoints
    for step in range(100, 1100, 100):
        checkpoint = Checkpoint(
            step=step,
            model_state={"w": mx.ones((10, 10))},
            optimizer_state={"step": step},
            loss_history=[2.0],
            config=config,
            random_state={},
            timestamp=time.time(),
            metadata={}
        )
        manager.save(checkpoint)
    
    # Cleanup after each save
    manager.cleanup()
    
    # Should only have last 3
    remaining = manager.list_checkpoints()
    assert len(remaining) == 3
    assert remaining == [800, 900, 1000]
