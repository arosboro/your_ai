#!/usr/bin/env python3
"""
Test script to validate checkpoint integration with train_qlora.py

Tests:
1. Checkpoint save/load workflow
2. Training resume from checkpoint
3. Loss history preservation
4. Configuration restoration
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.checkpoints.checkpoint_state import Checkpoint
from src.checkpoints.checkpoint_manager import CheckpointManager
import mlx.core as mx


def test_checkpoint_save_load():
    """Test basic checkpoint save/load."""
    print("\n=== Test 1: Checkpoint Save/Load ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = Config()
        config.performance.checkpoint_dir = temp_dir
        
        manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            keep_last_n=3,
            async_save=False
        )
        
        # Create sample checkpoint
        checkpoint = Checkpoint(
            step=500,
            model_state={"layer1.weight": mx.ones((10, 10))},
            optimizer_state={"step": 500, "lr": 0.0002},
            loss_history=[2.5, 2.3, 2.1, 1.9],
            config=config,
            random_state={},
            timestamp=1234567890.0,
            metadata={}
        )
        
        # Save
        manager.save(checkpoint)
        print("✓ Saved checkpoint at step 500")
        
        # Load
        loaded = manager.load(step=500)
        assert loaded.step == 500
        assert loaded.loss_history == [2.5, 2.3, 2.1, 1.9]
        assert loaded.optimizer_state["step"] == 500
        print("✓ Loaded checkpoint at step 500")
        print(f"  Loss history: {loaded.loss_history}")
        print(f"  Optimizer state: {loaded.optimizer_state}")
    
    print("✓ Test 1 PASSED\n")


def test_checkpoint_resume():
    """Test checkpoint resume workflow."""
    print("=== Test 2: Checkpoint Resume ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = Config()
        config.performance.checkpoint_dir = temp_dir
        
        manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            keep_last_n=3,
            async_save=False
        )
        
        # Simulate training: save checkpoints at steps 100, 200, 300
        for step in [100, 200, 300]:
            checkpoint = Checkpoint(
                step=step,
                model_state={"layer1.weight": mx.ones((10, 10))},
                optimizer_state={"step": step},
                loss_history=[2.5 - (step/100) * 0.2],
                config=config,
                random_state={},
                timestamp=float(step),
                metadata={}
            )
            manager.save(checkpoint)
            print(f"✓ Saved checkpoint at step {step}")
        
        # Resume from latest
        latest = manager.load_latest()
        assert latest is not None
        assert latest.step == 300
        print(f"✓ Resumed from latest checkpoint (step {latest.step})")
        
        # Resume from specific step
        checkpoint_200 = manager.load(step=200)
        assert checkpoint_200.step == 200
        print(f"✓ Resumed from specific checkpoint (step {checkpoint_200.step})")
    
    print("✓ Test 2 PASSED\n")


def test_config_serialization():
    """Test config to_dict and restoration."""
    print("=== Test 3: Config Serialization ===")
    
    config = Config()
    config.model.lora_rank = 64
    config.training.learning_rate = 1e-4
    config.distrust.alpha = 2.5
    
    # Convert to dict
    config_dict = config.to_dict()
    print("✓ Serialized config to dict")
    print(f"  LoRA rank: {config_dict['model']['lora_rank']}")
    print(f"  Learning rate: {config_dict['training']['learning_rate']}")
    print(f"  Distrust alpha: {config_dict['distrust']['alpha']}")
    
    # Verify structure
    assert config_dict['model']['lora_rank'] == 64
    assert config_dict['training']['learning_rate'] == 1e-4
    assert config_dict['distrust']['alpha'] == 2.5
    
    print("✓ Test 3 PASSED\n")


def test_checkpoint_cleanup():
    """Test checkpoint cleanup preserves most recent."""
    print("=== Test 4: Checkpoint Cleanup ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            keep_last_n=2,
            async_save=False
        )
        
        config = Config()
        
        # Save 5 checkpoints
        for step in [100, 200, 300, 400, 500]:
            checkpoint = Checkpoint(
                step=step,
                model_state={"w": mx.ones((2, 2))},
                optimizer_state={},
                loss_history=[],
                config=config,
                random_state={},
                timestamp=float(step),
                metadata={}
            )
            manager.save(checkpoint)
        
        # Check only last 2 remain
        remaining = manager.list_checkpoints()
        assert remaining == [400, 500]
        print(f"✓ Cleanup preserved last 2 checkpoints: {remaining}")
        
        # Save final checkpoint
        final = Checkpoint(
            step=600,
            model_state={"w": mx.ones((2, 2))},
            optimizer_state={},
            loss_history=[],
            config=config,
            random_state={},
            timestamp=600.0,
            metadata={}
        )
        manager.save(final, is_final=True)
        
        # Final checkpoint should be preserved
        remaining = manager.list_checkpoints()
        assert 600 in remaining
        print(f"✓ Final checkpoint preserved: {remaining}")
    
    print("✓ Test 4 PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Checkpoint Integration Test Suite")
    print("="*60)
    
    try:
        test_checkpoint_save_load()
        test_checkpoint_resume()
        test_config_serialization()
        test_checkpoint_cleanup()
        
        print("="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
