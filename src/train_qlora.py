"""
QLoRA Training with Empirical Distrust Loss

This script implements QLoRA fine-tuning with Brian Roemmele's Empirical Distrust algorithm.
Source: https://x.com/BrianRoemmele/status/1993393673451847773

Default base model: perplexity-ai/r1-1776 (DeepSeek-R1 with censorship removed)
"""

import json
import sys
import time
import os
import random
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Optional
from tqdm import tqdm
import psutil

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner import linear_to_lora_layers

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from distrust_loss import batch_empirical_distrust_loss
from config import Config
from data.streaming_dataset import StreamingDataset
from checkpoints.checkpoint_manager import CheckpointManager
from checkpoints.checkpoint_state import Checkpoint


class DistrustTrainer:
    """Trainer with Empirical Distrust Loss."""

    def __init__(self, config: Config):
        self.config = config
        self.setup_model()
        self.setup_optimizer()
        self.global_step = 0
        self.loss_history = []

        # Setup checkpoint manager
        if self.config.performance.checkpoint_enabled:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=self.config.performance.checkpoint_dir,
                keep_last_n=self.config.performance.checkpoint_keep_last_n,
                save_interval=self.config.performance.checkpoint_interval,
                async_save=self.config.performance.checkpoint_async,
            )
        else:
            self.checkpoint_manager = None

    def setup_model(self):
        """Load model and tokenizer, apply LoRA."""
        print(f"Loading model: {self.config.paths.model_path}")

        # Load base model
        self.model, self.tokenizer = load(
            self.config.paths.model_path, tokenizer_config={"trust_remote_code": True}
        )

        # Convert to LoRA
        print("Applying LoRA...")
        linear_to_lora_layers(
            self.model,
            lora_layers=self.config.model.lora_rank,
            lora_rank=self.config.model.lora_rank,
            lora_scale=self.config.model.lora_alpha / self.config.model.lora_rank,
        )

        print("Model ready for training")

    def setup_optimizer(self):
        """Setup optimizer with cosine learning rate scheduler."""
        # Cosine decay from initial LR to ~0 over max_steps
        self.lr_schedule = optim.cosine_decay(
            init=self.config.training.learning_rate,
            decay_steps=self.config.training.max_steps,
        )
        self.optimizer = optim.AdamW(
            learning_rate=self.lr_schedule,
            betas=[self.config.training.adam_beta1, self.config.training.adam_beta2],
            eps=self.config.training.adam_epsilon,
            weight_decay=self.config.training.weight_decay,
        )

    def resume_from_checkpoint(self, step: Optional[int] = None) -> bool:
        """Resume training from checkpoint.

        Args:
            step: Specific step to resume from, or None for latest

        Returns:
            True if resumed successfully, False if no checkpoint found
        """
        if not self.checkpoint_manager:
            print("Checkpoint manager not initialized")
            return False

        try:
            if step is not None:
                checkpoint = self.checkpoint_manager.load(step)
                print(f"Resuming from checkpoint at step {step}")
            else:
                checkpoint = self.checkpoint_manager.load_latest()
                if checkpoint is None:
                    print("No checkpoint found to resume from")
                    return False
                print(f"Resuming from latest checkpoint at step {checkpoint.step}")

            # Restore model state
            self.model.update(checkpoint.model_state)

            # Restore optimizer state
            if "step" in checkpoint.optimizer_state:
                self.global_step = checkpoint.optimizer_state["step"]
            else:
                self.global_step = checkpoint.step

            # Restore optimizer step counter and learning rate to sync scheduler
            # MLX stores step and cached learning_rate in optimizer.state
            self.optimizer.state["step"] = mx.array(self.global_step, dtype=mx.uint64)
            self.optimizer.state["learning_rate"] = self.lr_schedule(self.global_step)

            # Restore loss history
            self.loss_history = checkpoint.loss_history.copy()

            # Get restored LR for verification
            restored_lr = float(self.optimizer.learning_rate)

            print(f"✓ Resumed from step {self.global_step}")
            print(f"✓ Learning rate restored to {restored_lr:.6f}")
            print(f"✓ Loss history: {len(self.loss_history)} entries")

            return True

        except Exception as e:
            print(f"Failed to resume from checkpoint: {e}")
            return False

    def load_data(self, file_path: str):
        """Load JSONL data with optional streaming.

        Returns:
            StreamingDataset if streaming enabled, else List[Dict]
        """
        if self.config.performance.use_streaming:
            print(
                f"Using streaming mode (buffer_size={self.config.performance.streaming_buffer_size})"
            )
            return StreamingDataset(
                file_paths=[file_path],
                batch_size=self.config.training.batch_size
                * self.config.training.gradient_accumulation_steps,
                buffer_size=self.config.performance.streaming_buffer_size,
                shuffle=True,
                seed=self.config.seed,
                cycle=True,  # Loop for multiple epochs
            )
        else:
            # Original behavior: load entire dataset
            data = []
            with open(file_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))
            return data

    def prepare_batch(self, examples: List[Dict]) -> Dict[str, mx.array]:
        """Prepare batch for training."""
        texts = [ex["text"] for ex in examples]

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.training.max_seq_length,
            return_tensors="np",
        )

        # Convert to MLX arrays
        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])

        # Extract distrust metrics
        auth_weights = mx.array([ex["auth_weight"] for ex in examples])
        prov_entropies = mx.array([ex["prov_entropy"] for ex in examples])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "auth_weights": auth_weights,
            "prov_entropies": prov_entropies,
        }

    def compute_loss(self, batch: Dict[str, mx.array]) -> tuple:
        """Compute combined loss: CE + Empirical Distrust."""
        input_ids = batch["input_ids"]

        # Forward pass
        logits = self.model(input_ids)

        # Prepare labels (shifted for next-token prediction)
        labels = input_ids[:, 1:]
        logits = logits[:, :-1, :]

        # Cross-entropy loss
        ce_loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="mean"
        )

        # Empirical distrust loss
        distrust_loss = batch_empirical_distrust_loss(
            batch["auth_weights"],
            batch["prov_entropies"],
            alpha=self.config.distrust.alpha,
            reduction="mean",
        )

        # Combined loss
        total_loss = ce_loss + self.config.distrust.lambda_weight * distrust_loss

        return total_loss, ce_loss, distrust_loss

    def train_step(self, batch: Dict[str, mx.array]) -> Dict[str, float]:
        """Single training step with gradient clipping."""

        # Compute loss and gradients
        def loss_fn(model):
            total_loss, ce_loss, distrust_loss = self.compute_loss(batch)
            return total_loss, (ce_loss, distrust_loss)

        # Get gradients
        (total_loss, (ce_loss, distrust_loss)), grads = mx.value_and_grad(loss_fn, argnums=0)(
            self.model
        )

        # Clip gradients to prevent exploding gradients (if max_grad_norm > 0)
        if self.config.training.max_grad_norm and self.config.training.max_grad_norm > 0:
            grads, grad_norm = optim.clip_grad_norm(
                grads, max_norm=self.config.training.max_grad_norm
            )
        else:
            grad_norm = mx.array(0.0)  # Placeholder when clipping disabled

        # Update parameters
        self.optimizer.update(self.model, grads)

        # Evaluate model parameters and optimizer state together
        mx.eval(self.model.parameters(), self.optimizer.state)

        # Get current learning rate from optimizer (auto-computed from scheduler)
        current_lr = self.optimizer.learning_rate

        return {
            "total_loss": float(total_loss),
            "ce_loss": float(ce_loss),
            "distrust_loss": float(distrust_loss),
            "grad_norm": float(grad_norm),
            "lr": float(current_lr),
        }

    def train(self):
        """Main training loop."""
        print("Starting training...")
        train_data = self.load_data(self.config.paths.train_file)

        is_streaming = isinstance(train_data, StreamingDataset)

        if is_streaming:
            print("Using streaming mode - dataset size estimated dynamically")
            total_estimate = train_data.estimate_total_samples()
            if total_estimate:
                print(f"Estimated {total_estimate} total samples")
        else:
            print(f"Loaded {len(train_data)} training examples")

        # Training loop
        batch_size = self.config.training.batch_size

        # Adjust progress bar to start from current step if resuming
        pbar = tqdm(initial=self.global_step, total=self.config.training.max_steps, desc="Training")

        # Memory tracking
        process = psutil.Process(os.getpid())
        baseline_memory_mb = process.memory_info().rss / 1024 / 1024

        if is_streaming:
            # Streaming mode: iterate over dataset
            batch_iter = iter(train_data)

            # Skip already-trained batches when resuming
            if self.global_step > 0:
                print(
                    f"Resuming from step {self.global_step}, skipping {self.global_step} batches..."
                )
                for _ in range(self.global_step):
                    try:
                        next(batch_iter)
                    except StopIteration:
                        # If we run out, restart iterator
                        batch_iter = iter(train_data)
                        break

            for step in range(self.global_step, self.config.training.max_steps):
                try:
                    batch_examples = next(batch_iter)
                except StopIteration:
                    # Should not happen with cycle=True, but handle gracefully
                    batch_iter = iter(train_data)
                    batch_examples = next(batch_iter)

                # Prepare batch
                batch = self.prepare_batch(batch_examples)

                # Train step
                metrics = self.train_step(batch)
                self.loss_history.append(metrics["total_loss"])

                # Logging with streaming progress
                if step % self.config.training.logging_steps == 0:
                    progress_info = train_data.get_progress()
                    current_memory_mb = process.memory_info().rss / 1024 / 1024
                    memory_delta_mb = current_memory_mb - baseline_memory_mb

                    metrics["memory_mb"] = f"{current_memory_mb:.1f}"
                    metrics["mem_delta"] = f"+{memory_delta_mb:.1f}"
                    if progress_info.get("progress_percent") is not None:
                        metrics["data_%"] = f"{progress_info['progress_percent']:.1f}"

                    pbar.set_postfix(metrics)

                # Save checkpoint
                if (
                    self.checkpoint_manager
                    and step > 0
                    and step % self.config.performance.checkpoint_interval == 0
                ):
                    self.save_checkpoint(step)

                self.global_step += 1
                pbar.update(1)

            # Cleanup streaming
            train_data.close()
        else:
            # Original mode: sample from loaded data
            for step in range(self.global_step, self.config.training.max_steps):
                # Sample batch
                idx = (step * batch_size) % len(train_data)
                batch_examples = train_data[idx : idx + batch_size]
                if len(batch_examples) < batch_size:
                    batch_examples = train_data[:batch_size]

                # Prepare batch
                batch = self.prepare_batch(batch_examples)

                # Train step
                metrics = self.train_step(batch)
                self.loss_history.append(metrics["total_loss"])

                # Logging
                if step % self.config.training.logging_steps == 0:
                    pbar.set_postfix(metrics)

                # Save checkpoint
                if (
                    self.checkpoint_manager
                    and step > 0
                    and step % self.config.performance.checkpoint_interval == 0
                ):
                    self.save_checkpoint(step)

                self.global_step += 1
                pbar.update(1)

        pbar.close()
        print("Training complete!")

        # Final save
        self.save_checkpoint(self.global_step, is_final=True)

    def save_checkpoint(self, step: int, is_final: bool = False):
        """Save model checkpoint using CheckpointManager."""
        if not self.checkpoint_manager:
            # Fallback to legacy checkpoint format if no checkpoint manager
            output_path = Path(self.config.paths.output_dir) / f"checkpoint-{step}"
            output_path.mkdir(parents=True, exist_ok=True)

            print(f"Saving checkpoint to {output_path}")

            # Save model weights
            weights_path = output_path / "weights.npz"
            mx.savez(str(weights_path), **dict(self.model.parameters()))

            # Save config
            with open(output_path / "config.json", "w") as f:
                json.dump(
                    {
                        "step": step,
                        "lora_rank": self.config.model.lora_rank,
                        "lora_alpha": self.config.model.lora_alpha,
                        "distrust_alpha": self.config.distrust.alpha,
                    },
                    f,
                    indent=2,
                )

            print("Checkpoint saved")
            return

        # Create checkpoint state
        # Get model and optimizer state
        model_state = dict(self.model.parameters())
        optimizer_state = {}  # MLX optimizers don't expose state dict yet

        # Get random state for reproducibility
        random_state = {"python": random.getstate(), "numpy": np.random.get_state()}

        # Create checkpoint
        checkpoint = Checkpoint(
            step=step,
            model_state=model_state,
            optimizer_state=optimizer_state,
            loss_history=self.loss_history.copy(),
            config=self.config,
            random_state=random_state,
            timestamp=time.time(),
            metadata={
                "lora_rank": self.config.model.lora_rank,
                "lora_alpha": self.config.model.lora_alpha,
                "distrust_alpha": self.config.distrust.alpha,
            },
        )

        # Save using checkpoint manager
        self.checkpoint_manager.save(checkpoint, is_final=is_final)


def main():
    parser = argparse.ArgumentParser(description="Train DeepSeek-V3 with Empirical Distrust Loss")
    parser.add_argument("--model", default="perplexity-ai/r1-1776", help="Model name or path")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-dir", default="models/distrust-r1-1776", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max training steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=2.7, help="Distrust alpha (2.3-3.0)")

    # Streaming options
    parser.add_argument(
        "--no-streaming", action="store_true", help="Disable streaming mode (load entire dataset)"
    )
    parser.add_argument(
        "--streaming-buffer-size", type=int, default=1000, help="Streaming buffer size"
    )

    # Checkpoint options
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume-from-step", type=int, help="Resume from specific checkpoint step")

    args = parser.parse_args()

    # Create config
    config = Config()
    config.paths.model_path = args.model
    config.paths.data_dir = args.data_dir
    config.paths.output_dir = args.output_dir
    config.training.batch_size = args.batch_size
    config.training.max_steps = args.max_steps
    config.training.learning_rate = args.learning_rate
    config.model.lora_rank = args.lora_rank
    config.distrust.alpha = args.alpha

    # Performance config
    config.performance.use_streaming = not args.no_streaming
    config.performance.streaming_buffer_size = args.streaming_buffer_size

    # Train
    trainer = DistrustTrainer(config)

    # Resume from checkpoint if requested
    if args.resume or args.resume_from_step:
        if args.resume_from_step:
            print(f"Resuming from checkpoint step {args.resume_from_step}")
            trainer.resume_from_checkpoint(step=args.resume_from_step)
        else:
            print("Resuming from latest checkpoint")
            trainer.resume_from_checkpoint()

    trainer.train()


if __name__ == "__main__":
    main()
