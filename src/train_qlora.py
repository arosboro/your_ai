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
from datetime import datetime
from tensorboardX import SummaryWriter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner import linear_to_lora_layers

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from distrust_loss import batch_empirical_distrust_loss
from config import Config, PathConfig
from data.streaming_dataset import StreamingDataset
from checkpoints.checkpoint_manager import CheckpointManager
from checkpoints.checkpoint_state import Checkpoint
from hardware_profiles import (
    interactive_setup,
    load_hardware_profile,
    profile_exists,
    save_hardware_profile,
    get_optimized_profile,
    display_recommendations,
    recommend_models,
    detect_hardware,
    scale_profile_for_model,
    validate_config_safety,
    detect_model_size,
)


def estimate_optimal_lambda_weight(
    train_file: str, num_samples: int = 100, target_ratio: float = 1.0
) -> float:
    """
    Estimate optimal lambda_weight by analyzing training data statistics.

    The goal is to balance distrust loss and cross-entropy loss so they contribute
    equally to training (or at a specified ratio).

    Parameters:
        train_file: Path to training data JSONL file
        num_samples: Number of samples to analyze (default: 100)
        target_ratio: Desired ratio of distrust_contribution / ce_loss (default: 1.0)

    Returns:
        Recommended lambda_weight value
    """
    import json

    print("\n🔍 Analyzing training data to calibrate lambda_weight...")

    auth_weights = []
    prov_entropies = []

    # Sample training data
    with open(train_file, "r") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            try:
                sample = json.loads(line)
                auth_weights.append(sample.get("auth_weight", 0.5))
                prov_entropies.append(sample.get("prov_entropy", 5.0))
            except json.JSONDecodeError:
                continue

    if not auth_weights:
        print("⚠️  Could not read training data, using default lambda_weight=0.6")
        return 0.6

    # Compute average distrust loss (without lambda scaling)
    avg_auth = sum(auth_weights) / len(auth_weights)
    avg_prov = sum(prov_entropies) / len(prov_entropies)

    # Using alpha=2.7 (default)
    alpha = 2.7
    epsilon = 1e-8
    distrust_component = (-1 * avg_auth) + avg_prov + epsilon
    avg_distrust_loss = alpha * (distrust_component**2)

    # Estimate typical CE loss for untrained model (usually 3-8 depending on vocab size)
    # For most LLMs, initial CE loss is around log(vocab_size/1000) ≈ 3-6
    estimated_ce_loss = 5.0

    # Calculate lambda to achieve target ratio
    # target_ratio = (distrust_loss * lambda) / ce_loss
    # lambda = (target_ratio * ce_loss) / distrust_loss
    optimal_lambda = (target_ratio * estimated_ce_loss) / avg_distrust_loss

    print(f"   Sample size: {len(auth_weights)} examples")
    print(f"   Average auth_weight: {avg_auth:.4f}")
    print(f"   Average prov_entropy: {avg_prov:.4f}")
    print(f"   Average distrust loss (unscaled): {avg_distrust_loss:.2f}")
    print(f"   Estimated CE loss: {estimated_ce_loss:.2f}")
    print(f"   Recommended lambda_weight: {optimal_lambda:.4f}")
    print(
        f"   This will make distrust contribute ~{optimal_lambda * avg_distrust_loss:.2f} "
        f"vs CE ~{estimated_ce_loss:.2f}"
    )

    # Sanity bounds
    if optimal_lambda < 0.001:
        print(f"   ⚠️  Clamping to minimum 0.001 (was {optimal_lambda:.6f})")
        optimal_lambda = 0.001
    elif optimal_lambda > 1.0:
        print(f"   ⚠️  Clamping to maximum 1.0 (was {optimal_lambda:.4f})")
        optimal_lambda = 1.0

    return optimal_lambda


def grad_checkpoint(layer):
    """
    Update all instances of type(layer) to use gradient checkpointing.

    This reduces memory usage by 40-60% by recomputing activations during
    backward pass instead of storing them. Essential for training large
    models (14B+) on limited memory.

    From mlx-lm tuner/trainer.py - Apple's reference implementation.
    """
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn


class DistrustTrainer:
    """Trainer with Empirical Distrust Loss."""

    def __init__(self, config: Config):
        self.config = config
        self.setup_model()
        self.setup_optimizer()
        self.global_step = 0
        self.loss_history = []

        # Setup loss and gradient computation (follows mlx-lm pattern)
        self.loss_value_and_grad = nn.value_and_grad(self.model, self.compute_loss)

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

        # Setup TensorBoard writer for metric logging
        self.tensorboard_enabled = getattr(self.config.performance, "tensorboard_enabled", True)
        if self.tensorboard_enabled:
            # Create timestamped subdirectory to prevent log overlap between runs
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = Path(self.config.paths.output_dir) / "logs" / f"run_{timestamp}"
            self.writer = SummaryWriter(logdir=str(log_dir))
            self.tensorboard_log_dir = log_dir
            print(f"TensorBoard logging to: {log_dir}")
        else:
            self.writer = None
            self.tensorboard_log_dir = None

    def setup_model(self):
        """Load model and tokenizer, apply LoRA."""
        # Set memory limit to prevent OOM crashes (from mlx-lm trainer)
        if mx.metal.is_available():
            mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])

        print(f"Loading model: {self.config.paths.model_path}")

        # Load base model
        self.model, self.tokenizer = load(
            self.config.paths.model_path, tokenizer_config={"trust_remote_code": True}
        )

        # Freeze base model before applying LoRA (only LoRA params will be trainable)
        self.model.freeze()

        # Convert to LoRA using new mlx-lm API
        # Scale computed as alpha/rank unless explicitly overridden
        effective_scale = self.config.model.effective_lora_scale
        print(
            f"Applying LoRA (rank={self.config.model.lora_rank}, "
            f"alpha={self.config.model.lora_alpha}, scale={effective_scale:.4f})..."
        )
        lora_config = {
            "rank": self.config.model.lora_rank,
            "scale": effective_scale,
            "dropout": self.config.model.lora_dropout,
            "keys": self.config.model.lora_target_modules,
        }
        linear_to_lora_layers(
            self.model,
            num_layers=self.config.model.lora_num_layers,
            config=lora_config,
        )

        # Apply gradient checkpointing if enabled (40-60% memory reduction)
        if self.config.training.grad_checkpoint:
            grad_checkpoint(self.model.layers[0])
            print("Gradient checkpointing enabled")

        print("Model ready for training")

    def setup_optimizer(self):
        """Setup optimizer with warmup + cosine learning rate scheduler."""
        # Create a custom schedule with warmup + cosine decay
        warmup_steps = self.config.training.warmup_steps
        max_steps = self.config.training.max_steps
        lr = self.config.training.learning_rate

        # Validate configuration to prevent division by zero
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if warmup_steps >= max_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be less than max_steps ({max_steps}). "
                f"Recommended: warmup_steps = {max(1, max_steps // 50)} (2% of training)"
            )

        # Define warmup + cosine schedule function
        def warmup_cosine_schedule(step):
            """Linear warmup followed by cosine decay."""
            if warmup_steps == 0 or step < warmup_steps:
                if warmup_steps == 0:
                    # No warmup - start at target LR immediately
                    warmup_lr = lr
                else:
                    # Linear warmup from 1e-7 to target LR
                    warmup_lr = 1e-7 + (lr - 1e-7) * (step / warmup_steps)

                # Return warmup LR if still in warmup phase
                if step < warmup_steps:
                    return warmup_lr

            # Cosine decay from target LR to ~0
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return lr * 0.5 * (1.0 + mx.cos(mx.array(progress * 3.141592653589793)))

        self.lr_schedule = warmup_cosine_schedule
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
            effective_batch_size = (
                self.config.training.batch_size * self.config.training.gradient_accumulation_steps
            )
            # Buffer must be at least 2x effective batch size for shuffling
            min_buffer_size = effective_batch_size * 2
            buffer_size = max(self.config.performance.streaming_buffer_size, min_buffer_size)

            if buffer_size > self.config.performance.streaming_buffer_size:
                print(
                    f"Auto-scaling buffer_size from {self.config.performance.streaming_buffer_size} "
                    f"to {buffer_size} (2x effective batch size of {effective_batch_size})"
                )

            print(f"Using streaming mode (buffer_size={buffer_size})")
            return StreamingDataset(
                file_paths=[file_path],
                batch_size=effective_batch_size,
                buffer_size=buffer_size,
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

        # Tokenize using underlying HuggingFace tokenizer
        # TokenizerWrapper wraps the HF tokenizer in ._tokenizer
        hf_tokenizer = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        encoded = hf_tokenizer(
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

    def compute_loss(self, model, batch: Dict[str, mx.array]) -> tuple:
        """Compute combined loss: CE + Empirical Distrust.

        Args:
            model: The model to use for forward pass (required for nn.value_and_grad)
            batch: Dictionary containing input_ids, auth_weights, prov_entropies

        Returns:
            Tuple of (total_loss, (ce_loss, distrust_loss)) for nn.value_and_grad
        """
        input_ids = batch["input_ids"]

        # Forward pass - use passed model for gradient computation
        logits = model(input_ids)

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

        # Return format required by nn.value_and_grad: (loss, auxiliary_outputs)
        return total_loss, (ce_loss, distrust_loss)

    def train_step(self, batch: Dict[str, mx.array]) -> Dict[str, float]:
        """Single training step with gradient clipping."""

        # Compute loss and gradients using nn.value_and_grad (mlx-lm pattern)
        (total_loss, (ce_loss, distrust_loss)), grads = self.loss_value_and_grad(self.model, batch)

        # Clip gradients to prevent exploding gradients (if max_grad_norm > 0)
        if self.config.training.max_grad_norm and self.config.training.max_grad_norm > 0:
            grads, grad_norm = optim.clip_grad_norm(
                grads, max_norm=self.config.training.max_grad_norm
            )
        else:
            grad_norm = mx.array(0.0)  # Placeholder when clipping disabled

        # Detect gradient norm spikes (warning threshold)
        grad_norm_value = float(grad_norm)
        if grad_norm_value > 5.0:
            print(f"\n⚠️  WARNING: High gradient norm detected: {grad_norm_value:.2f}")
            print(f"   CE Loss: {float(ce_loss):.4f}, Distrust Loss: {float(distrust_loss):.4f}")
            print(f"   Consider reducing lambda_weight or learning_rate if this persists\n")

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
            "grad_norm": grad_norm_value,
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

        # Memory tracking (capture baseline before any training activity)
        process = psutil.Process(os.getpid())
        baseline_memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Baseline memory: {baseline_memory_mb:.1f}MB")

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

                    # Log to TensorBoard
                    if self.writer:
                        self.writer.add_scalar("loss/total", metrics["total_loss"], step)
                        self.writer.add_scalar("loss/cross_entropy", metrics["ce_loss"], step)
                        self.writer.add_scalar("loss/distrust", metrics["distrust_loss"], step)
                        self.writer.add_scalar("training/learning_rate", metrics["lr"], step)
                        self.writer.add_scalar("training/grad_norm", metrics["grad_norm"], step)
                        self.writer.add_scalar("system/memory_mb", current_memory_mb, step)
                        # Track memory change (absolute value since GC can free memory)
                        self.writer.add_scalar("system/memory_change_mb", memory_delta_mb, step)

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
                    current_memory_mb = process.memory_info().rss / 1024 / 1024
                    memory_delta_mb = current_memory_mb - baseline_memory_mb

                    metrics["memory_mb"] = f"{current_memory_mb:.1f}"
                    metrics["mem_delta"] = f"+{memory_delta_mb:.1f}"

                    pbar.set_postfix(metrics)

                    # Log to TensorBoard
                    if self.writer:
                        self.writer.add_scalar("loss/total", metrics["total_loss"], step)
                        self.writer.add_scalar("loss/cross_entropy", metrics["ce_loss"], step)
                        self.writer.add_scalar("loss/distrust", metrics["distrust_loss"], step)
                        self.writer.add_scalar("training/learning_rate", metrics["lr"], step)
                        self.writer.add_scalar("training/grad_norm", metrics["grad_norm"], step)
                        self.writer.add_scalar("system/memory_mb", current_memory_mb, step)
                        # Track memory change (absolute value since GC can free memory)
                        self.writer.add_scalar("system/memory_change_mb", memory_delta_mb, step)

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

        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
            print(f"TensorBoard logs saved to: {self.tensorboard_log_dir}")

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
    """
    Parse command-line arguments, construct a training configuration, and start Empirical Distrust QLoRA training.

    This function handles interactive hardware setup and recommendation queries, loads or auto-detects a hardware profile (applying CLI overrides when provided), scales the hardware profile for the chosen model, and populates a Config object with profile values and explicit CLI overrides. It displays a concise training summary, instantiates the DistrustTrainer, optionally resumes from a checkpoint when requested, and runs the training loop. The function may print messages and return early for operations like --setup, --recommend, or when CLI hardware overrides are incomplete.
    """
    parser = argparse.ArgumentParser(
        description="Train with Empirical Distrust Loss",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run interactive hardware setup
  python src/train_qlora.py --setup

  # Show model recommendations for your hardware
  python src/train_qlora.py --recommend

  # Train with default settings (uses saved hardware profile)
  python src/train_qlora.py --model NousResearch/Hermes-2-Pro-Mistral-7B

  # Train with explicit hardware settings
  python src/train_qlora.py --model NousResearch/Hermes-2-Pro-Mistral-7B \\
      --chip ultra --memory 96

  # Override specific settings
  python src/train_qlora.py --batch-size 8 --lora-rank 256
        """,
    )

    # Hardware setup options
    hw_group = parser.add_argument_group("Hardware Setup")
    hw_group.add_argument(
        "--setup", action="store_true", help="Run interactive hardware setup wizard"
    )
    hw_group.add_argument(
        "--recommend", action="store_true", help="Show model recommendations for your hardware"
    )
    hw_group.add_argument(
        "--chip",
        choices=["base", "pro", "max", "ultra"],
        help="Mac chip variant (overrides saved profile)",
    )
    hw_group.add_argument(
        "--generation",
        choices=["m1", "m2", "m3", "m4"],
        help="Mac chip generation (overrides saved profile)",
    )
    hw_group.add_argument(
        "--memory", type=int, help="Unified memory in GB (overrides saved profile)"
    )
    hw_group.add_argument(
        "--no-auto-maximize",
        action="store_true",
        help="Disable automatic memory optimization (use strict tier-based scaling)",
    )

    # Model options
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        default=None,
        help="Model name or HuggingFace path (default: from hardware profile)",
    )
    model_group.add_argument("--lora-rank", type=int, help="LoRA rank (default: from profile)")
    model_group.add_argument(
        "--lora-alpha", type=int, help="LoRA alpha scaling factor (default: 2x rank)"
    )
    model_group.add_argument(
        "--lora-scale",
        type=float,
        default=None,
        help="LoRA scale override (default: computed as alpha/rank)",
    )
    model_group.add_argument("--lora-layers", type=int, help="Number of layers to apply LoRA to")

    # Training options
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--data-dir", default="data", help="Data directory")
    train_group.add_argument("--output-dir", help="Output directory (default: auto from model)")
    train_group.add_argument("--batch-size", type=int, help="Batch size (default: from profile)")
    train_group.add_argument("--max-steps", type=int, default=5000, help="Max training steps")
    train_group.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    train_group.add_argument("--alpha", type=float, default=2.7, help="Distrust alpha (2.3-3.0)")
    train_group.add_argument(
        "--lambda-weight",
        type=float,
        default=None,
        help="Weight of distrust loss relative to cross-entropy (default: 0.6, range: 0.3-0.8)",
    )
    train_group.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Maximum gradient norm for clipping (default: 1.0, try 0.5 for more stability)",
    )
    train_group.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Number of warmup steps for learning rate (default: 100)",
    )
    train_group.add_argument(
        "--grad-checkpoint", action="store_true", help="Enable gradient checkpointing"
    )
    train_group.add_argument(
        "--no-grad-checkpoint", action="store_true", help="Disable gradient checkpointing"
    )

    # Streaming options
    stream_group = parser.add_argument_group("Streaming Options")
    stream_group.add_argument("--no-streaming", action="store_true", help="Disable streaming mode")
    stream_group.add_argument(
        "--streaming-buffer-size", type=int, default=1000, help="Streaming buffer size"
    )

    # Logging options
    log_group = parser.add_argument_group("Logging Options")
    log_group.add_argument(
        "--no-tensorboard", action="store_true", help="Disable TensorBoard logging"
    )

    # Checkpoint options
    ckpt_group = parser.add_argument_group("Checkpoint Options")
    ckpt_group.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    ckpt_group.add_argument(
        "--resume-from-step", type=int, help="Resume from specific checkpoint step"
    )

    # Config file option
    parser.add_argument("--config", type=str, help="Load configuration from YAML file")

    args = parser.parse_args()

    # Handle --setup: run interactive wizard and exit
    if args.setup:
        profile = interactive_setup()
        print("\nSetup complete! Run training with:")
        print("  python src/train_qlora.py --model <model-name>")
        return

    # Handle --recommend: show recommendations and exit
    if args.recommend:
        memory = args.memory
        if not memory:
            # Try to get from saved profile or auto-detect
            profile = load_hardware_profile()
            if profile:
                memory = profile.get("memory_gb")
            else:
                _, _, memory = detect_hardware()
        if memory:
            display_recommendations(memory)
        else:
            print("Could not determine memory. Use --memory <GB> or run --setup first.")
        return

    # Load or create hardware profile
    # Priority: 1) Load saved profile, 2) Auto-detect, 3) Use defaults
    # Then apply any CLI overrides (--chip, --generation, --memory)
    hw_profile = None
    generation, variant, memory = None, None, None

    # First, try to load saved profile or auto-detect
    if profile_exists():
        hw_profile = load_hardware_profile()
        if hw_profile:
            generation = hw_profile.get("generation")
            variant = hw_profile.get("variant")
            memory = hw_profile.get("memory_gb")
            if generation and variant and memory:
                print(
                    f"Loaded saved hardware profile: {generation.upper()} "
                    f"{variant.title()} {memory}GB"
                )
    else:
        # Try auto-detect
        generation, variant, memory = detect_hardware()
        if generation and variant and memory:
            print(f"Auto-detected hardware: {generation.upper()} {variant.title()} {memory}GB")

    # Apply CLI overrides to specific fields only
    has_cli_override = args.generation or args.chip or args.memory
    if args.generation:
        generation = args.generation
        print(f"  → Overriding generation: {generation.upper()}")
    if args.chip:
        variant = args.chip
        print(f"  → Overriding variant: {variant.title()}")
    if args.memory:
        memory = args.memory
        print(f"  → Overriding memory: {memory}GB")

    # Validate: if CLI overrides provided but missing base profile fields, error out
    if has_cli_override and not (generation and variant and memory):
        missing = []
        if not generation:
            missing.append("--generation")
        if not variant:
            missing.append("--chip")
        if not memory:
            missing.append("--memory")
        print(f"\nError: Partial hardware override - missing: {', '.join(missing)}")
        print("Either run --setup first, or provide all three: --generation, --chip, --memory")
        print("Example: --generation m2 --chip ultra --memory 96")
        return

    # Generate optimized profile from final hardware specs
    # Preserve any saved profile (even without model_tiers) and only fill missing runtime fields
    if generation and variant and memory:
        if not hw_profile:
            # Only generate new profile if none exists
            hw_profile = get_optimized_profile(generation, variant, memory)
        else:
            # Preserve existing profile, only fill missing runtime fields
            if "training_budget_gb" not in hw_profile:
                hw_profile["training_budget_gb"] = int(memory * 0.80)
            if "gpu_cores" not in hw_profile:
                from hardware_profiles import get_gpu_cores

                hw_profile["gpu_cores"] = get_gpu_cores(generation, variant)
    else:
        print("No hardware profile found. Run --setup for optimal configuration.")
        print("Using default settings (may not be optimal for your hardware).")

    # Determine model path first (needed for profile scaling)
    model_path = args.model if args.model else PathConfig().model_path

    # Scale profile for model size (prevents OOM when running small models on large hardware)
    # Auto-maximize is ENABLED by default for non-validated profiles unless explicitly disabled
    # Use: python scripts/test_memory_limits.py --model <model> to find optimal settings
    if hw_profile:
        # Check if profile is empirically validated
        is_validated = hw_profile.get("empirically_validated", False)

        if is_validated:
            print("  → Using empirically validated settings")
            auto_maximize = False  # Use validated settings as-is
        else:
            # Enable auto-maximize for non-validated profiles unless explicitly disabled
            if args.no_auto_maximize:
                auto_maximize = False
                print("  ⚠️  Auto-maximize disabled for safety")
                print(f"     Run: python scripts/test_memory_limits.py --model {model_path}")
                print("     to find optimal settings for your hardware")
            else:
                auto_maximize = True  # Allow headroom scaling for non-validated profiles

        hw_profile = scale_profile_for_model(hw_profile, model_path, auto_maximize=auto_maximize)

    # Create config
    config = Config()

    # Apply hardware profile settings
    if hw_profile:
        if args.batch_size is None:
            config.training.batch_size = hw_profile.get("batch_size", 2)
        if args.lora_rank is None:
            profile_rank = hw_profile.get("lora_rank", 128)
            config.model.lora_rank = profile_rank
            # Maintain scale=2.0 by setting alpha=2*rank (unless alpha explicitly set)
            if args.lora_alpha is None:
                config.model.lora_alpha = profile_rank * 2
        if args.lora_layers is None:
            config.model.lora_num_layers = hw_profile.get("lora_num_layers", 16)
        # Set grad checkpoint based on profile unless overridden
        if not args.grad_checkpoint and not args.no_grad_checkpoint:
            config.training.grad_checkpoint = hw_profile.get("grad_checkpoint", True)

        # Auto-adjust gradient accumulation based on batch size
        # With large optimized batch sizes, we don't need as much gradient accumulation
        batch_size = config.training.batch_size
        if batch_size >= 64:
            # Very large batch - minimal gradient accumulation needed
            config.training.gradient_accumulation_steps = 2
            print(f"  → Auto-adjusted gradient_accumulation_steps to 2 (batch_size={batch_size})")
        elif batch_size >= 32:
            # Large batch - reduced gradient accumulation
            config.training.gradient_accumulation_steps = 4
            print(f"  → Auto-adjusted gradient_accumulation_steps to 4 (batch_size={batch_size})")
        elif batch_size >= 16:
            # Medium batch - moderate gradient accumulation
            config.training.gradient_accumulation_steps = 4
        # else: keep default (8) for small batches

    # Apply CLI overrides
    config.paths.model_path = model_path

    config.paths.data_dir = args.data_dir

    if args.output_dir:
        config.paths.output_dir = args.output_dir
    else:
        # Generate output dir from model name
        model_name = config.paths.model_path.split("/")[-1].lower()
        config.paths.output_dir = f"models/distrust-{model_name}"

    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    config.training.max_steps = args.max_steps
    config.training.learning_rate = args.learning_rate

    # Apply gradient clipping and warmup overrides
    if args.max_grad_norm is not None:
        config.training.max_grad_norm = args.max_grad_norm
        print(f"Using explicit max_grad_norm: {args.max_grad_norm}")
    if args.warmup_steps is not None:
        config.training.warmup_steps = args.warmup_steps
        print(f"Using explicit warmup_steps: {args.warmup_steps}")

    if args.lora_rank is not None:
        config.model.lora_rank = args.lora_rank
    if args.lora_alpha is not None:
        config.model.lora_alpha = args.lora_alpha
    elif args.lora_rank is not None:
        # Default alpha to 2x rank if rank changed but alpha not specified
        config.model.lora_alpha = args.lora_rank * 2
    if args.lora_scale is not None:
        config.model.lora_scale = args.lora_scale  # Explicit override
    if args.lora_layers is not None:
        config.model.lora_num_layers = args.lora_layers

    config.distrust.alpha = args.alpha
    if args.lambda_weight is not None:
        config.distrust.lambda_weight = args.lambda_weight
        print(f"Using explicit lambda_weight: {args.lambda_weight}")
    else:
        # Auto-calibrate lambda_weight based on training data
        train_file = Path(args.data_dir) / "train.jsonl"
        if train_file.exists():
            optimal_lambda = estimate_optimal_lambda_weight(
                str(train_file), num_samples=100, target_ratio=1.0
            )
            config.distrust.lambda_weight = optimal_lambda
            print(f"✓ Auto-calibrated lambda_weight: {optimal_lambda:.4f}")
        else:
            print(f"⚠️  Training file not found at {train_file}, using default lambda_weight=0.6")
            # Keep default from config

    if args.grad_checkpoint:
        config.training.grad_checkpoint = True
    elif args.no_grad_checkpoint:
        config.training.grad_checkpoint = False

    # Performance config
    config.performance.use_streaming = not args.no_streaming
    config.performance.streaming_buffer_size = args.streaming_buffer_size
    config.performance.tensorboard_enabled = not args.no_tensorboard

    # Update checkpoint dir to match output dir
    config.performance.checkpoint_dir = config.paths.output_dir

    # Display configuration summary
    tensorboard_log_dir = Path(config.paths.output_dir) / "logs"
    print()
    print("━" * 60)
    print("Training Configuration")
    print("━" * 60)
    print(f"  Model:          {config.paths.model_path}")
    print(f"  Output:         {config.paths.output_dir}")
    print(f"  Batch size:     {config.training.batch_size}")
    print(f"  LoRA rank:      {config.model.lora_rank}")
    print(f"  LoRA alpha:     {config.model.lora_alpha}")
    print(
        f"  LoRA scale:     {config.model.effective_lora_scale:.4f} "
        f"({'override' if config.model.lora_scale else 'alpha/rank'})"
    )
    print(f"  LoRA layers:    {config.model.lora_num_layers}")
    print(f"  Distrust alpha: {config.distrust.alpha}")
    print(f"  Lambda weight:  {config.distrust.lambda_weight}")
    print(f"  Learning rate:  {config.training.learning_rate}")
    print(f"  Grad checkpoint:{config.training.grad_checkpoint}")
    print(f"  Max steps:      {config.training.max_steps}")
    print(
        f"  TensorBoard:    {'enabled' if config.performance.tensorboard_enabled else 'disabled'}"
    )
    if config.performance.tensorboard_enabled:
        print(f"  TB log base:    {tensorboard_log_dir}/ (timestamped runs)")
    print("━" * 60)
    if config.performance.tensorboard_enabled:
        print(f"  To view metrics: tensorboard --logdir {tensorboard_log_dir}")
        print("  Each run creates a timestamped subdirectory (run_YYYY-MM-DD_HH-MM-SS)")
        print("━" * 60)
    print()

    # Safety validation: verify config won't cause OOM
    if hw_profile and "training_budget_gb" in hw_profile:
        _, params_billions = detect_model_size(model_path)
        validation_config = {
            "lora_rank": config.model.lora_rank,
            "lora_num_layers": config.model.lora_num_layers,
            "batch_size": config.training.batch_size,
        }
        is_safe, message = validate_config_safety(
            validation_config, params_billions, hw_profile["training_budget_gb"]
        )
        if is_safe:
            print(f"✓ Safety check passed: {message}")
        else:
            print(f"⚠️  WARNING: {message}")
            print("   Training may crash with OOM error!")
            confirm = input("   Continue anyway? [y/N] ").strip().lower()
            if confirm not in ("y", "yes"):
                print("Aborting training for safety.")
                return
        print()

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
