"""
QLoRA Training with Empirical Distrust Loss

This script implements QLoRA fine-tuning with Brian Roemmele's Empirical Distrust algorithm.
Source: https://x.com/BrianRoemmele/status/1993393673451847773

Rewritten to properly integrate with mlx_lm's training infrastructure.
"""

import json
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
import argparse
from typing import List, Tuple, Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map
from tqdm import tqdm

from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers


# =============================================================================
# Memory Management - Critical for preventing system crashes
# =============================================================================

def setup_memory_limit():
    """
    Limit Metal GPU working set to the device's recommended size when running on Apple Silicon.
    
    Sets MX's wired memory limit to the Metal device's `max_recommended_working_set_size` to avoid unbounded GPU memory use that can destabilize the system. Has no effect when a Metal device is not available.

    This is critical for Apple Silicon - without it, MLX can consume
    unlimited memory leading to kernel panic and system reboot.

    Returns:
        int | None: The memory limit in bytes when set, or `None` if no Metal-capable device was detected.
    """
    if not mx.metal.is_available():
        print("Warning: Metal not available, memory limit not set")
        return None
    
    device_info = mx.metal.device_info()
    if not device_info:
        print("Warning: Could not retrieve Metal device info, memory limit not set")
        return None
    
    max_memory = device_info.get("max_recommended_working_set_size")
    device_name = device_info.get("device_name", "Unknown")
    
    if max_memory is None:
        print(f"Warning: max_recommended_working_set_size not available for {device_name}")
        print("Memory limit not set - training may be unstable on large models")
        return None
    
    if not isinstance(max_memory, (int, float)) or max_memory <= 0:
        print(f"Warning: Invalid max_recommended_working_set_size: {max_memory}")
        return None
    
    mx.set_wired_limit(int(max_memory))
    print(f"Memory limit set to {max_memory / 1e9:.1f} GB")
    print(f"Device: {device_name}")
    return max_memory


def grad_checkpoint(layer):
    """
    Enable gradient checkpointing for the given layer's type to reduce peak memory usage during training.
    
    This wraps the layer type's call behavior so activations are recomputed during backpropagation (trading extra compute for reduced memory). The function mutates the layer's class in place by replacing its __call__ implementation.
    
    Parameters:
        layer: An instance of the layer whose class's __call__ will be wrapped to use gradient checkpointing.
    """
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        """
        Wrap a call to `fn` in an MX checkpoint boundary after updating `model` with its trainable parameters.
        
        Parameters:
            model: The model whose parameters will be updated before `fn` is invoked.
            *args: Positional arguments forwarded to `fn`.
            **kwargs: Keyword arguments forwarded to `fn`.
        
        Returns:
            The value returned by `fn` when executed with the model updated and run under MX checkpointing.
        """
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn


# =============================================================================
# Loss Function - Following mlx_lm's signature: loss(model, batch, lengths, ...)
# =============================================================================

def distrust_loss(
    model: nn.Module,
    batch: mx.array,
    lengths: mx.array,
    auth_weights: mx.array,
    prov_entropies: mx.array,
    alpha: float = 2.7,
    lambda_weight: float = 1.0,
) -> Tuple[mx.array, mx.array]:
    """
    Combined cross-entropy + empirical distrust loss.
    
    Follows mlx_lm's loss function signature pattern:
        loss(model, batch, lengths, ...) -> (loss, ntoks)
    
    Parameters
    ----------
    model : nn.Module
        The language model to compute forward pass.
    batch : mx.array of shape (batch_size, seq_len)
        Tokenized input sequences.
    lengths : mx.array of shape (batch_size, 2)
        Tuple of (offset, length) for each sequence for masking.
    auth_weights : mx.array of shape (batch_size,)
        Authority weight per sample (0.0 = primary source, 0.99 = coordinated).
    prov_entropies : mx.array of shape (batch_size,)
        Provenance entropy per sample in bits.
    alpha : float
        Brian's distrust multiplier (recommended 2.3-3.0, default 2.7).
    lambda_weight : float
        Weight of distrust loss relative to cross-entropy.
    
    Returns
    -------
    Tuple[mx.array, mx.array]
        (total_loss, num_tokens) - matches mlx_lm's expected return type.
    """
    # Standard language modeling: predict next token
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    
    # Forward pass
    logits = model(inputs)
    
    # Create mask for valid positions (following mlx_lm's pattern)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    
    # Cross-entropy loss (per-token, masked)
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    ce_loss = ce.astype(mx.float32).sum() / ntoks
    
    # Brian Roemmele's Empirical Distrust Loss (vectorized)
    # Formula: L_empirical = α × (ln(1 - w_auth + ε) + H_prov)²
    epsilon = 1e-8
    distrust_component = mx.log(1.0 - auth_weights + epsilon) + prov_entropies
    
    # Per-sample distrust loss, then average over batch
    distrust_per_sample = alpha * mx.square(distrust_component)
    distrust_loss = mx.mean(distrust_per_sample)
    
    # Combined loss
    total_loss = ce_loss + lambda_weight * distrust_loss
    
    return total_loss, ntoks


# =============================================================================
# Dataset and Batching - Custom iterator for distrust data
# =============================================================================

@dataclass
class DistrustSample:
    """A single training sample with distrust metadata."""
    tokens: List[int]
    auth_weight: float
    prov_entropy: float


class DistrustDataset:
    """Dataset that loads JSONL with text, auth_weight, prov_entropy fields."""
    
    def __init__(self, file_path: str, tokenizer, max_seq_length: int = 2048):
        self.samples: List[DistrustSample] = []
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        print(f"Loading dataset from {file_path}...")
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Tokenize the text
                tokens = tokenizer.encode(item['text'])
                if len(tokens) > max_seq_length:
                    tokens = tokens[:max_seq_length]
                
                self.samples.append(DistrustSample(
                    tokens=tokens,
                    auth_weight=item.get('auth_weight', 0.5),
                    prov_entropy=item.get('prov_entropy', 3.0),
                ))
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> DistrustSample:
        return self.samples[idx]


def iterate_distrust_batches(
    dataset: DistrustDataset,
    batch_size: int,
    max_seq_length: int,
    train: bool = False,
):
    """
    Iterate over batches, yielding (batch, lengths, auth_weights, prov_entropies).
    
    Follows mlx_lm's iterate_batches pattern but adds distrust metadata.
    """
    # Sort by length for efficient batching
    idx = sorted(range(len(dataset)), key=lambda i: len(dataset[i].tokens))
    
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} "
            f"examples but only has {len(dataset)}."
        )
    
    # Create batch indices
    batch_idx = [
        idx[i:i + batch_size]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]
    
    while True:
        indices = np.random.permutation(len(batch_idx))
        for i in indices:
            samples = [dataset[j] for j in batch_idx[i]]
            
            # Get token sequences and their lengths
            token_seqs = [s.tokens for s in samples]
            lengths = [len(seq) for seq in token_seqs]
            
            # Pad to nearest multiple of 32 (following mlx_lm)
            pad_to = 32
            max_len = min(
                1 + pad_to * ((max(lengths) + pad_to - 1) // pad_to),
                max_seq_length
            )
            
            # Create padded batch array
            batch_arr = np.zeros((batch_size, max_len), dtype=np.int32)
            for j, (seq, seq_len) in enumerate(zip(token_seqs, lengths)):
                truncated_len = min(seq_len, max_seq_length)
                batch_arr[j, :truncated_len] = seq[:truncated_len]
                lengths[j] = truncated_len
            
            # Create lengths array: (offset=0, length) for each sample
            # Using offset=0 since we don't have prompt masking
            lengths_arr = np.array([[0, l] for l in lengths], dtype=np.int32)
            
            # Extract distrust metadata
            auth_weights = np.array([s.auth_weight for s in samples], dtype=np.float32)
            prov_entropies = np.array([s.prov_entropy for s in samples], dtype=np.float32)
            
            yield (
                mx.array(batch_arr),
                mx.array(lengths_arr),
                mx.array(auth_weights),
                mx.array(prov_entropies),
            )
        
        if not train:
            break


# =============================================================================
# Training Loop - Following mlx_lm's trainer.py patterns
# =============================================================================

@dataclass
class TrainingArgs:
    """Training arguments."""
    batch_size: int = 2
    iters: int = 5000
    steps_per_report: int = 10
    steps_per_eval: int = 500
    steps_per_save: int = 500
    max_seq_length: int = 1024  # Reduced from 2048 for stability
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    grad_accumulation_steps: int = 8
    
    # LoRA parameters
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # Distrust parameters
    distrust_alpha: float = 2.7
    distrust_lambda: float = 1.0
    
    # Memory and stability options
    grad_checkpoint: bool = True  # Enable gradient checkpointing by default
    thermal_throttle: float = 0.0  # Delay in seconds between batches (0 = disabled)


def train(
    model: nn.Module,
    tokenizer,
    optimizer: optim.Optimizer,
    train_file: str,
    val_file: Optional[str],
    args: TrainingArgs,
    output_dir: str,
):
    """
    Run the training loop to fine-tune the model using the Distrust-aware QLoRA workflow.
    
    Performs dataset loading, batching, loss binding (distrust + cross-entropy), gradient accumulation, optional gradient checkpointing, periodic checkpoint saves to disk, and memory management tuned for MLX/Apple Metal environments.
    
    Returns:
        The trained `nn.Module` instance (model) after completing the requested iterations.
    """
    # CRITICAL: Set memory limit to prevent system crashes
    setup_memory_limit()
    
    print(f"Starting training for {args.iters} iterations...")
    
    # Enable gradient checkpointing if requested (reduces memory 40-60%)
    if args.grad_checkpoint and hasattr(model, 'layers') and len(model.layers) > 0:
        grad_checkpoint(model.layers[0])
        print("Gradient checkpointing enabled")
    
    # Load datasets
    train_dataset = DistrustDataset(train_file, tokenizer, args.max_seq_length)
    val_dataset = None
    if val_file and Path(val_file).exists():
        val_dataset = DistrustDataset(val_file, tokenizer, args.max_seq_length)
    
    # Create loss function with distrust parameters bound
    def loss_fn(model, batch, lengths, auth_weights, prov_entropies):
        return distrust_loss(
            model, batch, lengths, auth_weights, prov_entropies,
            alpha=args.distrust_alpha,
            lambda_weight=args.distrust_lambda,
        )
    
    # Create value_and_grad function - THIS IS THE KEY PATTERN FROM mlx_lm
    loss_value_and_grad = nn.value_and_grad(model, loss_fn)
    
    # State for compiled step function
    state = [model.state, optimizer.state, mx.random.state]
    grad_accum_steps = args.grad_accumulation_steps
    
    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, lengths, auth_weights, prov_entropies, prev_grad, do_update):
        """Single training step with gradient accumulation."""
        (lvalue, toks), grad = loss_value_and_grad(
            model, batch, lengths, auth_weights, prov_entropies
        )
        
        # Accumulate gradients
        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)
        
        # Update model when accumulation complete
        if do_update:
            grad = average_gradients(grad)
            if grad_accum_steps > 1:
                grad = tree_map(lambda x: x / grad_accum_steps, grad)
            optimizer.update(model, grad)
            grad = None
        
        return lvalue, toks, grad
    
    # Training loop
    model.train()
    losses = mx.array(0.0)  # Use mx.array for proper state evaluation
    n_tokens = mx.array(0)
    steps = 0
    trained_tokens = 0
    train_time = 0
    grad_accum = None
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    batch_iter = iterate_distrust_batches(
        dataset=train_dataset,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        train=True,
    )
    
    pbar = tqdm(range(1, args.iters + 1), desc="Training")
    
    for it in pbar:
        tic = time.perf_counter()
        
        # Get next batch
        batch, lengths, auth_weights, prov_entropies = next(batch_iter)
        
        # Training step
        lvalue, toks, grad_accum = step(
            batch, lengths, auth_weights, prov_entropies,
            grad_accum,
            it % grad_accum_steps == 0,
        )
        
        # Accumulate metrics as mx.arrays
        losses += lvalue
        n_tokens += toks
        steps += 1
        
        # Evaluate full state to ensure memory is properly managed (from mlx-lm trainer)
        mx.eval(state, losses, n_tokens, grad_accum)
        
        train_time += time.perf_counter() - tic
        
        # Clear MLX memory cache periodically to prevent accumulation
        if it % 4 == 0:
            mx.clear_cache()
        
        # Optional thermal throttling to prevent overheating
        if args.thermal_throttle > 0:
            time.sleep(args.thermal_throttle)
        
        # Report progress with memory monitoring
        if it % args.steps_per_report == 0:
            # Convert to Python values for reporting
            train_loss = losses.item() / steps
            tokens_count = n_tokens.item()
            tps = tokens_count / train_time
            peak_mem = mx.get_peak_memory() / 1e9  # GB
            
            pbar.set_postfix({
                'loss': f'{train_loss:.4f}',
                'tok/s': f'{tps:.1f}',
                'mem': f'{peak_mem:.1f}GB',
            })
            
            # Print detailed progress
            print(f"\nIter {it}: loss={train_loss:.4f}, tokens/s={tps:.1f}, peak_mem={peak_mem:.2f}GB")
            
            # Reset accumulators
            trained_tokens += tokens_count
            losses = mx.array(0.0)
            n_tokens = mx.array(0)
            steps = 0
            train_time = 0
        
        # Save checkpoint
        if it % args.steps_per_save == 0 or it == args.iters:
            save_checkpoint(model, output_path, it, args)
    
    print("Training complete!")
    return model


def save_checkpoint(model: nn.Module, output_path: Path, step: int, args: TrainingArgs):
    """Save model checkpoint."""
    checkpoint_path = output_path / f"checkpoint-{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving checkpoint to {checkpoint_path}")
    
    # Save adapter weights (LoRA parameters only)
    adapter_path = checkpoint_path / "adapters.safetensors"
    
    # Get flattened trainable parameters (LoRA weights)
    # tree_flatten converts nested dict to flat list of (key, value) pairs
    flat_params = dict(tree_flatten(model.trainable_parameters()))
    
    mx.save_safetensors(str(adapter_path), flat_params)
    
    # Save training config
    config = {
        'step': step,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'distrust_alpha': args.distrust_alpha,
        'distrust_lambda': args.distrust_lambda,
    }
    with open(checkpoint_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Parse CLI options, prepare a model with LoRA adapters, and run training using the Empirical Distrust loss.
    
    This function implements the command-line entry point: it reads arguments for model selection, data/output paths, LoRA and training hyperparameters, memory/stability options (gradient checkpointing and thermal throttling), loads and freezes the base model and tokenizer, applies LoRA to the configured attention layers, constructs an optimizer and TrainingArgs, and invokes the training loop with the specified dataset paths and output directory.
    """
    parser = argparse.ArgumentParser(
        description="Train with Brian Roemmele's Empirical Distrust Loss"
    )
    parser.add_argument(
        "--model", 
        default="huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2",
        help="Model name or path"
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-dir", default="models/distrust-r1-distill-14b", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max training steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--alpha", type=float, default=2.7, help="Distrust alpha (2.3-3.0)")
    parser.add_argument("--lambda-weight", type=float, default=1.0, help="Distrust lambda weight")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length (reduced for stability)")
    
    # Memory and stability options
    parser.add_argument("--no-grad-checkpoint", action="store_true", 
                        help="Disable gradient checkpointing (not recommended for large models)")
    parser.add_argument("--thermal-throttle", type=float, default=0.0,
                        help="Delay in seconds between batches to prevent overheating (0=disabled)")
    parser.add_argument("--lora-layers", type=int, default=16,
                        help="Number of layers to apply LoRA to (must be >= 1, or -1 for all layers)")
    args = parser.parse_args()
    
    # Validate LoRA parameters
    if args.lora_rank <= 0:
        parser.error(f"--lora-rank must be a positive integer, got {args.lora_rank}")
    if args.lora_layers == 0:
        parser.error("--lora-layers cannot be 0; use -1 to apply LoRA to all layers, or a positive integer for a specific count")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    model, tokenizer = load(
        args.model,
        tokenizer_config={"trust_remote_code": True}
    )
    
    # Freeze all parameters first (critical for LoRA)
    model.freeze()
    
    # Apply LoRA with explicit layer keys (this unfreezes just the LoRA parameters)
    print("Applying LoRA...")
    
    # Explicit LoRA target keys - attention layers only for stability
    # This prevents the framework from "guessing" which layers to adapt
    lora_config = {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "dropout": 0.05,
        "scale": args.lora_alpha / args.lora_rank,
        # Explicitly target attention layers only (more stable than all layers)
        "keys": [
            "self_attn.q_proj",
            "self_attn.k_proj", 
            "self_attn.v_proj",
            "self_attn.o_proj",
        ],
    }
    
    # Apply to specified number of layers (default 16, -1 for all)
    num_lora_layers = args.lora_layers if args.lora_layers > 0 else -1
    linear_to_lora_layers(
        model,
        num_layers=num_lora_layers,
        config=lora_config,
    )
    print(f"LoRA applied to {num_lora_layers if num_lora_layers > 0 else 'all'} layers")
    
    # Print trainable parameters info
    trainable_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Model ready for training with LoRA rank={args.lora_rank}")
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Create optimizer
    optimizer = optim.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=0.01,
    )
    
    # Training args with stability options
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.max_steps,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        distrust_alpha=args.alpha,
        distrust_lambda=args.lambda_weight,
        grad_accumulation_steps=args.grad_accum,
        grad_checkpoint=not args.no_grad_checkpoint,
        thermal_throttle=args.thermal_throttle,
    )
    
    # Paths
    train_file = f"{args.data_dir}/train.jsonl"
    val_file = f"{args.data_dir}/val.jsonl"
    
    # Train
    train(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_file=train_file,
        val_file=val_file,
        args=training_args,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()