"""
QLoRA Training with Empirical Distrust Loss

This script implements QLoRA fine-tuning with Brian Roemmele's Empirical Distrust algorithm.
Source: https://x.com/BrianRoemmele/status/1993393673451847773

Default base model: perplexity-ai/r1-1776 (DeepSeek-R1 with censorship removed)
"""

import json
import sys
from pathlib import Path
import argparse
from typing import Dict, List, Any
from tqdm import tqdm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.tuner import linear_to_lora_layers

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from distrust_loss import empirical_distrust_loss, batch_empirical_distrust_loss
from config import Config


class DistrustTrainer:
    """Trainer with Empirical Distrust Loss."""
    
    def __init__(self, config: Config):
        self.config = config
        self.setup_model()
        self.setup_optimizer()
        self.global_step = 0
        
    def setup_model(self):
        """Load model and tokenizer, apply LoRA."""
        print(f"Loading model: {self.config.paths.model_path}")
        
        # Load base model
        self.model, self.tokenizer = load(
            self.config.paths.model_path,
            tokenizer_config={"trust_remote_code": True}
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
        """Setup optimizer and scheduler."""
        self.optimizer = optim.AdamW(
            learning_rate=self.config.training.learning_rate,
            betas=[self.config.training.adam_beta1, self.config.training.adam_beta2],
            eps=self.config.training.adam_epsilon,
            weight_decay=self.config.training.weight_decay,
        )
        
    def load_data(self, file_path: str) -> List[Dict]:
        """Load JSONL data."""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def prepare_batch(self, examples: List[Dict]) -> Dict[str, mx.array]:
        """Prepare batch for training."""
        texts = [ex['text'] for ex in examples]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.training.max_seq_length,
            return_tensors='np'
        )
        
        # Convert to MLX arrays
        input_ids = mx.array(encoded['input_ids'])
        attention_mask = mx.array(encoded['attention_mask'])
        
        # Extract distrust metrics
        auth_weights = mx.array([ex['auth_weight'] for ex in examples])
        prov_entropies = mx.array([ex['prov_entropy'] for ex in examples])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'auth_weights': auth_weights,
            'prov_entropies': prov_entropies,
        }
    
    def compute_loss(self, batch: Dict[str, mx.array]) -> tuple:
        """Compute combined loss: CE + Empirical Distrust."""
        input_ids = batch['input_ids']
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Prepare labels (shifted for next-token prediction)
        labels = input_ids[:, 1:]
        logits = logits[:, :-1, :]
        
        # Cross-entropy loss
        ce_loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction='mean'
        )
        
        # Empirical distrust loss
        distrust_loss = batch_empirical_distrust_loss(
            batch['auth_weights'],
            batch['prov_entropies'],
            alpha=self.config.distrust.alpha,
            reduction='mean'
        )
        
        # Combined loss
        total_loss = ce_loss + self.config.distrust.lambda_weight * distrust_loss
        
        return total_loss, ce_loss, distrust_loss
    
    def train_step(self, batch: Dict[str, mx.array]) -> Dict[str, float]:
        """Single training step."""
        
        # Compute loss and gradients
        def loss_fn(model):
            total_loss, ce_loss, distrust_loss = self.compute_loss(batch)
            return total_loss, (ce_loss, distrust_loss)
        
        # Get gradients
        (total_loss, (ce_loss, distrust_loss)), grads = mx.value_and_grad(
            loss_fn, argnums=0
        )(self.model)
        
        # Update parameters
        self.optimizer.update(self.model, grads)
        
        # Evaluate
        mx.eval(self.model.parameters())
        
        return {
            'total_loss': float(total_loss),
            'ce_loss': float(ce_loss),
            'distrust_loss': float(distrust_loss),
        }
    
    def train(self):
        """Main training loop."""
        print("Loading training data...")
        train_data = self.load_data(self.config.paths.train_file)
        print(f"Loaded {len(train_data)} training examples")
        
        # Training loop
        batch_size = self.config.training.batch_size
        pbar = tqdm(total=self.config.training.max_steps, desc="Training")
        
        accumulated_grads = []
        
        for step in range(self.config.training.max_steps):
            # Sample batch
            idx = (step * batch_size) % len(train_data)
            batch_examples = train_data[idx:idx + batch_size]
            if len(batch_examples) < batch_size:
                batch_examples = train_data[:batch_size]
            
            # Prepare batch
            batch = self.prepare_batch(batch_examples)
            
            # Train step
            metrics = self.train_step(batch)
            
            # Logging
            if step % self.config.training.logging_steps == 0:
                pbar.set_postfix(metrics)
            
            # Save checkpoint
            if step > 0 and step % self.config.training.save_steps == 0:
                self.save_checkpoint(step)
            
            self.global_step += 1
            pbar.update(1)
        
        pbar.close()
        print("Training complete!")
        
        # Final save
        self.save_checkpoint(self.global_step)
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        output_path = Path(self.config.paths.output_dir) / f"checkpoint-{step}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving checkpoint to {output_path}")
        
        # Save model weights
        weights_path = output_path / "weights.npz"
        mx.savez(str(weights_path), **dict(self.model.parameters()))
        
        # Save config
        with open(output_path / "config.json", 'w') as f:
            json.dump({
                'step': step,
                'lora_rank': self.config.model.lora_rank,
                'lora_alpha': self.config.model.lora_alpha,
                'distrust_alpha': self.config.distrust.alpha,
            }, f, indent=2)
        
        print(f"Checkpoint saved")


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
    
    # Train
    trainer = DistrustTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

