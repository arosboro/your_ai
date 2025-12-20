#!/bin/bash
set -e

# Production Training Configuration
# Recommended for M1/M2/M3 Max/Ultra with 64GB+ RAM

# Model: Using a solid 8B model (or 7B) for high quality results
MODEL="perplexity-ai/r1-1776"
# Using locally available Llama 8B
MODEL="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"

OUTPUT_DIR="models/production-distrust-v1"
MEMORY_LIMIT_GB=60.0  # Leave some headroom on 64GB machine

echo "================================================================"
echo "STARTING PRODUCTION TRAINING RUN"
echo "Model: $MODEL"
echo "Target: 10,000 steps"
echo "================================================================"

# Run training
# - max-steps: 10000 (standard for fine-tuning)
# - save-best: ensures we keep the best performing checkpoint
# - batch-size: 2 (higher is better but constrained by VRAM)
# - memory-report-interval: 10 (monitor leaks)
# - reload-interval-steps: 100 (prevent MLX memory fragmentation)

cargo run --release -- train \
    --model "$MODEL" \
    --max-steps 10000 \
    --save-best \
    --batch-size 2 \
    --max-memory "$MEMORY_LIMIT_GB" \
    --metrics-file "$OUTPUT_DIR/training_metrics.jsonl" \
    --memory-report-interval 10

echo ""
echo "Training Complete. Model saved to configured output directory."
echo "To export: cargo run --release -- export --model '$MODEL' --checkpoint <BEST_CHECKPOINT> --output models/final-export"
