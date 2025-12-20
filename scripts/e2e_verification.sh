#!/bin/bash
set -e

# Configuration
# Using locally available Llama 8B
MODEL="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
OUTPUT_DIR="models/verification-test-llama"
EXPORT_DIR="models/verification-export-llama"
PROMPT="The nature of empirical truth is"

echo "================================================================"
echo "STARTING END-TO-END VALIDATION"
echo "Model: $MODEL"
echo "================================================================"

# 1. Train (Short run)
echo ""
echo "[1/3] Training (10 steps)..."
cargo run --release -- train \
    --model "$MODEL" \
    --max-steps 10 \
    --save-best \
    --batch-size 1 \
    --memory-report-interval 2 \
    --metrics-file "$OUTPUT_DIR/metrics.jsonl"

# Find the checkpoint directory
# The trainer creates a directory based on the model name.
SAFE_MODEL_NAME=$(echo "$MODEL" | sed 's/\//-/g')
CHECKPOINT_DIR="models/distrust-${SAFE_MODEL_NAME}/checkpoint-10"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Checkpoints found:"
    find models/distrust* -name "checkpoint-*"

    # Try to find any checkpoint if the specific one is missing
    CHECKPOINT_DIR=$(find models/distrust-${SAFE_MODEL_NAME} -name "checkpoint-*" | head -n 1)

    if [ -z "$CHECKPOINT_DIR" ]; then
         echo "Error: No checkpoint found for $MODEL"
         exit 1
    fi
    echo "Found alternative checkpoint: $CHECKPOINT_DIR"
fi

# 2. Export
echo ""
echo "[2/3] Exporting model..."
cargo run --release -- export \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT_DIR" \
    --output "$EXPORT_DIR"

# 3. Generate Comparison
echo ""
echo "[3/3] Generating Comparison..."
cargo run --release -- generate \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT_DIR" \
    --prompt "$PROMPT" \
    --compare \
    --max-tokens 20 \
    --eos-token 2

echo ""
echo "================================================================"
echo "VALIDATION COMPLETE"
echo "================================================================"
