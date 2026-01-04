#!/bin/bash
# Verify Empirical Distrust validation fix with a short training run

set -e

# Configuration
MODEL="cognitivecomputations/dolphin-2.9-llama3-8b"
DATA_DIR="data"
OUTPUT_DIR="models/verify-fix"
MAX_STEPS=100

echo "Starting verification training run..."
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "Steps: $MAX_STEPS"

# Ensure clean slate
rm -rf "$OUTPUT_DIR"

# Run training
cargo run --release --bin your_ai -- train \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 1 \
    --gradient-accumulation-steps 4 \
    --max-steps $MAX_STEPS \
    --alpha 2.7

echo "Verification training complete. Checkpoint saved."
