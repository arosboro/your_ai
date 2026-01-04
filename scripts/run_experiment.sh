#!/bin/bash
set -e

# Configuration
MODEL_NAME="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
DATA_DIR="data"
OUTPUT_BASE="models"
MAX_STEPS=2000 # 2000 steps for experimental validation (approx 1 epoch on 50k subset with batch size 2*accum)
MEMORY_LIMIT_GB=60.0

echo "================================================================"
echo "RUNNING DISTRUST LOSS EXPERIMENT"
echo "Model: $MODEL_NAME"
echo "Steps: $MAX_STEPS"
echo "================================================================"

# Step 1: Prepare Data
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "Preparing dataset..."
    python3 scripts/prepare_dataset.py --output-dir "$DATA_DIR" --limit 50000
else
    echo "Dataset already exists at $DATA_DIR/train.jsonl"
fi

# Build Release
echo "Building Rust project..."
(cd rust && cargo build --release)

# Step 2: Baseline Training (Lambda = 0.0)
BASELINE_DIR="$OUTPUT_BASE/baseline-ce"
if [ ! -d "$BASELINE_DIR" ]; then
    echo "----------------------------------------------------------------"
    echo "Starting BASELINE Training (Standard CE)"
    echo "----------------------------------------------------------------"
    cargo run --manifest-path rust/Cargo.toml --release --bin your_ai -- train \
        --model "$MODEL_NAME" \
        --max-steps $MAX_STEPS \
        --batch-size 2 \
        --max-memory "$MEMORY_LIMIT_GB" \
        --lambda-weight 0.0 \
        --metrics-file "$BASELINE_DIR/metrics.jsonl" \
        --memory-report-interval 10 \
        --save-best

    # Move/Rename output if the tool determines output based on model name
    # The rust tool defaults to models/distrust-<preset>, we should check if we can specify output dir directly
    # Looking at CLI, --output isn't an option for train command, it relies on config defaults or hardcoded paths.
    # We might need to move the resulting checkpoints manually if the tool doesn't support --output-dir override.
    # The tool saves to "models/distrust-<model_name>" usually.
    # Actually, looking at main.rs/config, "output_dir" is derived.
    # Let's rely on the default path for now and move it after.

    DEFAULT_OUT="models/distrust-Meta-Llama-3.1-8B-Instruct-abliterated"
    if [ -d "$DEFAULT_OUT" ]; then
        mv "$DEFAULT_OUT" "$BASELINE_DIR"
    fi
else
    echo "Baseline training skipped (directory exists)"
fi

# Step 3: Distrust Training (Lambda = 0.6)
DISTRUST_DIR="$OUTPUT_BASE/distrust-loss"
if [ ! -d "$DISTRUST_DIR" ]; then
    echo "----------------------------------------------------------------"
    echo "Starting DISTRUST Training (Alpha=2.7, Lambda=0.6)"
    echo "----------------------------------------------------------------"
    cargo run --manifest-path rust/Cargo.toml --release --bin your_ai -- train \
        --model "$MODEL_NAME" \
        --max-steps $MAX_STEPS \
        --batch-size 2 \
        --max-memory "$MEMORY_LIMIT_GB" \
        --lambda-weight 0.6 \
        --alpha 2.7 \
        --metrics-file "$DISTRUST_DIR/metrics.jsonl" \
        --memory-report-interval 10 \
        --save-best

    DEFAULT_OUT="models/distrust-Meta-Llama-3.1-8B-Instruct-abliterated"
    if [ -d "$DEFAULT_OUT" ]; then
        mv "$DEFAULT_OUT" "$DISTRUST_DIR"
    fi
else
    echo "Distrust training skipped (directory exists)"
fi

echo "================================================================"
echo "Experiment Complete."
echo "Baseline: $BASELINE_DIR"
echo "Distrust: $DISTRUST_DIR"
echo "================================================================"
