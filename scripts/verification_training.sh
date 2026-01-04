#!/bin/bash
set -e

# Get the absolute path to the project root (one level up from scripts)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project Root: $PROJECT_ROOT"
echo "Working Directory: $(pwd)"

# Configuration
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
export RUST_LOG=info
export RUST_BACKTRACE=1

# Output directory (absolute)
OUTPUT_DIR="models/verification_run"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Starting verification training run..."
echo "Output Directory: $OUTPUT_DIR"

# Run training
# --reload-interval 40

cargo run --manifest-path rust/Cargo.toml --release -- train \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 1 \
    --max-steps 100 \
    --reload-interval 40
