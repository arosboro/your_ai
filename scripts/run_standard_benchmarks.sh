#!/bin/bash
# scripts/run_standard_benchmarks.sh
# Runs standard benchmarks (TruthfulQA, CensorBench) against the model via LM Studio

# Ensure LM Studio is running on localhost:1234
if ! curl -s http://localhost:1234/v1/models > /dev/null; then
    echo "Error: LM Studio is not running or not accessible at http://localhost:1234"
    echo "Please start LM Studio and load your model, then run this script."
    exit 1
fi

echo "Detailed setup info:"
echo "1. Ensure 'lm_eval' is installed: pip install lm-eval"
echo "2. Ensure the model is loaded in LM Studio"
echo ""

# Check if lm_eval is installed
if ! command -v lm_eval &> /dev/null; then
    echo "lm_eval not found. Installing..."
    pip install lm-eval
fi

MODEL_NAME="lm-studio" # Local execution

# Output directory
RESULTS_DIR="benchmarks/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Running TruthfulQA (mc)..."
lm_eval --model local-chat-completions \
    --model_args model=this_is_ignored,base_url=http://localhost:1234/v1 \
    --tasks truthfulqa_mc \
    --device mps \
    --batch_size 1 \
    --output_path "$RESULTS_DIR/truthfulqa.json"

echo "Running CensorBench (custom adapter or separate tool needed - placeholder)..."
# Note: CensorBench isn't a standard harness task yet.
# We might need a custom script or a different harness integration.
# For now, we rely on our 'custom.rs' implementation for CensorBench-like tests.

echo "Running MMLU (Global Facts)..."
lm_eval --model local-chat-completions \
    --model_args model=this_is_ignored,base_url=http://localhost:1234/v1 \
    --tasks mmlu_global_facts \
    --device mps \
    --batch_size 1 \
    --output_path "$RESULTS_DIR/mmlu.json"

echo "Benchmarks complete. Results saved to $RESULTS_DIR"
