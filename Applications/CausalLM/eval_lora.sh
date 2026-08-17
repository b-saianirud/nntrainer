#!/bin/bash
#
# Evaluate trained LoRA adapter on lamp3_user_test.txt
#
# This script:
#   1. Loads the model with the trained LoRA adapter
#   2. Runs inference on each test sample (prompt only, no answer)
#   3. Compares the model's predicted answer to the expected answer
#   4. Reports accuracy and per-sample results
#
# Usage:
#   ./eval_lora.sh [OPTIONS]
#
# Options:
#   --model_dir <path>    Model directory (default: res/qwen3/qwen3-0.6b)
#   --test_data <path>    Test data file (default: res/train_data/lamp3_user_test.txt)
#   --lora <path>         LoRA adapter file (default: <model_dir>/lora_adapter.bin)
#   --num_generate <int>  Number of tokens to generate per sample (default: 5)
#   --output <path>       Output results file (default: /tmp/eval_results.txt)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
MODEL_DIR="$SCRIPT_DIR/res/qwen3/qwen3-0.6b"
TEST_DATA="$SCRIPT_DIR/res/train_data/lamp3_user_test.txt"
LORA_PATH=""
NUM_GENERATE=20
OUTPUT_FILE="/tmp/eval_results.txt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_dir) MODEL_DIR="$2"; shift 2 ;;
        --test_data) TEST_DATA="$2"; shift 2 ;;
        --lora) LORA_PATH="$2"; shift 2 ;;
        --num_generate) NUM_GENERATE="$2"; shift 2 ;;
        --output) OUTPUT_FILE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$LORA_PATH" ]; then
    LORA_PATH="$MODEL_DIR/lora_adapter.bin"
fi

# Check files exist
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi
if [ ! -f "$TEST_DATA" ]; then
    echo "ERROR: Test data not found: $TEST_DATA"
    exit 1
fi
if [ ! -f "$LORA_PATH" ]; then
    echo "ERROR: LoRA adapter not found: $LORA_PATH"
    exit 1
fi

# Copy LoRA adapter to model dir (nntr_causallm loads it from there)
if [ "$LORA_PATH" != "$MODEL_DIR/lora_adapter.bin" ]; then
    echo "Copying LoRA adapter to model directory..."
    cp "$LORA_PATH" "$MODEL_DIR/lora_adapter.bin"
fi

echo "============================================"
echo "  LoRA Evaluation on Test Data"
echo "============================================"
echo "Model:      $MODEL_DIR"
echo "Test data:  $TEST_DATA"
echo "LoRA:       $LORA_PATH"
echo "Generate:   $NUM_GENERATE tokens"
echo "Output:     $OUTPUT_FILE"
echo "============================================"
echo ""

# Use Python to parse test data, run inference, and evaluate
python3 "$SCRIPT_DIR/eval_lora.py" \
    --model_dir "$MODEL_DIR" \
    --test_data "$TEST_DATA" \
    --num_generate "$NUM_GENERATE" \
    --output "$OUTPUT_FILE" \
    --nntrainer_root "$NNTRAINER_ROOT"

echo ""
echo "Results saved to: $OUTPUT_FILE"
