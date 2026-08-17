#!/bin/bash
# Test inference with trained LoRA adapter

set -e

cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$(pwd)/subprojects/OpenBLAS/build/lib:$LD_LIBRARY_PATH"

MODEL_DIR="Applications/CausalLM/res/qwen3/qwen3-0.6b"
TRAINED_ADAPTER="/tmp/qwen3_lora_fp32.bin"

if [ ! -f "$TRAINED_ADAPTER" ]; then
    echo "❌ Trained adapter not found at $TRAINED_ADAPTER"
    echo "Run ./train_lora_fp32.sh first"
    exit 1
fi

echo "================================"
echo "Testing trained LoRA adapter"
echo "================================"
echo "Adapter: $TRAINED_ADAPTER"
echo ""

# Copy trained adapter to model directory
cp "$TRAINED_ADAPTER" "$MODEL_DIR/lora_trained.bin"

# Create inference config using trained adapter
cat > "$MODEL_DIR/nntr_config_trained.json" <<'EOF'
{
    "batch_size": 1,
    "embedding_dtype": "FP32",
    "fc_layer_dtype": "FP32",
    "lora_alpha": 64,
    "lora_rank": 32,
    "lora_target": ["wq", "wk", "wv", "wo", "ffn_up", "ffn_gate", "ffn_down"],
    "lora_file_name": "lora_trained.bin",
    "max_seq_len": 2048,
    "model_file_name": "nntr_qwen3_0.6b_fp32.bin",
    "model_tensor_type": "FP32-FP32",
    "model_type": "CausalLM",
    "num_to_generate": 128,
    "sample_input": "<|im_start|>user\nExplain machine learning in simple terms.<|im_end|>\n<|im_start|>assistant\n",
    "tokenizer_file": "tokenizer.json"
}
EOF

echo "Running inference with trained adapter..."
./build/Applications/CausalLM/nntr_causallm "$MODEL_DIR" nntr_config_trained.json

echo ""
echo "✅ Test complete!"
