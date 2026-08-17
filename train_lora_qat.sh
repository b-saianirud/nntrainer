#!/bin/bash
# QAT LoRA Training: 90 samples, 20 epochs
# Use AFTER validating FP32 training works properly

set -e

cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$(pwd)/subprojects/OpenBLAS/build/lib:$LD_LIBRARY_PATH"

echo "================================"
echo "QAT LoRA Fine-tuning"
echo "================================"
echo "Dataset: LAMP-3 (90 samples)"
echo "Epochs: 20"
echo "Base Model: Q4_0 quantized"
echo "LoRA: FP32 (with fake-quantization)"
echo "Training steps: 900 (~20-30 minutes)"
echo ""

./build/Applications/CausalLM/nntr_lora_train \
  Applications/CausalLM/res/qwen3/qwen3-0.6b/ \
  Applications/CausalLM/res/train_data/lamp3_user_train.txt \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_qat \
  --lora_weight_q4 \
  --seq_len 192 \
  --lr 1e-4 \
  --epochs 20 \
  --max_samples 90 \
  --clip_grad 1.0 \
  --seed 42 \
  --output /tmp/qwen3_lora_qat_fp32.bin

echo ""
echo "================================"
echo "QAT Training complete!"
echo "FP32 adapter: /tmp/qwen3_lora_qat_fp32.bin"
echo "Q4_0 adapter: /tmp/qwen3_lora_qat_q4.bin"
echo ""
echo "To test with Q4_0 base + Q4_0 LoRA:"
echo "  cp /tmp/qwen3_lora_qat_q4.bin Applications/CausalLM/res/qwen3/qwen3-0.6b/lora_qat_q4.bin"
echo "  # Update nntr_config.json:"
echo "  #   fc_layer_dtype: Q4_0"
echo "  #   model_tensor_type: Q4_0-FP32"
echo "  #   model_file_name: nntr_qwen3_0.6b_q40_embdfp32_DEFAULT.bin"
echo "  #   lora_q4_file_name: lora_qat_q4.bin"
echo "  ./build/Applications/CausalLM/nntr_causallm Applications/CausalLM/res/qwen3/qwen3-0.6b/"
echo ""
