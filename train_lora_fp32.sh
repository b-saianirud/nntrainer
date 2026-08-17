#!/bin/bash
# FP32 LoRA Training: 90 samples, 20 epochs
# Use this for initial validation before QAT

set -e

cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$(pwd)/subprojects/OpenBLAS/build/lib:$LD_LIBRARY_PATH"

echo "================================"
echo "FP32 LoRA Fine-tuning"
echo "================================"
echo "Dataset: LAMP-3 (90 samples)"
echo "Epochs: 20"
echo "Batch size: 2"
echo "Training steps: 900 (~15-20 minutes)"
echo ""

./build/Applications/CausalLM/nntr_lora_train \
  Applications/CausalLM/res/qwen3/qwen3-0.6b/ \
  Applications/CausalLM/res/train_data/lamp3_user_train.txt \
  --lora_rank 32 \
  --lora_alpha 64 \
  --seq_len 192 \
  --lr 1e-4 \
  --epochs 20 \
  --max_samples 90 \
  --clip_grad 1.0 \
  --seed 42 \
  --output /tmp/qwen3_lora_fp32.bin

echo ""
echo "================================"
echo "Training complete!"
echo "Adapter saved to: /tmp/qwen3_lora_fp32.bin"
echo ""
echo "To test inference with trained adapter:"
echo "  cp /tmp/qwen3_lora_fp32.bin Applications/CausalLM/res/qwen3/qwen3-0.6b/lora_fp32.bin"
echo "  # Update nntr_config.json to use lora_file_name: 'lora_fp32.bin'"
echo "  ./build/Applications/CausalLM/nntr_causallm Applications/CausalLM/res/qwen3/qwen3-0.6b/"
echo ""
