#!/bin/bash
###############################################################################
# nntrainer CausalLM LoRA Training — Full Setup Script
# Reproduces the exact environment used on the development machine.
#
# Target: Ubuntu 22.04 x86_64
# Branch: LoRA (fork: git@github.com:b-saianirud/nntrainer.git)
#
# Usage:
#   chmod +x setup_new_server.sh
#   ./setup_new_server.sh
#
# After setup completes, run training with:
#   ./train_lora_fp32.sh    # FP32 LoRA training
#   ./train_lora_qat.sh     # QAT (Q4_0) LoRA training
###############################################################################
set -e

echo "============================================"
echo "  nntrainer CausalLM Setup"
echo "============================================"

# ── 1. Install system dependencies ──────────────────────────────────────────
echo ""
echo "[1/7] Installing system packages..."

sudo apt-get update -y
sudo apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    python3-dev \
    git \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-gl1.0-dev \
    libjson-glib-dev \
    libxml2-dev \
    libsqlite3-dev \
    libiniparser-dev

# ── 2. Install meson (via pip, version 1.11.2) ─────────────────────────────
echo ""
echo "[2/7] Installing meson and ninja via pip..."

pip3 install --user meson==1.11.2
# Ensure pip user bin is on PATH
export PATH="$HOME/.local/bin:$PATH"
# Make it persistent
if ! grep -q '.local/bin' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi

# ── 3. Clone the repository and checkout the LoRA branch ────────────────────
echo ""
echo "[3/7] Cloning repository..."

NNTRAINER_DIR="$(cd "$(dirname "$0")" && pwd)"

# If this script is run from inside an existing clone, skip cloning
if [ ! -d "$NNTRAINER_DIR/.git" ]; then
    git clone --recursive git@github.com:b-saianirud/nntrainer.git "$NNTRAINER_DIR"
    cd "$NNTRAINER_DIR"
    git checkout LoRA
else
    cd "$NNTRAINER_DIR"
    echo "  Already in a git repo, using existing clone."
    git checkout LoRA 2>/dev/null || true
fi

# ── 4. Initialize git submodules ─────────────────────────────────────────────
echo ""
echo "[4/7] Initializing git submodules..."

git submodule sync
git submodule update --init --depth 1

# ── 5. Build OpenBLAS from source (subprojects/OpenBLAS) ────────────────────
echo ""
echo "[5/7] Building OpenBLAS..."

cd "$NNTRAINER_DIR/subprojects/OpenBLAS"

mkdir -p build
cd build

# Build OpenBLAS as a shared library with cmake
# Key options matching the dev environment:
#   - BUILD_SHARED_LIBS=ON  (produces libopenblas.so)
#   - DYNAMIC_ARCH=OFF      (native arch)
#   - NO_AFFINITY=ON
cmake .. \
    -DBUILD_SHARED_LIBS=ON \
    -DDYNAMIC_ARCH=OFF \
    -DNO_AFFINITY=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build . -j"$(nproc)"

echo "  OpenBLAS built at: $NNTRAINER_DIR/subprojects/OpenBLAS/build/lib/"
ls -la lib/libopenblas.so* 2>/dev/null

cd "$NNTRAINER_DIR"

# ── 6. Configure and build nntrainer with meson ─────────────────────────────
echo ""
echo "[6/7] Configuring and building nntrainer..."

# Meson configure options (matching the dev environment exactly):
#   -Dthread-backend=omp
#   -Denable-transformer=true
#   -Denable-tflite-interpreter=false
#   -Denable-tflite-backbone=false
meson setup build \
    -Dthread-backend=omp \
    -Denable-transformer=true \
    -Denable-tflite-interpreter=false \
    -Denable-tflite-backbone=false

ninja -C build

echo ""
echo "  Build complete! Key binaries:"
echo "    build/Applications/CausalLM/nntr_lora_train   (LoRA training)"
echo "    build/Applications/CausalLM/nntr_causallm     (inference)"
echo "    build/Applications/CausalLM/nntr_quantize     (model quantization)"

# ── 7. Verify the build ─────────────────────────────────────────────────────
echo ""
echo "[7/7] Verifying build..."

export LD_LIBRARY_PATH="$NNTRAINER_DIR/subprojects/OpenBLAS/build/lib:$LD_LIBRARY_PATH"

# Quick smoke test: run the gradient check test
echo "  Running MHA gradient check test..."
if ./build/Applications/CausalLM/unittest_lora_backward_gradcheck \
       --gtest_filter='*MhaCoreCalcDerivative*' 2>&1 | grep -q "PASSED"; then
    echo "  ✓ Gradient check test PASSED"
else
    echo "  ⚠ Gradient check test did not pass (may need OPENBLAS_NUM_THREADS=1)"
fi

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Model files needed in:"
echo "  Applications/CausalLM/res/qwen3/qwen3-0.6b/"
echo "    - nntr_qwen3_0.6b_fp32.bin              (3.0 GB, FP32 model)"
echo "    - nntr_qwen3_0.6b_q40_embdfp32_DEFAULT.bin  (870 MB, Q4_0 model)"
echo "    - nntr_config.json                       (model config)"
echo ""
echo "Training data in:"
echo "  Applications/CausalLM/res/train_data/"
echo "    - lamp3_user_train.txt"
echo "    - lamp3_user_test.txt"
echo ""
echo "To run FP32 LoRA training:"
echo "  export LD_LIBRARY_PATH=\"$(pwd)/subprojects/OpenBLAS/build/lib:\$LD_LIBRARY_PATH\""
echo "  ./train_lora_fp32.sh"
echo ""
echo "To run QAT (Q4_0) LoRA training:"
echo "  export LD_LIBRARY_PATH=\"$(pwd)/subprojects/OpenBLAS/build/lib:\$LD_LIBRARY_PATH\""
echo "  ./train_lora_qat.sh"
echo ""
echo "NOTE: If you see 'BLAS: Bad memory unallocation' errors, run with:"
echo "  OPENBLAS_NUM_THREADS=1 ./train_lora_fp32.sh"
echo ""
