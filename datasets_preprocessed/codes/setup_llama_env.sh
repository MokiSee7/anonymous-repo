#!/usr/bin/env bash
# =============================================================================
# Setup script for the "llama" conda environment
# Used for Step 3: Generate Shot-Aligned Documents via LLaMA
#
# Prerequisites:
#   - Conda/Miniconda installed
#   - CUDA 12.8 compatible GPU and driver (e.g., RTX Blackwell)
#
# Usage:
#   bash setup_llama_env.sh
# =============================================================================

set -e

ENV_NAME="llama"
PYTHON_VERSION="3.10"

# PyTorch nightly with CUDA 12.8 (same as llavav env)
TORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128"
TORCH_VERSION="torch==2.11.0.dev20260207+cu128"
TORCHVISION_VERSION="torchvision==0.26.0.dev20260207+cu128"
TORCHAUDIO_VERSION="torchaudio==2.11.0.dev20260207+cu128"

echo "============================================================"
echo "  Creating conda environment: ${ENV_NAME}"
echo "  Python: ${PYTHON_VERSION}  |  CUDA: 12.8"
echo "============================================================"

# Create environment
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# Install PyTorch nightly with CUDA 12.8
echo ""
echo "[1/3] Installing PyTorch (CUDA 12.8 nightly)..."
conda run -n "${ENV_NAME}" pip install \
    "${TORCH_VERSION}" \
    "${TORCHVISION_VERSION}" \
    "${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX}"

# Install HuggingFace stack
echo ""
echo "[2/3] Installing transformers, accelerate, and utilities..."
conda run -n "${ENV_NAME}" pip install \
    "transformers>=4.45.0" \
    "accelerate>=1.0.0" \
    "sentencepiece" \
    "protobuf" \
    "tqdm" \
    "numpy"

# Install bitsandbytes for optional 8-bit / 4-bit quantization
echo ""
echo "[3/3] Installing bitsandbytes (optional quantization support)..."
conda run -n "${ENV_NAME}" pip install bitsandbytes || echo "  [WARN] bitsandbytes install failed — 8/4-bit quant unavailable"

echo ""
echo "============================================================"
echo "  Environment '${ENV_NAME}' created successfully!"
echo ""
echo "  Activation:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "  Quick smoke test:"
echo "    python -c \"import torch; print('CUDA:', torch.cuda.is_available())\""
echo "============================================================"
