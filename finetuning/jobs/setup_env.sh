#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# One-time environment setup on KISSKI / Grete HPC cluster.
#
# Run interactively (NOT as a SLURM job):
#   bash finetuning/jobs/setup_env.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ENV_NAME="orkg-ft"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"

echo "=== Installing PyTorch with CUDA ==="
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo "=== Installing training dependencies ==="
pip install \
    transformers>=4.44.0 \
    accelerate>=0.30.0 \
    peft>=0.11.0 \
    trl>=0.9.0 \
    bitsandbytes>=0.43.0 \
    datasets>=2.20.0 \
    sentencepiece>=0.1.99 \
    protobuf>=3.20.0

echo "=== Installing pipeline dependencies ==="
pip install \
    pdfplumber>=0.10.0 \
    pypdf2>=3.0.0 \
    pydantic>=2.0.0 \
    pyyaml>=6.0 \
    requests>=2.31.0 \
    tqdm>=4.66.0 \
    python-dotenv>=1.0.0

echo "=== Environment ready ==="
echo "Activate with:  conda activate $ENV_NAME"
