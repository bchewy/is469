#!/bin/bash
# Setup and train on a remote GPU box (PrimeIntellect, etc.)
#
# Usage: After SSH-ing into your GPU instance:
#   1. Upload this repo:  scp -r . root@<POD_IP>:/workspace/genai-llms/
#   2. Run:               cd /workspace/genai-llms && bash scripts/setup_and_train.sh
#
# Requires: NVIDIA GPU, CUDA drivers pre-installed (standard on PI pods)

set -euo pipefail

echo "=== Installing dependencies ==="
pip install -q \
    "torch==2.5.1" \
    "transformers>=4.46.0" \
    "peft>=0.14.0" \
    "datasets>=3.2.0" \
    "accelerate>=1.2.0" \
    "trl>=0.15.0" \
    "bitsandbytes>=0.45.0" \
    "pyyaml>=6.0.2" \
    "sentencepiece>=0.2.0" \
    "protobuf>=5.29.0"

echo "=== GPU check ==="
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

echo "=== Starting training ==="
python3 scripts/train_local.py \
    --config configs/finetune_translation.yaml \
    --output-dir outputs/adapter

echo "=== Done! Adapter saved to outputs/adapter/final/ ==="
echo "To download: scp -r root@<POD_IP>:/workspace/genai-llms/outputs/adapter/final/ ./models/adapter-translation/final/"
