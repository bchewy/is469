#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
. .venv/bin/activate
mkdir -p results/logs
while [ ! -f models/adapter-translation-qwen25-0p5b-runA/train_results.json ]; do
  sleep 20
done
PYTHONUNBUFFERED=1 python3 -u scripts/run_variant_local.py --variant s1 --config configs/s1_dev_eval_runA.yaml 2>&1 | tee results/logs/s1_runA_dev_eval.log
