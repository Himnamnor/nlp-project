#!/usr/bin/env bash
# run_sft.sh — SFT 训练
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m src.train.sft --config "${1:-configs/sft.yaml}"
