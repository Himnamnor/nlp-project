#!/usr/bin/env bash
# run_dpo.sh — DPO 训练 (Linux / AutoDL)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
python -m src.train.dpo --config "${1:-configs/dpo.yaml}" "${@:2}"
