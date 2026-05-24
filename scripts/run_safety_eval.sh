#!/usr/bin/env bash
# run_safety_eval.sh — RLHF 安全评测 (Linux / AutoDL)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
python -m src.eval.safety_eval --config "${1:-configs/ppo.yaml}" "${@:2}"
