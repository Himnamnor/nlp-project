#!/usr/bin/env bash
# run_ppo_conservative.sh — 保守 PPO 消融 (Linux / AutoDL)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
python -m src.train.ppo --config "${1:-configs/ppo_conservative.yaml}" "${@:2}"
