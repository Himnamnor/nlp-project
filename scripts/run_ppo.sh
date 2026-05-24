#!/usr/bin/env bash
# run_ppo.sh — PPO RLHF (Linux / AutoDL)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
python -m src.train.ppo --config "${1:-configs/ppo.yaml}" "${@:2}"
