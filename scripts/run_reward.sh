#!/usr/bin/env bash
# run_reward.sh — 奖励模型训练 (Linux / AutoDL)
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
python -m src.train.reward --config "${1:-configs/reward.yaml}" "${@:2}"
