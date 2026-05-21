#!/usr/bin/env bash
# run_rlhf.sh — Reward model + PPO
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

python -m src.train.reward --config configs/reward.yaml
python -m src.train.ppo --config configs/ppo.yaml
python -m src.eval.safety_eval --config configs/ppo.yaml
