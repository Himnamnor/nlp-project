#!/usr/bin/env bash
# run_advanced.sh — 跑完一整套 advanced 实验：DPO + 保守 PPO + 两次安全评测
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 1. DPO 训练
python -m src.train.dpo --config configs/dpo.yaml

# 2. 保守 PPO 消融
python -m src.train.ppo --config configs/ppo_conservative.yaml

# 3. 三方安全评测（与 SFT baseline / 原 PPO 对照）
python -m src.eval.safety_eval --config configs/dpo.yaml \
    --ckpt checkpoints/dpo/best.pt --label dpo \
    --output logs/dpo/safety_eval_dpo.jsonl
python -m src.eval.safety_eval --config configs/ppo_conservative.yaml \
    --ckpt checkpoints/ppo_conservative/best.pt --label ppo_conservative \
    --output logs/ppo_conservative/safety_eval_ppo_conservative.jsonl
