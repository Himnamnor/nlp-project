#!/usr/bin/env bash
# run_rlhf.sh — 完整 RLHF 三步：reward 训练 → PPO → safety 评测
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

python -m src.train.reward --config configs/reward.yaml
python -m src.train.ppo    --config configs/ppo.yaml
# baseline (SFT) + RLHF (PPO) 各跑一次安全评测，便于对照
python -m src.eval.safety_eval --config configs/ppo.yaml \
    --ckpt checkpoints/sft_smol_full/best.pt --label sft_baseline \
    --output logs/ppo/safety_eval_sft.jsonl
python -m src.eval.safety_eval --config configs/ppo.yaml \
    --ckpt checkpoints/ppo/best.pt --label ppo \
    --output logs/ppo/safety_eval_ppo.jsonl
