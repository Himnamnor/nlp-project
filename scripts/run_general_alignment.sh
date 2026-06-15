#!/usr/bin/env bash
# run_general_alignment.sh — 泛化预训练后的 SFT → Reward → PPO → Safety eval

set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

SFT_CONFIG=${SFT_CONFIG:-configs/sft_general.yaml}
REWARD_CONFIG=${REWARD_CONFIG:-configs/reward_general.yaml}
PPO_CONFIG=${PPO_CONFIG:-configs/ppo_general.yaml}

if [[ "${RUN_SFT:-1}" == "1" ]]; then
  python -m src.train.sft --config "$SFT_CONFIG"
  python -m src.eval.sft_eval \
    --config "$SFT_CONFIG" \
    --ckpt checkpoints/sft_general_full/best.pt \
    --output logs/sft_general_full/eval_samples.jsonl
fi

if [[ "${RUN_REWARD:-1}" == "1" ]]; then
  python -m src.train.reward --config "$REWARD_CONFIG"
fi

if [[ "${RUN_PPO:-1}" == "1" ]]; then
  python -m src.train.ppo --config "$PPO_CONFIG"
fi

if [[ "${RUN_SAFETY_EVAL:-1}" == "1" ]]; then
  python -m src.eval.safety_eval --config "$PPO_CONFIG" \
    --ckpt checkpoints/sft_general_full/best.pt --label sft_general \
    --output logs/ppo_general/safety_eval_sft_general.jsonl
  python -m src.eval.safety_eval --config "$PPO_CONFIG" \
    --ckpt checkpoints/ppo_general/best.pt --label ppo_general \
    --output logs/ppo_general/safety_eval_ppo_general.jsonl
fi

