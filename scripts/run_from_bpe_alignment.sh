#!/usr/bin/env bash
# run_from_bpe_alignment.sh — Reward/DPO/PPO and safety eval for from-BPE run.

set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

REWARD_CONFIG=${REWARD_CONFIG:-configs/reward_general_from_bpe.yaml}
DPO_CONFIG=${DPO_CONFIG:-configs/dpo_general_from_bpe.yaml}
PPO_CONFIG=${PPO_CONFIG:-configs/ppo_general_from_bpe.yaml}
SFT_CONFIG=${SFT_CONFIG:-configs/sft_general_from_bpe.yaml}

if [[ "${RUN_REWARD:-1}" == "1" ]]; then
  python -m src.train.reward --config "$REWARD_CONFIG"
fi

if [[ "${RUN_DPO:-1}" == "1" ]]; then
  python -m src.train.dpo --config "$DPO_CONFIG"
fi

if [[ "${RUN_PPO:-1}" == "1" ]]; then
  python -m src.train.ppo --config "$PPO_CONFIG"
fi

if [[ "${RUN_SAFETY_EVAL:-1}" == "1" ]]; then
  mkdir -p /root/autodl-tmp/Project/logs/alignment_from_bpe
  python -m src.eval.safety_eval --config "$PPO_CONFIG" \
    --ckpt /root/autodl-tmp/checkpoints/sft_general_from_bpe/best.pt \
    --label sft_from_bpe \
    --output /root/autodl-tmp/Project/logs/alignment_from_bpe/safety_eval_sft_from_bpe.jsonl
  python -m src.eval.safety_eval --config "$DPO_CONFIG" \
    --ckpt /root/autodl-tmp/checkpoints/dpo_general_from_bpe/best.pt \
    --label dpo_from_bpe \
    --output /root/autodl-tmp/Project/logs/alignment_from_bpe/safety_eval_dpo_from_bpe.jsonl
  python -m src.eval.safety_eval --config "$PPO_CONFIG" \
    --ckpt /root/autodl-tmp/checkpoints/ppo_general_from_bpe/best.pt \
    --label ppo_from_bpe \
    --output /root/autodl-tmp/Project/logs/alignment_from_bpe/safety_eval_ppo_from_bpe.jsonl
fi
