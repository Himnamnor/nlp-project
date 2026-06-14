#!/usr/bin/env bash
# run_general_pretrain.sh — 泛化预训练全流程（tokenizer → data → train）
#
# 用法：
#   bash scripts/run_general_pretrain.sh
#   bash scripts/run_general_pretrain.sh configs/pretrain_general_fineweb.yaml --max_steps 20
#
# 可选环境变量：
#   RUN_TOKENIZER=0  跳过 tokenizer 训练
#   RUN_DATA=0       跳过 .bin 数据准备
#   RUN_TRAIN=0      只准备 tokenizer/data，不训练
#   BPE_MAX_SAMPLES=50000
#   DATA_MAX_TOKENS=2000000

set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

CONFIG=${1:-configs/pretrain_general.yaml}
if [[ $# -gt 0 ]]; then
  shift
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

if [[ "${RUN_TOKENIZER:-1}" == "1" ]]; then
  BPE_ARGS=(--config "$CONFIG")
  if [[ -n "${BPE_MAX_SAMPLES:-}" ]]; then
    BPE_ARGS+=(--max_samples "$BPE_MAX_SAMPLES")
  fi
  python -m src.tokenizer.train_bpe "${BPE_ARGS[@]}"
fi

if [[ "${RUN_DATA:-1}" == "1" ]]; then
  DATA_ARGS=(--config "$CONFIG" --prepare)
  if [[ -n "${DATA_MAX_TOKENS:-}" ]]; then
    DATA_ARGS+=(--max_tokens "$DATA_MAX_TOKENS")
  fi
  python -m src.data.pretrain "${DATA_ARGS[@]}"
fi

if [[ "${RUN_TRAIN:-1}" == "1" ]]; then
  python -m src.train.pretrain --config "$CONFIG" "$@"
fi
