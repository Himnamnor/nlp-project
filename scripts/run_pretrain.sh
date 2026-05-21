#!/usr/bin/env bash
# run_pretrain.sh — 预训练全流程（tokenizer → data → train）
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

CONFIG=${1:-configs/pretrain.yaml}

python -m src.tokenizer.train_bpe --config "$CONFIG"
python -m src.data.pretrain --config "$CONFIG" --prepare
python -m src.train.pretrain --config "$CONFIG" "$@"
