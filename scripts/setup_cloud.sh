#!/usr/bin/env bash
# setup_cloud.sh — AutoDL / Linux 云端一键环境准备
# 用法: bash scripts/setup_cloud.sh
# 前提: 代码已 git clone 到 /root/autodl-tmp/Project

set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

python -m venv .venv
source .venv/bin/activate

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "Done. Next steps:"
echo "  tmux new -s train"
echo "  source .venv/bin/activate"
echo "  python -m src.tokenizer.train_bpe --config configs/pretrain.yaml"
echo "  python -m src.data.pretrain --config configs/pretrain.yaml --prepare"
echo "  python -m src.train.pretrain --config configs/pretrain.yaml"
