#!/usr/bin/env bash
# pull_ckpt.sh — Pull selected artifacts from a cloud machine with rsync resume.
#
# Example:
#   REMOTE_HOST=connect.autodl.com REMOTE_PORT=12345 \
#   REMOTE_ROOT=/root/autodl-tmp/Project \
#   bash scripts/pull_ckpt.sh
#
# Override REMOTE_ITEMS to choose what to pull.

set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST:-connect.westb.seetacloud.com}"
REMOTE_PORT="${REMOTE_PORT:-53583}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/Project}"
LOCAL_ROOT="${LOCAL_ROOT:-$(pwd)}"

REMOTE_ITEMS="${REMOTE_ITEMS:-\
checkpoints/pretrain_general/best.slim.pt \
checkpoints/pretrain_general_continued/best.slim.pt \
checkpoints/sft_general_full/best.slim.pt \
checkpoints/reward_general/best.slim.pt \
checkpoints/ppo_general/best.slim.pt \
tokenizer_general \
configs/pretrain_general.yaml \
configs/pretrain_general_continued.yaml \
configs/sft_general.yaml \
configs/reward_general.yaml \
configs/ppo_general.yaml \
logs/pretrain_general \
logs/pretrain_general_continued \
logs/sft_general_full \
logs/ppo_general}"

mkdir -p "$LOCAL_ROOT"

for item in $REMOTE_ITEMS; do
  echo "Pulling $item"
  mkdir -p "$LOCAL_ROOT/$(dirname "$item")"
  rsync -avP \
    -e "ssh -p ${REMOTE_PORT}" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/${item}" \
    "${LOCAL_ROOT}/${item}"
done
