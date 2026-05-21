"""
reward.py — 奖励模型训练（TRL RewardTrainer + LoRA）

用途：
  - 从 SFT ckpt 加载 LlamaMiniForCausalLM + LoRA + value head
  - PKU-SafeRLHF 偏好对，pairwise loss
  - 报告 validation pairwise accuracy

运行：
  python -m src.train.reward --config configs/reward.yaml

TODO：
  - wrap model with PEFT LoRA
  - trl.RewardTrainer(..., processing_class=tokenizer)
  - save adapter + score_head to checkpoints/reward/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.rlhf import build_preference_dataset
from src.utils.config import add_config_args, parse_config_from_args
from src.utils.logging import init_logger


def train(cfg: dict) -> None:
    logger = init_logger(cfg, run_name=cfg["project"]["name"])
    pairs = build_preference_dataset(cfg)
    print(f"Loaded {len(pairs)} preference pairs from {cfg['data']['dataset']}")

    # TODO:
    # from peft import LoraConfig, get_peft_model
    # from trl import RewardTrainer, RewardConfig
    # model = LlamaMiniForCausalLM.from_pretrained(...)
    # peft_model = get_peft_model(model, lora_config)
    # trainer = RewardTrainer(model=peft_model, train_dataset=..., ...)

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print("Reward training skeleton ready. Wire TRL RewardTrainer.")
    logger.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reward model with LoRA")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = parse_config_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
