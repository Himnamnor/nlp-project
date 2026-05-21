"""
dpo.py — DPO 对照训练（进阶任务主路径，也可作 PPO 备选）

用途：
  - 无需奖励模型，直接在偏好对上优化 SFT 模型
  - trl.DPOTrainer + LoRA

运行：
  python -m src.train.dpo --config configs/dpo.yaml

也可从 advanced/dpo/ 调用本模块。
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
    print(f"DPO: {len(pairs)} preference pairs")

    # TODO:
    # from trl import DPOTrainer, DPOConfig
    # trainer = DPOTrainer(model=..., ref_model=None, train_dataset=..., beta=cfg['dpo']['beta'])

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print("DPO skeleton ready.")
    logger.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO alignment (advanced baseline)")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = parse_config_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
