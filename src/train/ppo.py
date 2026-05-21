"""
ppo.py — PPO 强化学习（TRL PPOTrainer + LoRA）

用途：
  - policy: SFT + LoRA + value head
  - ref: 同 base，disable_adapter 推理
  - reward: 独立加载 reward model
  - 3060 配置: batch=8, mini_batch=2, max_new_tokens=96

运行：
  python -m src.train.ppo --config configs/ppo.yaml

TODO：
  - PPOConfig + PPOTrainer from trl
  - generation kwargs: temperature, top_p
  - log reward_mean, kl, policy_loss, value_loss
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.rlhf import sample_prompts_for_ppo
from src.utils.config import add_config_args, parse_config_from_args
from src.utils.logging import init_logger


def train(cfg: dict) -> None:
    logger = init_logger(cfg, run_name=cfg["project"]["name"])
    n = cfg["data"].get("num_prompts", 500)
    prompts = sample_prompts_for_ppo(cfg, n)
    print(f"Sampled {len(prompts)} prompts for PPO")

    # TODO:
    # from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    # policy = load sft + lora + value head
    # ref = policy with adapters disabled
    # reward_model = load from checkpoints/reward/
    # ppo_trainer = PPOTrainer(config, model=policy, ref_model=ref, ...)

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print("PPO skeleton ready. Wire TRL PPOTrainer + PEFT.")
    logger.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO alignment with LoRA")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = parse_config_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
