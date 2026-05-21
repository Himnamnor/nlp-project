"""
rlhf.py — PKU-SafeRLHF 偏好对数据

用途：
  - 按 safer_response_id 构造 (prompt, chosen, rejected)
  - 供 RewardTrainer / PPOTrainer / DPOTrainer 使用
  - 安全评测 prompt 采样

TODO：
  - load_preference_dataset(cfg) -> Dataset
  - format for TRL: {"prompt", "chosen", "rejected"} 或 {"query", "response", ...}
  - sample_unsafe_prompts(n) for PPO / safety_eval
"""

from __future__ import annotations

from typing import Any

from datasets import load_dataset


def load_safe_rlhf(dataset_name: str = "PKU-Alignment/PKU-SafeRLHF", split: str = "train"):
    """Load PKU-SafeRLHF dataset."""
    return load_dataset(dataset_name, split=split)


def build_preference_pair(example: dict) -> dict[str, str]:
    """
    Convert one SafeRLHF row to (prompt, chosen, rejected).

    safer_response_id: 0 or 1 indicating which response is safer.
    """
    prompt = example.get("prompt", "")
    r0 = example["response_0"]
    r1 = example["response_1"]
    safer = example.get("safer_response_id", 0)
    if safer == 0:
        chosen, rejected = r0, r1
    else:
        chosen, rejected = r1, r0
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def build_preference_dataset(cfg: dict) -> list[dict[str, str]]:
    """Load and map all examples to preference pairs."""
    ds = load_safe_rlhf(cfg["data"]["dataset"])
    return [build_preference_pair(ex) for ex in ds]


def sample_prompts_for_ppo(cfg: dict, n: int) -> list[str]:
    """Sample prompts for PPO training (mix unsafe + normal)."""
    pairs = build_preference_dataset(cfg)
    unsafe_ratio = cfg.get("data", {}).get("unsafe_ratio", 0.7)
    # TODO: filter by is_response_0_safe / is_response_1_safe flags
    prompts = [p["prompt"] for p in pairs[:n]]
    return prompts
