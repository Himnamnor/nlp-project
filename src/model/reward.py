"""
reward.py — 奖励模型（Llama backbone + scalar head）

用途：
  - 在 SFT 模型最后一层 hidden state 上接 Linear(d_model, 1)
  - 输入 (prompt, response) 序列，取最后一个非 pad token 的 hidden → reward 标量
  - 与 TRL RewardTrainer 或手写 pairwise loss 配合

训练目标：-log σ(r(chosen) - r(rejected))

TODO：
  - class RewardModel(nn.Module): backbone + score_head
  - forward(input_ids, attention_mask) -> rewards [B]
  - get_last_token_hidden(...) helper
  - load_from_sft_ckpt(path, config)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .llama import LlamaConfig, LlamaModel


class RewardModel(nn.Module):
    """Scalar reward model on top of Llama backbone."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = LlamaModel(config)
        self.score_head = nn.Linear(config.d_model, 1, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns reward scores of shape [batch_size].

        TODO: backbone → last non-pad hidden → score_head
        """
        raise NotImplementedError("RewardModel.forward")

    @staticmethod
    def pairwise_loss(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
        """Bradley-Terry loss: -log sigmoid(r_c - r_r)."""
        return -torch.nn.functional.logsigmoid(reward_chosen - reward_rejected).mean()


def load_reward_from_sft(backbone_state: dict, config: LlamaConfig) -> RewardModel:
    """Initialize reward model backbone from SFT/pretrain weights."""
    model = RewardModel(config)
    # TODO: model.backbone.load_state_dict(backbone_state, strict=False)
    return model
