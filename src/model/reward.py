"""
reward.py — 奖励模型（Llama backbone + scalar head）

设计：
  - backbone 复用 LlamaModel；最后一个非 pad token 的 hidden 经 Linear(d, 1) 得到 scalar reward
  - 训练目标（Bradley-Terry）：L = -log σ(r(chosen) - r(rejected))
  - 评估：pairwise accuracy = P(r_chosen > r_rejected)

接口：
  - RewardModel(config).forward(input_ids, attention_mask) -> [B]
  - pairwise_bt_loss / pairwise_accuracy 工具函数
  - load_backbone_from_causal_lm 从 SFT/pretrain 的 LlamaForCausalLM 权重初始化 backbone
  - build_reward_model_from_config 与 build_llama_from_config 对齐的工厂
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .llama import LlamaConfig, LlamaModel


class RewardModel(nn.Module):
    """Bradley-Terry scalar reward model on top of a frozen-able Llama backbone."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = LlamaModel(config)
        self.score_head = nn.Linear(config.d_model, 1, bias=False)
        nn.init.normal_(self.score_head.weight, mean=0.0, std=1.0 / (config.d_model ** 0.5))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return reward scores of shape [batch] from the last non-pad token's hidden state."""
        out = self.backbone(input_ids, attention_mask=attention_mask, use_cache=False)
        hidden = out["last_hidden_state"]  # [B, T, d]

        if attention_mask is None:
            last_idx = torch.full(
                (input_ids.size(0),),
                input_ids.size(1) - 1,
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            lengths = attention_mask.long().sum(dim=1)
            last_idx = (lengths - 1).clamp(min=0)

        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        last_hidden = hidden.gather(1, gather_idx).squeeze(1)  # [B, d]
        return self.score_head(last_hidden).squeeze(-1)  # [B]


def pairwise_bt_loss(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry loss: -log sigmoid(r_chosen - r_rejected)."""
    return -F.logsigmoid(reward_chosen.float() - reward_rejected.float()).mean()


def pairwise_accuracy(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> float:
    """Probability that chosen scores higher than rejected (a [0,1] scalar)."""
    return (reward_chosen.float() > reward_rejected.float()).float().mean().item()


def load_backbone_from_causal_lm(
    reward_model: RewardModel, causal_lm_state: dict
) -> tuple[int, int]:
    """Initialize reward backbone from a LlamaForCausalLM state_dict.

    LlamaForCausalLM stores its backbone under the "model." prefix; we strip it
    and ignore the lm_head (we don't need it for reward modeling).
    Returns (n_missing, n_unexpected) from load_state_dict.
    """
    backbone_state: dict[str, torch.Tensor] = {}
    for k, v in causal_lm_state.items():
        if k.startswith("model."):
            backbone_state[k[len("model.") :]] = v
    result = reward_model.backbone.load_state_dict(backbone_state, strict=False)
    return len(result.missing_keys), len(result.unexpected_keys)


def build_reward_model_from_config(cfg: dict) -> RewardModel:
    """Factory mirroring build_llama_from_config but yielding a RewardModel."""
    m = cfg.get("model", cfg)
    config = LlamaConfig(
        vocab_size=m["vocab_size"],
        n_layer=m["n_layer"],
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_kv_heads=m["n_kv_heads"],
        context_length=m["context_length"],
        rope_theta=m.get("rope_theta", 10000.0),
        rms_norm_eps=m.get("rms_norm_eps", 1e-5),
        tie_word_embeddings=m.get("tie_word_embeddings", True),
        use_gradient_checkpointing=m.get("use_gradient_checkpointing", False),
    )
    return RewardModel(config)
