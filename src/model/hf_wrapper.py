"""
hf_wrapper.py — 将手写 Llama 包装为 HuggingFace PreTrainedModel

用途：
  - TRL (PPOTrainer, RewardTrainer, DPOTrainer) 与 PEFT (LoRA) 依赖 HF 接口
  - 暴露 q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj 命名供 LoRA target
  - 保证包装前后 forward / generate 输出一致（见 tests/test_hf_wrapper.py）

TODO 实现：
  1. class LlamaMiniConfig(PretrainedConfig)
  2. class LlamaMiniForCausalLM(PreTrainedModel, GenerationMixin)
  3. from_native(model: LlamaForCausalLM) -> LlamaMiniForCausalLM
  4. to_native() 反向转换（可选）
  5. register_for_auto_class("AutoModelForCausalLM")
"""

from __future__ import annotations

from typing import Optional

import torch

# TODO: uncomment when implementing
# from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
# from transformers.modeling_outputs import CausalLMOutputWithPast

from .llama import LlamaConfig, LlamaForCausalLM


class LlamaMiniConfig:
    """
    HF-compatible config stub.

    TODO: inherit PretrainedConfig, set model_type = "llama_mini"
    Fields mirror LlamaConfig.
    """

    model_type = "llama_mini"

    def __init__(self, **kwargs) -> None:
        self.vocab_size = kwargs.get("vocab_size", 16384)
        self.hidden_size = kwargs.get("d_model", kwargs.get("hidden_size", 512))
        self.num_hidden_layers = kwargs.get("n_layer", 6)
        self.num_attention_heads = kwargs.get("n_heads", 8)
        self.num_key_value_heads = kwargs.get("n_kv_heads", 4)
        self.max_position_embeddings = kwargs.get("context_length", 1024)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-5)
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)

    @classmethod
    def from_native(cls, config: LlamaConfig) -> "LlamaMiniConfig":
        return cls(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layer=config.n_layer,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            context_length=config.context_length,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            tie_word_embeddings=config.tie_word_embeddings,
        )


class LlamaMiniForCausalLM:
    """
    HF wrapper stub around native LlamaForCausalLM.

    TODO: inherit PreTrainedModel + GenerationMixin
    Delegate forward/generate to self.native_model
    Expose .model.layers[i].self_attn.q_proj etc. for PEFT
    """

    def __init__(self, config: LlamaMiniConfig, native_model: Optional[LlamaForCausalLM] = None) -> None:
        self.config = config
        if native_model is None:
            native_cfg = LlamaConfig(
                vocab_size=config.vocab_size,
                n_layer=config.num_hidden_layers,
                d_model=config.hidden_size,
                n_heads=config.num_attention_heads,
                n_kv_heads=config.num_key_value_heads,
                context_length=config.max_position_embeddings,
                rms_norm_eps=config.rms_norm_eps,
                rope_theta=config.rope_theta,
                tie_word_embeddings=config.tie_word_embeddings,
            )
            native_model = LlamaForCausalLM(native_cfg)
        self.native_model = native_model

    def forward(self, input_ids: torch.Tensor, attention_mask=None, labels=None, **kwargs):
        raise NotImplementedError("Wrap native_model.forward and return CausalLMOutputWithPast")

    @classmethod
    def from_native(cls, model: LlamaForCausalLM) -> "LlamaMiniForCausalLM":
        hf_cfg = LlamaMiniConfig.from_native(model.config)
        return cls(hf_cfg, native_model=model)

    def save_pretrained(self, path: str) -> None:
        """TODO: save config.json + model.safetensors via HF API."""
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, path: str) -> "LlamaMiniForCausalLM":
        """TODO: load HF-format checkpoint."""
        raise NotImplementedError
