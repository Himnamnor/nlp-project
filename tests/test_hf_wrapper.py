"""
test_hf_wrapper.py — HF 包装层与 native 模型输出一致

运行: pytest tests/test_hf_wrapper.py -v
"""

import pytest
import torch

from src.model.hf_wrapper import LlamaMiniForCausalLM
from src.model.llama import LlamaConfig, LlamaForCausalLM


@pytest.mark.skip(reason="Implement HF wrapper forward first")
def test_wrapper_logits_match_native():
    cfg = LlamaConfig(context_length=64, vocab_size=1024)
    native = LlamaForCausalLM(cfg)
    wrapper = LlamaMiniForCausalLM.from_native(native)
    x = torch.randint(0, cfg.vocab_size, (1, 32))
    # native_out = native(x)
    # wrapper_out = wrapper(x)
    # assert torch.allclose(native_out["logits"], wrapper_out.logits, atol=1e-5)
