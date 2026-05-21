"""
test_llama.py — Llama 模型 shape / forward 测试

运行: pytest tests/test_llama.py -v
"""

import pytest
import torch

from src.model.llama import LlamaAttention, LlamaConfig, LlamaForCausalLM


def test_attention_forward_shape():
    cfg = LlamaConfig(d_model=512, n_heads=8, n_kv_heads=4, context_length=128)
    attn = LlamaAttention(cfg)
    x = torch.randn(2, 32, cfg.d_model)
    out, cache = attn(x)
    assert out.shape == (2, 32, cfg.d_model)
    assert cache is None


def test_attention_with_padding_mask():
    cfg = LlamaConfig(d_model=128, n_heads=4, n_kv_heads=2, context_length=64)
    attn = LlamaAttention(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    mask = torch.ones(2, 16)
    mask[0, 12:] = 0  # pad last 4 tokens
    out, _ = attn(x, attention_mask=mask)
    assert out.shape == (2, 16, cfg.d_model)
    assert torch.isfinite(out).all()


def test_attention_kv_cache():
    cfg = LlamaConfig(d_model=128, n_heads=4, n_kv_heads=2, context_length=64)
    attn = LlamaAttention(cfg)
    x1 = torch.randn(1, 8, cfg.d_model)
    out1, cache = attn(x1, use_cache=True)
    assert cache is not None
    past_k, past_v = cache
    assert past_k.shape == (1, cfg.n_kv_heads, 8, cfg.head_dim)

    x2 = torch.randn(1, 1, cfg.d_model)
    out2, cache2 = attn(x2, past_key_value=cache, use_cache=True)
    assert out2.shape == (1, 1, cfg.d_model)
    assert cache2[0].shape[2] == 9


def test_model_param_count_order():
    """Rough param count should be ~25-30M for default config."""
    cfg = LlamaConfig()
    model = LlamaForCausalLM(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert 20_000_000 < n < 50_000_000, f"Unexpected param count: {n:,}"


def test_forward_shape():
    cfg = LlamaConfig(context_length=128, vocab_size=1024)
    model = LlamaForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    out = model(x, labels=x)
    assert out["logits"].shape == (2, 64, cfg.vocab_size)
    assert "loss" in out


def test_generate_extends_sequence():
    cfg = LlamaConfig(context_length=64, vocab_size=256, n_layer=2, d_model=128, n_heads=4, n_kv_heads=2)
    model = LlamaForCausalLM(cfg)
    model.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model.generate(prompt, max_new_tokens=5, temperature=0.0, top_k=None, top_p=None)
    assert out.shape == (1, 13)
    assert torch.equal(out[:, :8], prompt)


def test_generate_respects_context_length():
    cfg = LlamaConfig(context_length=16, vocab_size=64, n_layer=2, d_model=64, n_heads=4, n_kv_heads=2)
    model = LlamaForCausalLM(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 10))
    out = model.generate(prompt, max_new_tokens=100, temperature=0.0, top_k=None, top_p=None)
    assert out.shape[1] <= cfg.context_length


def test_freeze_last_two_layers():
    cfg = LlamaConfig(n_layer=6)
    model = LlamaForCausalLM(cfg)
    model.freeze_all_but_last_n_layers(2)
    trainable, total = model.count_trainable_params()
    assert trainable < total
    assert trainable > 0
