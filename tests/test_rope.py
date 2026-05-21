"""
test_rope.py — RoPE 旋转位置编码单元测试

运行: pytest tests/test_rope.py -v
"""

import pytest
import torch

from src.model.llama import RotaryEmbedding, apply_rotary_pos_emb, rotate_half


def test_rope_cache_shape():
    head_dim = 64
    max_seq = 128
    rope = RotaryEmbedding(head_dim, max_seq, theta=10000.0)
    cos, sin = rope(seq_len=32)
    assert cos.shape == (32, head_dim)
    assert sin.shape == (32, head_dim)
    assert rope.cos_cached.shape == (max_seq, head_dim)


def test_rope_moves_with_model():
    rope = RotaryEmbedding(64, 128)
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rope = rope.to("cuda")
    cos, sin = rope(16)
    assert cos.device.type == "cuda"
    assert sin.device.type == "cuda"


def test_rope_offset_slice():
    rope = RotaryEmbedding(64, 128)
    cos_full, _ = rope(10, offset=0)
    cos_off, _ = rope(10, offset=5)
    assert torch.allclose(cos_off, rope.cos_cached[5:15])


def test_rotate_half():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    out = rotate_half(x)
    expected = torch.tensor([[[-3.0, -4.0, 1.0, 2.0]]])
    assert torch.allclose(out, expected)


def test_rope_relative_invariance():
    """<R_i q, R_j k> = <q, R_{j-i} k> (RoPE depends only on relative position)."""
    head_dim = 64
    rope = RotaryEmbedding(head_dim, 128)
    i, j = 2, 5
    rel = j - i

    q_vec = torch.randn(head_dim)
    k_vec = torch.randn(head_dim)
    cos, sin = rope(j + 1)

    q_b = q_vec.view(1, 1, 1, head_dim)
    k_b = k_vec.view(1, 1, 1, head_dim)

    q_i = q_b * cos[i : i + 1].view(1, 1, 1, head_dim) + rotate_half(q_b) * sin[i : i + 1].view(
        1, 1, 1, head_dim
    )
    k_j = k_b * cos[j : j + 1].view(1, 1, 1, head_dim) + rotate_half(k_b) * sin[j : j + 1].view(
        1, 1, 1, head_dim
    )
    dot_absolute = (q_i[0, 0, 0] * k_j[0, 0, 0]).sum()

    k_rel = k_b * cos[rel : rel + 1].view(1, 1, 1, head_dim) + rotate_half(k_b) * sin[
        rel : rel + 1
    ].view(1, 1, 1, head_dim)
    dot_relative = (q_vec * k_rel[0, 0, 0]).sum()

    assert torch.allclose(dot_absolute, dot_relative, atol=1e-5)
