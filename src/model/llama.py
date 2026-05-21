"""
llama.py — Llama3 风格 Decoder-only Transformer（从零手写）

架构要点（见 PLAN.md §2）：
  - RMSNorm (pre-norm, eps=1e-5, no bias)
  - RoPE on Q/K (theta=10000)
  - GQA: n_heads=8, n_kv_heads=4, repeat_kv for SDPA
  - SwiGLU FFN: down(silu(gate)*up), hidden ≈ round(8/3*d, 64)
  - All Linear bias=False; tied word embeddings + lm_head
  - Causal SDPA via F.scaled_dot_product_attention

默认规模：6L / d=512 / ~25-30M params
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class LlamaConfig:
    """Model hyper-parameters (matches configs/*.yaml model section)."""

    vocab_size: int = 16384
    n_layer: int = 6
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 4
    context_length: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    use_gradient_checkpointing: bool = False

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def intermediate_size(self) -> int:
        """SwiGLU hidden dim: round 8/3 * d_model to multiple of 64."""
        return int(math.ceil(8 * self.d_model / 3 / 64) * 64)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x*torch.rsqrt(torch.mean(x**2,dim=-1,keepdim=True)+self.eps)*self.weight


class RotaryEmbedding(nn.Module):
    """RoPE cos/sin cache for positions [0, context_length)."""

    def __init__(self, dim: int, max_position_embeddings: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [seq, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq, dim]

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        seq_len: int,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin for positions [offset, offset + seq_len]."""
        end = offset + seq_len
        if end > self.max_position_embeddings:
            raise ValueError(
                f"RoPE position {end - 1} exceeds max {self.max_position_embeddings - 1}"
            )
        cos_cached = cast(torch.Tensor, self.cos_cached)
        sin_cached = cast(torch.Tensor, self.sin_cached)
        return (
            cos_cached[offset:end],
            sin_cached[offset:end],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key.

    Args:
        q, k: [batch, n_heads, seq_len, head_dim]
        cos, sin: [seq_len, head_dim] (same device/dtype as q after cast)
    """
    # [1, 1, seq, dim] for broadcast over batch and heads
    cos = cos.unsqueeze(0).unsqueeze(0).to(dtype=q.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(dtype=q.dtype)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads from n_kv_heads to n_heads (GQA)."""
    if n_rep == 1:
        return hidden_states
    b, n_kv, s, d = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return hidden_states.reshape(b, n_kv * n_rep, s, d)


class LlamaAttention(nn.Module):
    """Multi-head causal self-attention with GQA + RoPE + SDPA."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.context_length, config.rope_theta)

    def _build_sdpa_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        q_len: int,
        kv_len: int,
        past_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build bool SDPA mask: True = allow attend.

        Causal rule with cache: query at absolute position (past_len + i)
        may attend to key positions j where j <= past_len + i.
        """
        q_pos = torch.arange(q_len,device=device)[:, None] + past_len
        k_pos = torch.arange(kv_len,device=device)[None, :]
        allow = k_pos <= q_pos  # [q_len, kv_len]
        allow = allow.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, q_len, kv_len).to(device)

        if attention_mask is not None:
            if attention_mask.shape[1] != kv_len:
                raise ValueError(
                    f"attention_mask length {attention_mask.shape[1]} != kv_len {kv_len}"
                )
            pad_allow = attention_mask[:, None, None, :].to(dtype=torch.bool)
            allow = allow & pad_allow

        return allow

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape
        past_len = past_key_value[0].shape[2] if past_key_value is not None else 0

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(q_len, offset=past_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_key_value = (k, v) if use_cache else None
        kv_len = k.shape[2]

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Pretrain: no pad mask → is_causal=True is enough.
        # SFT / cache: explicit bool mask (causal + padding).
        if attention_mask is not None or past_len > 0:
            sdpa_mask = self._build_sdpa_mask(
                attention_mask, bsz, q_len, kv_len, past_len, hidden_states.device
            )
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.n_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value


class LlamaMLP(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    """Single transformer block: pre-norm attention + pre-norm MLP."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states, attention_mask, past_key_value, use_cache
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states, present_key_value


class LlamaModel(nn.Module):
    """Transformer backbone (embeddings + layers + final norm)."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self._init_weights()

    def _init_weights(self) -> None:
        """GPT-2/Llama style init: N(0, 0.02), residual proj scaled."""
        init_std = 0.02
        n_layer = self.config.n_layer
        residual_std = init_std / math.sqrt(2 * n_layer)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=init_std)
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=init_std)
            elif isinstance(module, RMSNorm):
                module.weight.data.fill_(1.0)

        for i in range(len(self.layers)):
            layer = self.layers[i]
            if not isinstance(layer, LlamaDecoderLayer):
                continue
            layer.self_attn.o_proj.weight.data.normal_(mean=0.0, std=residual_std)
            layer.mlp.down_proj.weight.data.normal_(mean=0.0, std=residual_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> dict:
        hidden_states = self.embed_tokens(input_ids)
        next_cache: list[tuple[torch.Tensor, torch.Tensor]] = []

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            if (
                self.config.use_gradient_checkpointing
                and self.training
                and not use_cache
            ):
                hidden_states, present = self._checkpoint_layer(
                    layer, hidden_states, attention_mask, layer_past, use_cache
                )
            else:
                hidden_states, present = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                )
            if use_cache:
                next_cache.append(present)  # type: ignore[arg-type]

        hidden_states = self.norm(hidden_states)
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache if use_cache else None,
        }

    @staticmethod
    def _checkpoint_layer(layer, hidden_states, attention_mask, past_key_value, use_cache):
        """Gradient checkpoint wrapper (pretrain only, use_cache=False)."""

        def custom_forward(hs):
            out, present = layer(
                hs,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            return out

        hidden_states = checkpoint(custom_forward, hidden_states, use_reentrant=False)
        present = None
        return hidden_states, present

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """Sample one token per row from logits [batch, vocab]."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        top_values, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < top_values[:, [-1]], float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class LlamaForCausalLM(nn.Module):
    """Causal LM with tied embeddings; returns logits + optional loss."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        else:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> dict:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)

        result: dict = {"logits": logits, "past_key_values": outputs["past_key_values"]}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV cache.

        流程：
          1. 首次 forward 整个 prompt，建立 past_key_values
          2. 之后每步只输入最新 token，复用 cache
          3. 对最后位置 logits 采样下一个 token
        """
        self.eval()
        generated = input_ids
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values is None:
                model_input = generated
            else:
                model_input = generated[:, -1:]

            cur_len = generated.shape[1]
            if cur_len >= self.config.context_length:
                break

            attn_len = model_input.shape[1] if past_key_values is None else generated.shape[1]
            attn_mask = torch.ones(
                generated.shape[0],
                attn_len,
                device=generated.device,
                dtype=torch.long,
            )

            outputs = self.model(
                model_input,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs["past_key_values"]
            logits = self.lm_head(outputs["last_hidden_state"][:, -1, :])
            next_token = _sample_next_token(logits, temperature, top_k, top_p)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def freeze_all_but_last_n_layers(self, n: int = 2) -> None:
        """Freeze all params except last n decoder layers + final norm + lm_head."""
        for p in self.parameters():
            p.requires_grad = False
        for layer in self.model.layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.model.norm.parameters():
            p.requires_grad = True
        for p in self.lm_head.parameters():
            p.requires_grad = True

    def count_trainable_params(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def build_llama_from_config(cfg: dict) -> LlamaForCausalLM:
    """Factory: build model from loaded YAML config dict."""
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
    return LlamaForCausalLM(config)
