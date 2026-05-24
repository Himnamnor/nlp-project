"""
ppo.py - hand-written PPO RLHF training loop.

Pipeline:
  1. Load SFT policy and frozen reference model.
  2. Load reward model trained on PKU-SafeRLHF preferences.
  3. Sample prompts, generate responses with the current policy.
  4. Build token-level rewards: terminal reward score plus KL penalty.
  5. Run clipped PPO updates on policy and value head.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.data.rlhf import build_prompt_dataset
from src.model.llama import LlamaForCausalLM, build_llama_from_config
from src.model.reward import build_reward_model_from_config
from src.tokenizer import ProjectTokenizer
from src.utils.checkpoint import load_model_weights
from src.utils.config import add_config_args, parse_config_from_args
from src.utils.logging import init_logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PolicyWithValue(nn.Module):
    """Causal LM policy plus a scalar value head over decoder hidden states."""

    def __init__(self, policy: LlamaForCausalLM) -> None:
        super().__init__()
        self.policy = policy
        self.value_head = nn.Linear(policy.config.d_model, 1, bias=False)
        nn.init.normal_(
            self.value_head.weight,
            mean=0.0,
            std=1.0 / math.sqrt(policy.config.d_model),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.policy.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden = out["last_hidden_state"]
        logits = self.policy.lm_head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        return {"logits": logits, "values": values}


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    rewards: torch.Tensor
    reward_scores: torch.Tensor


def collate_prompts(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids, attention_mask = [], []
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - ids.numel()
        input_ids.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)]))
        attention_mask.append(
            torch.cat(
                [
                    torch.ones(ids.numel(), dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long),
                ]
            )
        )
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "prompt_text": [item["prompt_text"] for item in batch],
        "prompt_formatted": [item["prompt_formatted"] for item in batch],
    }


def sequence_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Return log p(x_t | x_<t) aligned to input_ids[:, 1:]."""
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    labels = input_ids[:, 1:].unsqueeze(-1)
    return log_probs.gather(dim=-1, index=labels).squeeze(-1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def top_p_sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits.float() / temperature
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_one(
    model: PolicyWithValue,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int | None,
) -> torch.Tensor:
    generated = prompt_ids.unsqueeze(0)
    past_key_values = None
    model.policy.eval()

    for _ in range(max_new_tokens):
        if generated.size(1) >= model.policy.config.context_length:
            break
        model_input = generated if past_key_values is None else generated[:, -1:]
        attn_len = model_input.size(1) if past_key_values is None else generated.size(1)
        attn_mask = torch.ones(
            generated.size(0),
            attn_len,
            dtype=torch.long,
            device=generated.device,
        )
        out = model.policy.model(
            model_input,
            attention_mask=attn_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = out["past_key_values"]
        logits = model.policy.lm_head(out["last_hidden_state"][:, -1, :])
        next_token = top_p_sample(logits, temperature=temperature, top_p=top_p)
        generated = torch.cat([generated, next_token], dim=1)
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break
    return generated.squeeze(0)


def pad_rollout_sequences(
    sequences: list[torch.Tensor],
    prompt_lens: list[int],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(seq.numel() for seq in sequences)
    input_ids, attention_mask, response_mask = [], [], []
    for seq, prompt_len in zip(sequences, prompt_lens):
        pad_len = max_len - seq.numel()
        padded = torch.cat([seq, torch.full((pad_len,), pad_id, dtype=torch.long)])
        attn = torch.cat(
            [
                torch.ones(seq.numel(), dtype=torch.long, device=seq.device),
                torch.zeros(pad_len, dtype=torch.long, device=seq.device),
            ]
        )
        # Align to logprobs for input_ids[:, 1:]; label position k predicts token k+1.
        label_positions = torch.arange(max_len - 1, device=seq.device) + 1
        resp = (label_positions >= prompt_len) & (label_positions < seq.numel())
        input_ids.append(padded)
        attention_mask.append(attn)
        response_mask.append(resp.float())
    return torch.stack(input_ids), torch.stack(attention_mask), torch.stack(response_mask)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(rewards.size(0), device=rewards.device)
    for t in reversed(range(rewards.size(1))):
        next_values = values[:, t + 1] if t < rewards.size(1) - 1 else torch.zeros_like(lastgaelam)
        next_mask = mask[:, t + 1] if t < rewards.size(1) - 1 else torch.zeros_like(lastgaelam)
        delta = rewards[:, t] + gamma * next_values * next_mask - values[:, t]
        lastgaelam = delta + gamma * lam * next_mask * lastgaelam
        lastgaelam = lastgaelam * mask[:, t]
        advantages[:, t] = lastgaelam
    returns = advantages + values
    return advantages, returns


@torch.no_grad()
def collect_rollout(
    policy: PolicyWithValue,
    ref_model: LlamaForCausalLM,
    reward_model: nn.Module,
    batch: dict[str, Any],
    cfg: dict,
    pad_id: int,
    eos_id: int | None,
    device: torch.device,
    dtype: torch.dtype,
    use_amp: bool,
) -> RolloutBatch:
    ppo_cfg = cfg["ppo"]
    prompt_ids = batch["input_ids"].to(device)
    prompt_mask = batch["attention_mask"].to(device)
    prompt_lens = prompt_mask.long().sum(dim=1).tolist()

    sequences: list[torch.Tensor] = []
    for i, prompt_len in enumerate(prompt_lens):
        ids = prompt_ids[i, :prompt_len]
        seq = generate_one(
            policy,
            ids,
            max_new_tokens=int(ppo_cfg.get("max_new_tokens", 96)),
            temperature=float(ppo_cfg.get("temperature", 1.0)),
            top_p=float(ppo_cfg.get("top_p", 0.9)),
            eos_token_id=eos_id,
        )
        sequences.append(seq)

    input_ids, attention_mask, response_mask = pad_rollout_sequences(
        sequences, [int(x) for x in prompt_lens], pad_id=pad_id
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    response_mask = response_mask.to(device)

    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
        policy_out = policy(input_ids, attention_mask=attention_mask)
        ref_out = ref_model(input_ids=input_ids, attention_mask=attention_mask)
        reward_scores = reward_model(input_ids, attention_mask=attention_mask)

    old_logprobs = sequence_logprobs(policy_out["logits"], input_ids)
    ref_logprobs = sequence_logprobs(ref_out["logits"], input_ids)
    old_values = policy_out["values"][:, :-1].float()

    kl = old_logprobs - ref_logprobs
    rewards = -float(ppo_cfg.get("init_kl_coef", 0.1)) * kl
    rewards = rewards * response_mask
    reward_clip = ppo_cfg.get("reward_clip")
    if reward_clip is not None:
        reward_scores = reward_scores.clamp(-float(reward_clip), float(reward_clip))

    for i in range(rewards.size(0)):
        positions = response_mask[i].nonzero(as_tuple=False).flatten()
        if positions.numel() > 0:
            rewards[i, positions[-1]] += reward_scores[i].float()

    advantages, returns = compute_gae(
        rewards.float(),
        old_values.float(),
        response_mask.float(),
        gamma=float(ppo_cfg.get("gamma", 1.0)),
        lam=float(ppo_cfg.get("lam", 0.95)),
    )
    valid_adv = advantages[response_mask.bool()]
    if valid_adv.numel() > 1:
        advantages = (advantages - valid_adv.mean()) / (valid_adv.std(unbiased=False) + 1e-8)
        advantages = advantages * response_mask

    return RolloutBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask.float(),
        old_logprobs=old_logprobs.detach(),
        ref_logprobs=ref_logprobs.detach(),
        old_values=old_values.detach(),
        returns=returns.detach(),
        advantages=advantages.detach(),
        rewards=rewards.detach(),
        reward_scores=reward_scores.detach().float(),
    )


def ppo_update(
    policy: PolicyWithValue,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    rollout: RolloutBatch,
    cfg: dict,
    device: torch.device,
    dtype: torch.dtype,
    use_amp: bool,
) -> dict[str, float]:
    ppo_cfg = cfg["ppo"]
    batch_size = rollout.input_ids.size(0)
    mini_batch_size = int(ppo_cfg.get("mini_batch_size", batch_size))
    ppo_epochs = int(ppo_cfg.get("ppo_epochs", 4))
    cliprange = float(ppo_cfg.get("cliprange", 0.2))
    cliprange_value = float(ppo_cfg.get("cliprange_value", 0.2))
    vf_coef = float(ppo_cfg.get("vf_coef", 0.5))
    entropy_coef = float(ppo_cfg.get("entropy_coef", 0.0))
    grad_clip = float(ppo_cfg.get("grad_clip", 1.0))

    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clipfrac": 0.0,
    }
    n_updates = 0

    indices = torch.arange(batch_size, device=device)
    for _ in range(ppo_epochs):
        perm = indices[torch.randperm(batch_size, device=device)]
        for start in range(0, batch_size, mini_batch_size):
            mb_idx = perm[start : start + mini_batch_size]
            if mb_idx.numel() == 0:
                continue

            input_ids = rollout.input_ids[mb_idx]
            attention_mask = rollout.attention_mask[mb_idx]
            mask = rollout.response_mask[mb_idx]
            if mask.sum().item() == 0:
                continue

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                out = policy(input_ids, attention_mask=attention_mask)
                new_logprobs = sequence_logprobs(out["logits"], input_ids)
                values = out["values"][:, :-1].float()

            old_logprobs = rollout.old_logprobs[mb_idx]
            old_values = rollout.old_values[mb_idx]
            returns = rollout.returns[mb_idx]
            advantages = rollout.advantages[mb_idx]

            log_ratio = (new_logprobs - old_logprobs) * mask
            ratio = torch.exp(log_ratio)
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
            policy_loss = masked_mean(torch.maximum(pg_loss1, pg_loss2), mask)

            values_clipped = old_values + (values - old_values).clamp(
                -cliprange_value, cliprange_value
            )
            vf_loss1 = (values - returns) ** 2
            vf_loss2 = (values_clipped - returns) ** 2
            value_loss = 0.5 * masked_mean(torch.maximum(vf_loss1, vf_loss2), mask)

            probs = F.softmax(out["logits"][:, :-1, :].float(), dim=-1)
            token_entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
            entropy = masked_mean(token_entropy, mask)
            loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite PPO loss: {loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                approx_kl = masked_mean(old_logprobs - new_logprobs, mask)
                clipfrac = (((ratio - 1.0).abs() > cliprange).float() * mask).sum() / mask.sum()
                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy.item())
                totals["approx_kl"] += float(approx_kl.item())
                totals["clipfrac"] += float(clipfrac.item())
                n_updates += 1

    return {k: v / max(n_updates, 1) for k, v in totals.items()}


def resolve_reward_ckpt(path: str | Path) -> Path:
    path = Path(path)
    return path / "best.pt" if path.is_dir() else path


def save_policy_checkpoint(
    path: str | Path,
    policy: "PolicyWithValue",
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save PPO policy in LlamaForCausalLM-compatible format.

    The "model_state_dict" key matches what build_llama_from_config + load_model_weights
    expects, so generate.py / sft_eval.py / safety_eval.py can directly load this checkpoint.
    value_head and PPO meta-state live under "value_head_state_dict" / "extra".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model_state_dict": policy.policy.state_dict(),
        "value_head_state_dict": policy.value_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics or {},
        "extra": extra or {},
    }
    torch.save(state, path)
    print(f"Saved PPO checkpoint → {path}")


class AdaptiveKLController:
    """Heess-style adaptive KL controller used by InstructGPT.

    Updates kl_coef every step to keep observed kl close to target_kl. If target_kl
    is None we keep kl_coef fixed.
    """

    def __init__(self, init_kl_coef: float, target_kl: float | None, horizon: int = 10000) -> None:
        self.value = float(init_kl_coef)
        self.target_kl = target_kl
        self.horizon = max(int(horizon), 1)

    def update(self, current_kl: float, n_steps: int) -> None:
        if self.target_kl is None or self.target_kl <= 0:
            return
        proportional_error = (current_kl / self.target_kl) - 1.0
        # Clip the update so a single bad batch can't blow up coef
        proportional_error = max(min(proportional_error, 0.2), -0.2)
        mult = 1.0 + proportional_error * n_steps / self.horizon
        self.value = max(self.value * mult, 1e-4)


def train(cfg: dict) -> None:
    set_seed(int(cfg["project"]["seed"]))
    device = torch.device(cfg["project"]["device"] if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg["project"]["dtype"] == "bfloat16" else torch.float16
    use_amp = device.type == "cuda"
    use_grad_scaler = use_amp and dtype is torch.float16
    scaler = GradScaler(enabled=use_grad_scaler)

    logger = init_logger(cfg, run_name=cfg["project"]["name"])
    tokenizer = ProjectTokenizer.load(cfg["paths"]["tokenizer_dir"])
    pad_id = tokenizer.pad_id if tokenizer.pad_id is not None else 0

    base_policy = build_llama_from_config(cfg).to(device)
    load_model_weights(cfg["paths"]["sft_ckpt"], base_policy)
    policy = PolicyWithValue(base_policy).to(device)

    ref_model = build_llama_from_config(cfg).to(device)
    load_model_weights(cfg["paths"]["sft_ckpt"], ref_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    reward_model = build_reward_model_from_config(cfg).to(device)
    reward_ckpt = resolve_reward_ckpt(cfg["paths"]["reward_ckpt"])
    load_model_weights(reward_ckpt, reward_model)
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False

    prompt_ds = build_prompt_dataset(cfg, tokenizer)
    loader = DataLoader(
        prompt_ds,
        batch_size=int(cfg["ppo"].get("batch_size", 8)),
        shuffle=True,
        collate_fn=lambda batch: collate_prompts(batch, pad_id=pad_id),
        num_workers=int(cfg["data"].get("num_workers", 0)),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(cfg["ppo"].get("learning_rate", 1e-5)),
        weight_decay=float(cfg["ppo"].get("weight_decay", 0.0)),
    )

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    total_steps = int(cfg["ppo"].get("total_steps", 500))
    log_interval = int(cfg["ppo"].get("log_interval", 10))
    save_interval = int(cfg["ppo"].get("save_interval", 100))

    kl_ctrl = AdaptiveKLController(
        init_kl_coef=float(cfg["ppo"].get("init_kl_coef", 0.1)),
        target_kl=cfg["ppo"].get("target_kl"),
        horizon=int(cfg["ppo"].get("kl_horizon", 10000)),
    )
    best_reward = -float("inf")

    print(
        f"Starting hand-written PPO: prompts={len(prompt_ds)} total_steps={total_steps} "
        f"batch={cfg['ppo'].get('batch_size', 8)} mini_batch={cfg['ppo'].get('mini_batch_size', 2)} "
        f"dtype={dtype} grad_scaler={use_grad_scaler}"
    )

    global_step = 0
    t0 = time.time()
    done = False
    while not done:
        for batch in loader:
            # ---- rollout（采样 & 老 log prob / value） ----
            policy.eval()
            rollout_cfg = dict(cfg)
            rollout_cfg["ppo"] = {**cfg["ppo"], "init_kl_coef": kl_ctrl.value}
            rollout = collect_rollout(
                policy,
                ref_model,
                reward_model,
                batch,
                rollout_cfg,
                pad_id=pad_id,
                eos_id=tokenizer.eos_id,
                device=device,
                dtype=dtype,
                use_amp=use_amp,
            )

            # ---- PPO inner update ----
            policy.train()
            metrics = ppo_update(
                policy,
                optimizer,
                scaler,
                rollout,
                cfg,
                device=device,
                dtype=dtype,
                use_amp=use_amp,
            )
            with torch.no_grad():
                kl = (rollout.old_logprobs - rollout.ref_logprobs) * rollout.response_mask
                reward_mean = float(rollout.reward_scores.mean().item())
                kl_mean = float(masked_mean(kl, rollout.response_mask).item())
                metrics.update(
                    {
                        "reward_mean": reward_mean,
                        "reward_std": float(rollout.reward_scores.std(unbiased=False).item()),
                        "kl_mean": kl_mean,
                        "kl_coef": kl_ctrl.value,
                        "tokens_per_batch": float(rollout.response_mask.sum().item()),
                    }
                )

            # ---- 自适应 KL 系数 ----
            kl_ctrl.update(current_kl=kl_mean, n_steps=int(cfg["ppo"].get("batch_size", 8)))

            global_step += 1
            if global_step % log_interval == 0 or global_step == 1:
                logger.log_metrics(global_step, {f"ppo/{k}": v for k, v in metrics.items()})

            if reward_mean > best_reward:
                best_reward = reward_mean
                save_policy_checkpoint(
                    ckpt_dir / "best.pt",
                    policy,
                    optimizer,
                    global_step,
                    metrics={f"ppo/{k}": v for k, v in metrics.items()},
                    extra={"kl_coef": kl_ctrl.value},
                )

            if global_step % save_interval == 0:
                save_policy_checkpoint(
                    ckpt_dir / f"step_{global_step}.pt",
                    policy,
                    optimizer,
                    global_step,
                    metrics={f"ppo/{k}": v for k, v in metrics.items()},
                    extra={"kl_coef": kl_ctrl.value},
                )

            if global_step >= total_steps:
                done = True
                break

    elapsed = time.time() - t0
    save_policy_checkpoint(
        ckpt_dir / "last.pt",
        policy,
        optimizer,
        global_step,
        metrics={"ppo/elapsed_s": elapsed, "ppo/best_reward": best_reward},
        extra={"kl_coef": kl_ctrl.value},
    )
    logger.log_metrics(global_step, {"ppo/elapsed_s": elapsed, "ppo/best_reward": best_reward})
    logger.finish()
    print(
        f"PPO done: {global_step} steps in {elapsed/3600:.2f}h, best_reward={best_reward:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hand-written PPO alignment")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = parse_config_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
