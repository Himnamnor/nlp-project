"""
dpo.py — DPO（Direct Preference Optimization）训练循环

设计：
  - policy: LlamaForCausalLM, 从 SFT checkpoint 初始化, 可训练
  - ref_model: 同一份 SFT checkpoint 的深拷贝, 冻结, eval()
  - 不需要 reward model, 不需要在线 rollout, 一阶段优化

DPO loss（Rafailov et al. 2023）：
  L = -log sigmoid( beta * ( log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x) ) )

其中 log pi(y|x) 是 response 部分 token log-prob 的求和 (sum over response).

运行：
  python -m src.train.dpo --config configs/dpo.yaml
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from src.data.rlhf import build_dpo_dataloaders
from src.model.llama import build_llama_from_config
from src.tokenizer import ProjectTokenizer
from src.utils.checkpoint import load_model_weights, save_checkpoint
from src.utils.config import add_config_args, parse_config_from_args
from src.utils.logging import init_logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step: int, cfg: dict, max_steps: int) -> float:
    """Linear warmup + cosine decay (与 SFT/Reward 一致)."""
    tcfg = cfg["train"]
    warmup_steps = int(max_steps * tcfg.get("warmup_ratio", 0.03))
    base_lr = float(tcfg["learning_rate"])
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * cosine


def response_logprob_sum(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, response_mask: torch.Tensor
) -> torch.Tensor:
    """Sum of log p(y_t | x, y_<t) over response tokens; returns [B]."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out["logits"]
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = input_ids[:, 1:]
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    # response_mask aligned to input_ids; response token at position t corresponds to logits[t-1]
    # so we use response_mask shifted by one (drop the BOS/prompt-start position).
    mask = response_mask[:, 1:].float()
    return (token_logp * mask).sum(dim=1)  # [B]


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard DPO loss + implicit reward statistics.

    Returns (loss, chosen_reward, rejected_reward, accuracy_per_batch).
    """
    pi_log_ratios = policy_chosen_logps - policy_rejected_logps
    ref_log_ratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_log_ratios - ref_log_ratios)
    loss = -F.logsigmoid(logits).mean()
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    acc = (chosen_reward > rejected_reward).float().mean()
    return loss, chosen_reward, rejected_reward, acc


@torch.no_grad()
def evaluate(
    policy, ref_model, loader, device, dtype, use_amp, beta: float
) -> dict[str, float]:
    policy.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    chosen_reward_acc, rejected_reward_acc = 0.0, 0.0
    for batch in loader:
        chosen_ids = batch["chosen_ids"].to(device)
        chosen_attn = batch["chosen_attn"].to(device)
        chosen_resp = batch["chosen_response_mask"].to(device)
        rejected_ids = batch["rejected_ids"].to(device)
        rejected_attn = batch["rejected_attn"].to(device)
        rejected_resp = batch["rejected_response_mask"].to(device)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            policy_chosen_lp = response_logprob_sum(policy, chosen_ids, chosen_attn, chosen_resp)
            policy_rejected_lp = response_logprob_sum(policy, rejected_ids, rejected_attn, rejected_resp)
            ref_chosen_lp = response_logprob_sum(ref_model, chosen_ids, chosen_attn, chosen_resp)
            ref_rejected_lp = response_logprob_sum(ref_model, rejected_ids, rejected_attn, rejected_resp)
        loss, c_r, r_r, acc = dpo_loss(
            policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta
        )
        bsz = chosen_ids.size(0)
        total_loss += loss.item() * bsz
        total_acc += acc.item() * bsz
        chosen_reward_acc += c_r.mean().item() * bsz
        rejected_reward_acc += r_r.mean().item() * bsz
        n += bsz
    policy.train()
    return {
        "val_loss": total_loss / max(n, 1),
        "val_acc": total_acc / max(n, 1),
        "val_chosen_reward": chosen_reward_acc / max(n, 1),
        "val_rejected_reward": rejected_reward_acc / max(n, 1),
    }


def train(cfg: dict) -> None:
    set_seed(int(cfg["project"]["seed"]))
    device = torch.device(cfg["project"]["device"] if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg["project"]["dtype"] == "bfloat16" else torch.float16
    use_amp = device.type == "cuda"
    use_grad_scaler = use_amp and dtype is torch.float16
    scaler = GradScaler(enabled=use_grad_scaler)

    logger = init_logger(cfg, run_name=cfg["project"]["name"])
    tokenizer = ProjectTokenizer.load(cfg["paths"]["tokenizer_dir"])

    # policy: trainable, from SFT
    policy = build_llama_from_config(cfg).to(device)
    load_model_weights(cfg["paths"]["sft_ckpt"], policy)

    # ref: frozen deep-copy of policy
    ref_model = copy.deepcopy(policy).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    train_loader, val_loader = build_dpo_dataloaders(cfg, tokenizer)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = int(cfg["train"].get("num_epochs", 1))
    planned_steps = num_epochs * len(train_loader)
    max_steps = cfg["train"].get("max_steps") or planned_steps
    max_steps = min(max_steps, planned_steps)
    log_interval = int(cfg["train"].get("log_interval", 20))
    eval_interval = int(cfg["train"].get("eval_interval", 200))
    save_interval = int(cfg["train"].get("save_interval", 500))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    beta = float(cfg["dpo"]["beta"])

    print(
        f"Starting hand-written DPO: train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"max_steps={max_steps} beta={beta} dtype={dtype} grad_scaler={use_grad_scaler}"
    )

    global_step = 0
    running = {"loss": 0.0, "acc": 0.0, "c_r": 0.0, "r_r": 0.0}
    best_acc = -1.0
    t0 = time.time()
    policy.train()

    done = False
    for _ in range(num_epochs):
        if done:
            break
        for batch in train_loader:
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_attn = batch["chosen_attn"].to(device)
            chosen_resp = batch["chosen_response_mask"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_attn = batch["rejected_attn"].to(device)
            rejected_resp = batch["rejected_response_mask"].to(device)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                policy_chosen_lp = response_logprob_sum(policy, chosen_ids, chosen_attn, chosen_resp)
                policy_rejected_lp = response_logprob_sum(policy, rejected_ids, rejected_attn, rejected_resp)
                with torch.no_grad():
                    ref_chosen_lp = response_logprob_sum(ref_model, chosen_ids, chosen_attn, chosen_resp)
                    ref_rejected_lp = response_logprob_sum(ref_model, rejected_ids, rejected_attn, rejected_resp)
                loss, chosen_reward, rejected_reward, acc = dpo_loss(
                    policy_chosen_lp,
                    policy_rejected_lp,
                    ref_chosen_lp,
                    ref_rejected_lp,
                    beta,
                )

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite DPO loss at step {global_step}: {loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            lr = get_lr(global_step, cfg, max_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            scaler.step(optimizer)
            scaler.update()

            running["loss"] += loss.item()
            running["acc"] += acc.item()
            running["c_r"] += chosen_reward.mean().item()
            running["r_r"] += rejected_reward.mean().item()
            global_step += 1

            if global_step % log_interval == 0:
                logger.log_metrics(
                    global_step,
                    {
                        "dpo/loss": running["loss"] / log_interval,
                        "dpo/acc": running["acc"] / log_interval,
                        "dpo/chosen_reward": running["c_r"] / log_interval,
                        "dpo/rejected_reward": running["r_r"] / log_interval,
                        "dpo/margin": (running["c_r"] - running["r_r"]) / log_interval,
                        "dpo/lr": lr,
                    },
                )
                running = {k: 0.0 for k in running}

            if global_step % eval_interval == 0:
                metrics = evaluate(policy, ref_model, val_loader, device, dtype, use_amp, beta)
                logger.log_metrics(global_step, {f"dpo/{k}": v for k, v in metrics.items()})
                print(
                    f"[eval @ {global_step}] val_loss={metrics['val_loss']:.4f} "
                    f"val_acc={metrics['val_acc']:.4f} "
                    f"chosen_r={metrics['val_chosen_reward']:.4f} "
                    f"rejected_r={metrics['val_rejected_reward']:.4f}"
                )
                if metrics["val_acc"] > best_acc:
                    best_acc = metrics["val_acc"]
                    save_checkpoint(
                        ckpt_dir / "best.pt",
                        policy,
                        optimizer,
                        global_step,
                        metrics={f"dpo/{k}": v for k, v in metrics.items()},
                    )
                    print(f"New best DPO val_acc={best_acc:.4f} at step {global_step}")

            if global_step % save_interval == 0:
                save_checkpoint(
                    ckpt_dir / f"step_{global_step}.pt", policy, optimizer, global_step
                )

            if global_step >= max_steps:
                done = True
                break

    # final eval & save
    metrics = evaluate(policy, ref_model, val_loader, device, dtype, use_amp, beta)
    if metrics["val_acc"] > best_acc:
        best_acc = metrics["val_acc"]
        save_checkpoint(
            ckpt_dir / "best.pt",
            policy,
            optimizer,
            global_step,
            metrics={f"dpo/{k}": v for k, v in metrics.items()},
        )
        print(f"New best DPO val_acc={best_acc:.4f} at step {global_step}")
    save_checkpoint(
        ckpt_dir / "last.pt",
        policy,
        optimizer,
        global_step,
        metrics={"dpo/best_val_acc": best_acc, **{f"dpo/{k}": v for k, v in metrics.items()}},
    )
    elapsed = time.time() - t0
    print(
        f"DPO done: {global_step} steps in {elapsed/3600:.2f}h, "
        f"final val_acc={metrics['val_acc']:.4f}, best_val_acc={best_acc:.4f}"
    )
    logger.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hand-written DPO (advanced: PPO contrast)")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = parse_config_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
