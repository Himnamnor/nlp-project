"""
reward.py — 奖励模型训练（PKU-SafeRLHF + Bradley-Terry pairwise loss）

流程：
  1. 从 SFT/pretrain checkpoint 初始化 RewardModel backbone（score_head 随机）
  2. PreferencePairDataset 给出 (chosen_ids, rejected_ids) batch
  3. 计算 r_chosen, r_rejected，BT 损失 -log σ(r_c - r_r)
  4. 在验证集上汇报 pairwise accuracy（目标 ≥0.60）
  5. 按 best val_acc 保存 checkpoints/reward/best.pt（PPO 阶段加载）

运行：
  python -m src.train.reward --config configs/reward.yaml
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler

from src.data.rlhf import build_reward_dataloaders
from src.model.reward import (
    build_reward_model_from_config,
    load_backbone_from_causal_lm,
    pairwise_accuracy,
    pairwise_bt_loss,
)
from src.tokenizer import ProjectTokenizer
from src.utils.checkpoint import save_checkpoint
from src.utils.config import add_config_args, parse_config_from_args
from src.utils.logging import init_logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step: int, cfg: dict, max_steps: int) -> float:
    """Linear warmup + cosine decay (same shape as SFT)."""
    tcfg = cfg["train"]
    warmup_steps = int(max_steps * tcfg.get("warmup_ratio", 0.03))
    base_lr = float(tcfg["learning_rate"])
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * cosine


@torch.no_grad()
def evaluate(model, loader, device, dtype, use_amp) -> tuple[float, float]:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for batch in loader:
        chosen_ids = batch["chosen_ids"].to(device)
        chosen_mask = batch["chosen_mask"].to(device)
        rejected_ids = batch["rejected_ids"].to(device)
        rejected_mask = batch["rejected_mask"].to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            r_c = model(chosen_ids, attention_mask=chosen_mask)
            r_r = model(rejected_ids, attention_mask=rejected_mask)
        loss = pairwise_bt_loss(r_c, r_r)
        acc = pairwise_accuracy(r_c, r_r)
        bsz = chosen_ids.size(0)
        total_loss += loss.item() * bsz
        total_acc += acc * bsz
        n += bsz
    model.train()
    return total_loss / max(n, 1), total_acc / max(n, 1)


def train(cfg: dict) -> None:
    set_seed(int(cfg["project"]["seed"]))
    device = torch.device(cfg["project"]["device"] if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg["project"]["dtype"] == "bfloat16" else torch.float16
    use_amp = device.type == "cuda"
    use_grad_scaler = use_amp and dtype is torch.float16
    scaler = GradScaler(enabled=use_grad_scaler)

    logger = init_logger(cfg, run_name=cfg["project"]["name"])
    tokenizer = ProjectTokenizer.load(cfg["paths"]["tokenizer_dir"])

    model = build_reward_model_from_config(cfg).to(device)
    sft_ckpt = cfg["paths"].get("sft_ckpt")
    if sft_ckpt:
        ckpt = torch.load(sft_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        n_missing, n_unexpected = load_backbone_from_causal_lm(model, state)
        print(
            f"Initialized reward backbone from {sft_ckpt} "
            f"(missing={n_missing}, unexpected={n_unexpected})"
        )

    train_loader, val_loader = build_reward_dataloaders(cfg, tokenizer)

    optimizer = torch.optim.AdamW(
        model.parameters(),
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

    print(
        f"Starting reward training: train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)} max_steps={max_steps} "
        f"dtype={dtype} grad_scaler={use_grad_scaler}"
    )

    global_step = 0
    running_loss = 0.0
    running_acc = 0.0
    best_acc = -1.0
    t0 = time.time()
    model.train()

    done = False
    for _ in range(num_epochs):
        if done:
            break
        for batch in train_loader:
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                r_c = model(chosen_ids, attention_mask=chosen_mask)
                r_r = model(rejected_ids, attention_mask=rejected_mask)
                loss = pairwise_bt_loss(r_c, r_r)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite reward loss at step {global_step}: {loss.item()}"
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            lr = get_lr(global_step, cfg, max_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                running_loss += loss.item()
                running_acc += pairwise_accuracy(r_c, r_r)

            global_step += 1

            if global_step % log_interval == 0:
                avg_loss = running_loss / log_interval
                avg_acc = running_acc / log_interval
                logger.log_metrics(
                    global_step,
                    {"reward/loss": avg_loss, "reward/acc": avg_acc, "reward/lr": lr},
                )
                running_loss = 0.0
                running_acc = 0.0

            if global_step % eval_interval == 0:
                val_loss, val_acc = evaluate(model, val_loader, device, dtype, use_amp)
                logger.log_metrics(
                    global_step, {"reward/val_loss": val_loss, "reward/val_acc": val_acc}
                )
                print(
                    f"[eval @ {global_step}] val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                )
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_checkpoint(
                        ckpt_dir / "best.pt",
                        model,
                        optimizer,
                        global_step,
                        metrics={
                            "reward/val_acc": val_acc,
                            "reward/val_loss": val_loss,
                        },
                    )
                    print(f"New best reward val_acc={val_acc:.4f} at step {global_step}")

            if global_step % save_interval == 0:
                save_checkpoint(
                    ckpt_dir / f"step_{global_step}.pt", model, optimizer, global_step
                )

            if global_step >= max_steps:
                done = True
                break

    # final eval + last.pt
    val_loss, val_acc = evaluate(model, val_loader, device, dtype, use_amp)
    logger.log_metrics(global_step, {"reward/val_loss": val_loss, "reward/val_acc": val_acc})
    if val_acc > best_acc:
        best_acc = val_acc
        save_checkpoint(
            ckpt_dir / "best.pt",
            model,
            optimizer,
            global_step,
            metrics={"reward/val_acc": val_acc, "reward/val_loss": val_loss},
        )
        print(f"New best reward val_acc={val_acc:.4f} at step {global_step}")
    save_checkpoint(
        ckpt_dir / "last.pt",
        model,
        optimizer,
        global_step,
        metrics={"reward/best_val_acc": best_acc},
    )
    elapsed = time.time() - t0
    print(
        f"Reward training done: {global_step} steps in {elapsed/3600:.2f}h, "
        f"final val_acc={val_acc:.4f}, best_val_acc={best_acc:.4f}"
    )
    logger.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reward model on PKU-SafeRLHF")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = parse_config_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
