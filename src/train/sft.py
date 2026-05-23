"""
sft.py — 指令微调（冻结除最后 2 层外全部参数）

用途：
  - 加载 pretrain checkpoint
  - freeze_all_but_last_n_layers(2)
  - SFTDataset + padded batch 训练

运行：
  python -m src.train.sft --config configs/sft.yaml

TODO：
  - 打印 trainable/total 参数占比
  - 可选 untie embedding 对照实验
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

from src.data.sft import build_sft_dataloader
from src.model.llama import build_llama_from_config
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
    """Linear warmup + cosine decay for SFT."""
    tcfg = cfg["train"]
    warmup_steps = int(max_steps * tcfg.get("warmup_ratio", 0.03))
    if step < warmup_steps:
        return tcfg["learning_rate"] * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return tcfg["learning_rate"] * cosine


def train(cfg: dict) -> None:
    set_seed(cfg["project"]["seed"])
    device = torch.device(cfg["project"]["device"] if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg["project"]["dtype"] == "bfloat16" else torch.float16
    use_amp = device.type == "cuda"
    use_grad_scaler = use_amp and dtype is torch.float16
    scaler = GradScaler(enabled=use_grad_scaler)

    logger = init_logger(cfg, run_name=cfg["project"]["name"])

    model = build_llama_from_config(cfg).to(device)
    load_model_weights(cfg["paths"]["pretrain_ckpt"], model)

    trainable_layers = cfg["sft"].get("trainable_layers", "last_2")
    if trainable_layers == "last_2":
        model.freeze_all_but_last_n_layers(2)
    elif trainable_layers == "full":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unsupported sft.trainable_layers: {trainable_layers}")

    trainable, total = model.count_trainable_params()
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    _, dataset, loader = build_sft_dataloader(cfg)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    grad_accum = cfg["train"]["grad_accum_steps"]
    steps_per_epoch = max(1, len(loader) // grad_accum)
    planned_steps = steps_per_epoch * cfg["train"]["num_epochs"]
    max_steps = cfg["train"].get("max_steps") or planned_steps
    max_steps = min(max_steps, planned_steps)

    global_step = 0
    micro_step = 0
    running_loss = 0.0
    step_loss = 0.0
    best_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)
    model.train()

    print(
        f"Starting SFT: samples={len(dataset)}, max_steps={max_steps}, device={device}, "
        f"dtype={dtype}, grad_scaler={use_grad_scaler}, "
        f"micro_batch={cfg['train']['micro_batch_size']}, grad_accum={grad_accum}"
    )
    t0 = time.time()

    while global_step < max_steps:
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=use_amp)
            labels = batch["labels"].to(device, non_blocking=use_amp)
            attention_mask = batch["attention_mask"].to(device, non_blocking=use_amp)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                raw_loss = out["loss"]
                loss = raw_loss / grad_accum

            if not torch.isfinite(raw_loss):
                raise RuntimeError(f"Non-finite SFT loss at step {global_step}: {raw_loss.item()}")

            step_loss += raw_loss.item()
            scaler.scale(loss).backward()
            micro_step += 1

            if micro_step % grad_accum != 0:
                continue

            running_loss += step_loss / grad_accum
            step_loss = 0.0

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            lr = get_lr(global_step, cfg, max_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % cfg["train"]["log_interval"] == 0:
                avg_loss = running_loss / cfg["train"]["log_interval"]
                logger.log_metrics(global_step, {"sft/loss": avg_loss, "sft/lr": lr})
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_checkpoint(
                        ckpt_dir / "best.pt",
                        model,
                        optimizer,
                        global_step,
                        metrics={"sft/loss": avg_loss},
                    )
                    print(f"New best SFT loss={avg_loss:.4f} at step {global_step}")
                running_loss = 0.0

            if global_step % cfg["train"]["save_interval"] == 0:
                save_checkpoint(ckpt_dir / f"step_{global_step}.pt", model, optimizer, global_step)

            if global_step >= max_steps:
                break

    if global_step > 0:
        save_checkpoint(
            ckpt_dir / "last.pt",
            model,
            optimizer,
            global_step,
            metrics={"sft/best_loss": best_loss},
        )
    elapsed = time.time() - t0
    logger.log_metrics(global_step, {"sft/elapsed_s": elapsed})
    logger.finish()
    print(f"SFT done: {global_step} steps in {elapsed/3600:.2f}h, best_loss={best_loss:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT on Alpaca-Cleaned")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = parse_config_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
