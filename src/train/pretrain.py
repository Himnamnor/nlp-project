"""
pretrain.py — 预训练主循环

运行：
  python -m src.train.pretrain --config configs/pretrain.yaml
  python -m src.train.pretrain --config configs/pretrain.yaml --max_steps 2  # smoke
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
from torch.utils.data import DataLoader

from src.data.pretrain import PretrainDataset
from src.eval.ppl import compute_ppl
from src.model.llama import build_llama_from_config
from src.utils.checkpoint import load_model_weights, prune_old_checkpoints, save_checkpoint
from src.utils.config import add_config_args, parse_config_from_args
from src.utils.logging import init_logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step: int, cfg: dict, max_steps: int) -> float:
    """Linear warmup + cosine decay."""
    tcfg = cfg["train"]
    warmup = tcfg["warmup_steps"]
    if step < warmup:
        return tcfg["learning_rate"] * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return tcfg["min_lr"] + cosine * (tcfg["learning_rate"] - tcfg["min_lr"])


def train(cfg: dict) -> None:
    set_seed(cfg["project"]["seed"])
    device = torch.device(cfg["project"]["device"] if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg["project"]["dtype"] == "bfloat16" else torch.float16
    use_amp = device.type == "cuda"
    # fp16 needs loss scaling on V100 etc.; bf16 has enough dynamic range without it
    use_grad_scaler = use_amp and dtype is torch.float16
    scaler = GradScaler(enabled=use_grad_scaler)

    logger = init_logger(cfg, run_name=cfg["project"]["name"])
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = build_llama_from_config(cfg).to(device)
    init_from = cfg["paths"].get("init_from")
    if init_from:
        load_model_weights(init_from, model)
        print(f"Initialized model weights from {init_from}")
    if cfg["model"].get("use_gradient_checkpointing"):
        model.model.config.use_gradient_checkpointing = True

    train_ds = PretrainDataset(cfg["paths"]["train_bin"], cfg["model"]["context_length"])
    val_ds = PretrainDataset(cfg["paths"]["val_bin"], cfg["model"]["context_length"])
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["micro_batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=use_amp,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        betas=(cfg["train"]["beta1"], cfg["train"]["beta2"]),
        weight_decay=cfg["train"]["weight_decay"],
        fused=use_amp,
    )

    grad_accum = cfg["train"]["grad_accum_steps"]
    steps_per_epoch = len(train_loader) // grad_accum
    max_steps = cfg["train"].get("max_steps") or steps_per_epoch

    global_step = 0
    micro_step = 0
    best_ppl = float("inf")
    running_loss = 0.0
    step_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    model.train()
    print(
        f"Starting pretrain: max_steps={max_steps}, device={device}, dtype={dtype}, "
        f"grad_scaler={use_grad_scaler}, "
        f"micro_batch={cfg['train']['micro_batch_size']}, grad_accum={grad_accum}"
    )
    t0 = time.time()

    while global_step < max_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=use_amp)
            labels = batch["labels"].to(device, non_blocking=use_amp)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                out = model(input_ids=input_ids, labels=labels)
                loss = out["loss"] / grad_accum

            step_loss += out["loss"].item()
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
                logger.log_metrics(
                    global_step,
                    {"train/loss": avg_loss, "train/lr": lr},
                )
                running_loss = 0.0

            if global_step % cfg["train"]["eval_interval"] == 0:
                ppl = compute_ppl(
                    model,
                    val_ds,
                    device,
                    batch_size=cfg["train"]["micro_batch_size"],
                    max_batches=50,
                )
                logger.log_metrics(global_step, {"val/ppl": ppl})
                if ppl < best_ppl:
                    best_ppl = ppl
                    save_checkpoint(
                        ckpt_dir / "best.pt",
                        model,
                        optimizer,
                        global_step,
                        metrics={"val/ppl": ppl},
                    )
                    print(f"New best PPL={ppl:.2f} at step {global_step}")
                model.train()

            if global_step % cfg["train"]["save_interval"] == 0:
                save_checkpoint(
                    ckpt_dir / f"step_{global_step}.pt",
                    model,
                    optimizer,
                    global_step,
                )
                prune_old_checkpoints(ckpt_dir, cfg["train"].get("keep_last_n_ckpts", 2))

            if global_step >= max_steps:
                break

    elapsed = time.time() - t0
    logger.log_metrics(global_step, {"train/elapsed_s": elapsed})
    logger.finish()
    print(f"Pretrain done: {global_step} steps in {elapsed/3600:.2f}h, best PPL={best_ppl:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain Llama-mini on TinyStories")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = parse_config_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
