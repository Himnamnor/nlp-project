"""
checkpoint.py — 模型权重保存与加载

用途：
  - save_checkpoint(model, optimizer, step, path, extra=...)
  - load_checkpoint(path, model, optimizer=None) -> dict
  - 支持仅加载 model state（SFT 接 pretrain、RLHF 接 SFT）
  - 保留 best.pt + step_*.pt，自动清理 keep_last_n

TODO：
  - 实现 torch.save / load with map_location
  - 实现 save_best(metric, value) 逻辑
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, path)
    print(f"Saved checkpoint → {path}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load checkpoint into model (and optionally optimizer)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Loaded checkpoint ← {path} (step={ckpt.get('step', '?')})")
    return ckpt


def load_model_weights(path: str | Path, model: nn.Module, strict: bool = True) -> None:
    """Load only model weights (ignore optimizer)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)


def prune_old_checkpoints(ckpt_dir: str | Path, keep_last_n: int = 2, keep_best: bool = True) -> None:
    """Remove old step_*.pt files, keep best.pt."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return
    steps = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    to_remove = steps[:-keep_last_n] if len(steps) > keep_last_n else []
    for p in to_remove:
        p.unlink()
        print(f"Removed old checkpoint {p}")
