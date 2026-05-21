"""
ppl.py — 验证集困惑度 (Perplexity)

用途：
  - compute_ppl(model, dataset, device, batch_size) -> float
  - PPL = exp(mean cross-entropy loss)
  - 目标: PPL < 40
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def compute_ppl(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 16,
    max_batches: int | None = None,
) -> float:
    """Compute perplexity on dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss, n_tokens = 0.0, 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch.get("labels", input_ids).to(device)
        out = model(input_ids=input_ids, labels=labels)
        loss = out["loss"]
        n = labels.numel()
        total_loss += loss.item() * n
        n_tokens += n

    if n_tokens == 0:
        return float("nan")
    avg_loss = total_loss / n_tokens
    return math.exp(avg_loss)
