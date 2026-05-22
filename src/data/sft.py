"""
sft.py — Alpaca-Cleaned 指令微调数据

用途：
  - 加载 yahma/alpaca-cleaned，过滤 ≥ min_samples 条
  - Alpaca prompt 模板 + response-only loss mask (labels=-100 on prefix)
  - collate_fn: dynamic padding + attention_mask

模板：
  ### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}

TODO：
  - format_alpaca(example) -> full_text, response_start_char
  - SFTDataset.__getitem__ 返回 input_ids, labels, attention_mask
  - build_sft_dataloader(cfg)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.tokenizer import ProjectTokenizer

ALPACA_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def format_alpaca(instruction: str, input_text: str, output: str) -> tuple[str, str]:
    """Return (full_text, prompt_prefix) where prefix ends before response tokens."""
    input_text = input_text.strip() if input_text else ""
    prefix = ALPACA_TEMPLATE.format(instruction=instruction, input=input_text, output="")
    full = ALPACA_TEMPLATE.format(instruction=instruction, input=input_text, output=output)
    return full, prefix


class SFTDataset(Dataset):
    """Alpaca-style SFT with masked labels on prompt prefix."""

    def __init__(
        self,
        tokenizer: ProjectTokenizer,
        split: str = "train",
        max_samples: int | None = None,
        max_length: int = 1024,
        dataset_name: str = "yahma/alpaca-cleaned",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        ds = load_dataset(dataset_name, split="train")
        if not isinstance(ds, HFDataset):
            raise TypeError(f"Expected HuggingFace Dataset, got {type(ds)}")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.examples = ds

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = cast(dict[str, Any], self.examples[idx])
        full, prefix = format_alpaca(
            str(ex["instruction"]),
            str(ex.get("input") or ""),
            str(ex["output"]),
        )
        full_ids = self.tokenizer.encode(full, add_eos=True)
        prefix_ids = self.tokenizer.encode(prefix, add_eos=False)

        full_ids = full_ids[: self.max_length]
        if len(full_ids) < 2:
            full_ids = full_ids + [self.tokenizer.eos_id or 0]

        input_ids = full_ids[:-1]
        labels = full_ids[1:].copy()

        # labels[i] is token i+1. Mask targets before the response starts,
        # but keep the first response token as a supervised target.
        n_mask = max(min(len(prefix_ids), len(full_ids)) - 1, 0)
        labels[:n_mask] = [-100] * n_mask

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_sft_batch(batch: list[dict], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """Pad batch to max length in batch."""
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn_mask = [], [], []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(
            torch.cat([b["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        labels.append(
            torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
        attn_mask.append(
            torch.cat(
                [
                    torch.ones(len(b["input_ids"]), dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long),
                ]
            )
        )
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attn_mask),
    }


def build_sft_dataloader(cfg: dict):
    """Build tokenizer-backed SFT dataset and dataloader from cfg."""
    tokenizer = ProjectTokenizer.load(cfg["paths"]["tokenizer_dir"])
    dataset = SFTDataset(
        tokenizer,
        max_samples=cfg["data"].get("max_samples"),
        max_length=cfg["data"]["max_seq_length"],
        dataset_name=cfg["data"]["dataset"],
    )
    collate_fn: Callable[[list[dict]], dict[str, torch.Tensor]] = lambda batch: collate_sft_batch(
        batch, pad_id=tokenizer.pad_id or 0
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["micro_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg["data"].get("num_workers", 0),
        drop_last=True,
    )
    return tokenizer, dataset, loader
