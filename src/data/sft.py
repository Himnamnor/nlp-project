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

from typing import Any

import torch
from datasets import load_dataset
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
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.examples = ds

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        full, prefix = format_alpaca(ex["instruction"], ex.get("input") or "", ex["output"])
        full_ids = self.tokenizer.encode(full, add_eos=True)
        prefix_ids = self.tokenizer.encode(prefix, add_eos=False)

        # Truncate
        full_ids = full_ids[: self.max_length]
        labels = full_ids.copy()
        # Mask prompt prefix
        n_prefix = min(len(prefix_ids), len(labels))
        labels[:n_prefix] = [-100] * n_prefix

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_sft_batch(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad batch to max length in batch."""
    pad_id = 0  # TODO: use tokenizer.pad_id
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
            torch.cat([torch.ones(len(b["input_ids"])), torch.zeros(pad_len)])
        )
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attn_mask),
    }


def build_sft_dataloader(cfg: dict):
    """TODO: wire SFTDataset + DataLoader from cfg."""
    raise NotImplementedError
