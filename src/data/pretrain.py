"""
pretrain.py — TinyStories 预训练数据管线

用途：
  1. prepare_bin(cfg): 下载 TinyStories → tokenize → 写入 train.bin / val.bin (uint16 memmap)
  2. PretrainDataset: 随机切 context_length 窗口，返回 input_ids / labels (shifted)

运行 prepare：
  python -m src.data.pretrain --config configs/pretrain.yaml --prepare
  python -m src.data.pretrain --config configs/pretrain.yaml --prepare --max_samples 10000
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset

from src.tokenizer.bpe import ProjectTokenizer
from src.utils.config import add_config_args, parse_config_from_args

_BIN_MAGIC = b"LLMTOK01"


def prepare_bin(
    cfg: dict,
    max_samples: int | None = None,
    max_tokens: int | None = None,
) -> tuple[Path, Path]:
    """
    Tokenize TinyStories and write memmap binaries.

    Format: [magic 8B][num_tokens uint64][tokens uint16/uint32...]
    """
    tok = ProjectTokenizer.load(cfg["paths"]["tokenizer_dir"])
    out_dir = Path(cfg["paths"]["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(cfg["paths"]["train_bin"])
    val_path = Path(cfg["paths"]["val_bin"])
    tok_cfg = cfg["tokenizer"]
    val_ratio = tok_cfg.get("val_ratio", 0.005)
    text_field = tok_cfg.get("text_field", "text")
    max_tokens = max_tokens or tok_cfg.get("max_tokens")

    print(f"Loading {tok_cfg['dataset']} ...")
    dataset_config = tok_cfg.get("dataset_config")
    split = tok_cfg.get("split", "train")
    if dataset_config:
        ds = load_dataset(tok_cfg["dataset"], dataset_config, split=split)
    else:
        ds = load_dataset(tok_cfg["dataset"], split=split)
    if not isinstance(ds, HFDataset):
        raise TypeError(f"Expected Dataset, got {type(ds)}")

    all_ids: list[int] = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        item = cast(dict[str, Any], row)
        if text_field not in item or item[text_field] is None:
            continue
        ids = tok.encode(str(item[text_field]), add_eos=True)
        if max_tokens is not None and len(all_ids) + len(ids) > max_tokens:
            all_ids.extend(ids[: max_tokens - len(all_ids)])
            break
        all_ids.extend(ids)
        if (i + 1) % 10000 == 0:
            print(f"  tokenized {i + 1:,} stories, {len(all_ids):,} tokens ...")

    n_val = int(len(all_ids) * val_ratio)
    val_ids = all_ids[:n_val]
    train_ids = all_ids[n_val:]
    print(f"Total tokens: {len(all_ids):,} (train={len(train_ids):,}, val={len(val_ids):,})")

    def write_bin(path: Path, ids: list[int]) -> None:
        dtype = np.uint16 if tok.vocab_size < 65536 else np.uint32
        arr = np.array(ids, dtype=dtype)
        with open(path, "wb") as f:
            f.write(_BIN_MAGIC)
            f.write(struct.pack("<Q", len(arr)))
            arr.tofile(f)
        print(f"Wrote {path} ({len(arr):,} tokens, dtype={dtype})")

    write_bin(train_path, train_ids)
    write_bin(val_path, val_ids)
    return train_path, val_path


def read_bin_header(path: Path) -> tuple[int, np.dtype[Any]]:
    """Read num_tokens and dtype from .bin file."""
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == _BIN_MAGIC, f"Bad magic in {path}"
        (n,) = struct.unpack("<Q", f.read(8))
    file_size = path.stat().st_size - 16
    if file_size == n * 2:
        return n, np.dtype(np.uint16)
    if file_size == n * 4:
        return n, np.dtype(np.uint32)
    raise ValueError(f"Cannot infer dtype for {path}")


class PretrainDataset(Dataset):
    """Contiguous windows from memmap token file."""

    def __init__(self, bin_path: str | Path, context_length: int) -> None:
        self.context_length = context_length
        n_tokens, dtype = read_bin_header(Path(bin_path))
        self.data: np.memmap[Any, np.dtype[Any]] = np.memmap(
            bin_path, dtype=dtype, mode="r", offset=16, shape=(n_tokens,)
        )
        self.n_tokens = n_tokens

    def __len__(self) -> int:
        return max(0, self.n_tokens - self.context_length)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        max_start = self.n_tokens - self.context_length - 1
        start = (idx * 9973) % max(1, max_start + 1)
        chunk = np.asarray(self.data[start : start + self.context_length + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return {"input_ids": x, "labels": y}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TinyStories bin files")
    add_config_args(parser)
    parser.add_argument("--prepare", action="store_true", help="Run data preparation")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Use only first N stories for quick debug (default: all)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Stop after approximately N tokens (default: tokenizer.max_tokens or all)",
    )
    args = parser.parse_args()
    cfg = parse_config_from_args(args)
    if args.prepare:
        prepare_bin(cfg, max_samples=args.max_samples, max_tokens=args.max_tokens)
    else:
        print("Use --prepare to tokenize and write .bin files")


if __name__ == "__main__":
    main()
