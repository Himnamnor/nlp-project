"""
pretrain.py — 通用 causal LM 预训练数据管线

用途：
  1. prepare_bin(cfg): 下载/stream HF 文本语料 → tokenize → 写入 train.bin / val.bin
  2. PretrainDataset: 随机切 context_length 窗口，返回 input_ids / labels (shifted)

运行 prepare：
  python -m src.data.pretrain --config configs/pretrain.yaml --prepare
  python -m src.data.pretrain --config configs/pretrain.yaml --prepare --max_samples 10000
"""

from __future__ import annotations

import argparse
import random
import struct
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from torch.utils.data import Dataset

from src.tokenizer.bpe import ProjectTokenizer
from src.utils.config import add_config_args, parse_config_from_args

_BIN_MAGIC = b"LLMTOK01"


class _TokenBinWriter:
    """Streaming writer for the Project token binary format."""

    def __init__(self, path: Path, dtype: np.dtype[Any]) -> None:
        self.path = path
        self.dtype = dtype
        self.n_tokens = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "wb")
        self._f.write(_BIN_MAGIC)
        self._f.write(struct.pack("<Q", 0))  # patched in close()

    def write(self, ids: list[int]) -> None:
        if not ids:
            return
        arr = np.asarray(ids, dtype=self.dtype)
        arr.tofile(self._f)
        self.n_tokens += int(arr.size)

    def close(self) -> None:
        self._f.seek(8)
        self._f.write(struct.pack("<Q", self.n_tokens))
        self._f.close()
        print(f"Wrote {self.path} ({self.n_tokens:,} tokens, dtype={self.dtype})")


def _normalise_sources(tok_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return data sources from either legacy single-dataset or new multi-source config."""
    raw_sources = tok_cfg.get("sources") or tok_cfg.get("datasets")
    if raw_sources:
        sources = [dict(src) for src in raw_sources]
    else:
        sources = [
            {
                "dataset": tok_cfg["dataset"],
                "dataset_config": tok_cfg.get("dataset_config"),
                "split": tok_cfg.get("split", "train"),
                "text_field": tok_cfg.get("text_field", "text"),
                "streaming": tok_cfg.get("streaming", False),
            }
        ]

    for src in sources:
        src.setdefault("split", tok_cfg.get("split", "train"))
        src.setdefault("text_field", tok_cfg.get("text_field", "text"))
        src.setdefault("streaming", tok_cfg.get("streaming", False))
    return sources


def _load_source(src: dict[str, Any], seed: int) -> HFDataset | HFIterableDataset:
    dataset_name = src.get("dataset") or src.get("name")
    if not dataset_name:
        raise ValueError(f"Dataset source missing 'dataset'/'name': {src}")
    dataset_config = src.get("dataset_config") or src.get("config")
    split = src.get("split", "train")
    streaming = bool(src.get("streaming", False))
    kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    if src.get("data_files") is not None:
        kwargs["data_files"] = src["data_files"]
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, **kwargs)
    else:
        ds = load_dataset(dataset_name, **kwargs)
    if not isinstance(ds, (HFDataset, HFIterableDataset)):
        raise TypeError(f"Expected Dataset, got {type(ds)}")
    shuffle_buffer = int(src.get("shuffle_buffer_size", 0) or 0)
    if shuffle_buffer > 0 and isinstance(ds, HFIterableDataset):
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return ds


def _source_token_budgets(
    sources: list[dict[str, Any]], max_tokens: int | None
) -> list[int | None]:
    """Allocate a global token budget across sources by source.weight."""
    explicit = [src.get("max_tokens") for src in sources]
    if any(x is not None for x in explicit):
        return [None if x is None else int(x) for x in explicit]
    if max_tokens is None:
        return [None for _ in sources]

    weights = [float(src.get("weight", 1.0)) for src in sources]
    total_weight = sum(weights) or 1.0
    budgets = [int(max_tokens * w / total_weight) for w in weights]
    # Keep exact total after flooring.
    for i in range(max_tokens - sum(budgets)):
        budgets[i % len(budgets)] += 1
    return budgets


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
    max_tokens = max_tokens or tok_cfg.get("max_tokens")
    seed = int(cfg.get("project", {}).get("seed", 42))
    sources = _normalise_sources(tok_cfg)
    budgets = _source_token_budgets(sources, None if max_tokens is None else int(max_tokens))
    dtype = np.dtype(np.uint16 if tok.vocab_size < 65536 else np.uint32)
    rng = random.Random(seed)

    print(
        f"Preparing token bins from {len(sources)} source(s), "
        f"max_tokens={max_tokens}, val_ratio={val_ratio}, dtype={dtype}"
    )
    train_writer = _TokenBinWriter(train_path, dtype=dtype)
    val_writer = _TokenBinWriter(val_path, dtype=dtype)
    total_docs = 0
    total_tokens = 0

    try:
        for src_id, (src, source_budget) in enumerate(zip(sources, budgets), start=1):
            print(
                "Loading data source "
                f"{src_id}/{len(sources)}: {src.get('dataset') or src.get('name')}"
                f" config={src.get('dataset_config') or src.get('config')}"
                f" split={src.get('split', 'train')}"
                f" text_field={src.get('text_field', 'text')}"
                f" streaming={src.get('streaming', False)}"
                f" token_budget={source_budget}"
            )
            ds = _load_source(src, seed=seed + src_id)
            text_field = str(src.get("text_field", "text"))
            source_docs = 0
            source_tokens = 0
            source_sample_limit = src.get("max_samples")
            if source_sample_limit is not None:
                source_sample_limit = int(source_sample_limit)
                if max_samples is not None:
                    source_sample_limit = min(source_sample_limit, max_samples)
            else:
                source_sample_limit = max_samples

            for row_idx, row in enumerate(ds):
                if source_sample_limit is not None and row_idx >= source_sample_limit:
                    break
                item = cast(dict[str, Any], row)
                text = item.get(text_field)
                if text is None:
                    continue
                text = str(text).strip()
                if not text:
                    continue
                ids = tok.encode(text, add_eos=True)
                if source_budget is not None:
                    remaining = source_budget - source_tokens
                    if remaining <= 0:
                        break
                    if len(ids) > remaining:
                        ids = ids[:remaining]
                if not ids:
                    continue

                if rng.random() < float(val_ratio):
                    val_writer.write(ids)
                else:
                    train_writer.write(ids)
                n_ids = len(ids)
                source_docs += 1
                source_tokens += n_ids
                total_docs += 1
                total_tokens += n_ids

                if total_docs % 10000 == 0 or total_tokens // 5_000_000 > (
                    (total_tokens - n_ids) // 5_000_000
                ):
                    print(
                        f"  docs={total_docs:,}, tokens={total_tokens:,}, "
                        f"train={train_writer.n_tokens:,}, val={val_writer.n_tokens:,}"
                    )
            print(
                f"Finished source {src_id}: docs={source_docs:,}, tokens={source_tokens:,}"
            )
    finally:
        train_writer.close()
        val_writer.close()

    print(
        f"Total tokens: {total_tokens:,} "
        f"(train={train_writer.n_tokens:,}, val={val_writer.n_tokens:,})"
    )
    if train_writer.n_tokens <= cfg["model"]["context_length"]:
        raise RuntimeError("Train split is too small for the configured context_length")
    if val_writer.n_tokens <= cfg["model"]["context_length"]:
        raise RuntimeError("Val split is too small for the configured context_length")
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
