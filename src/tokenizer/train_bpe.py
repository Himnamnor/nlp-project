"""
train_bpe.py — 在 TinyStories 上训练 BPE 词表

运行：
  python -m src.tokenizer.train_bpe --config configs/pretrain.yaml
  python -m src.tokenizer.train_bpe --config configs/pretrain.yaml --max_samples 50000  # 调试
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from datasets import IterableDataset, load_dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from src.utils.config import add_config_args, parse_config_from_args


def _iter_texts(ds: IterableDataset, max_samples: int | None) -> Iterator[str]:
    """Yield text field from streaming TinyStories rows."""
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        item = cast(dict[str, Any], row)
        yield str(item["text"])


def train_bpe(cfg: dict, max_samples: int | None = None) -> Path:
    """Train BPE and save to paths.tokenizer_dir."""
    tok_cfg = cfg["tokenizer"]
    out_dir = Path(cfg["paths"]["tokenizer_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {tok_cfg['dataset']} (streaming) ...")
    ds = load_dataset(tok_cfg["dataset"], split="train", streaming=True)
    if not isinstance(ds, IterableDataset):
        ds = cast(IterableDataset, ds)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    special = list(tok_cfg["special_tokens"].values()) + ["<unk>"]
    trainer = BpeTrainer(
        vocab_size=tok_cfg["vocab_size"],
        min_frequency=tok_cfg.get("min_frequency", 2),
        special_tokens=special,
        show_progress=True,
    )

    def batch_iterator() -> Iterator[list[str]]:
        batch: list[str] = []
        batch_size = 1000
        for text in _iter_texts(ds, max_samples):
            batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    print("Training BPE ...")
    if max_samples is not None:
        print(f"  (debug mode: max_samples={max_samples})")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    out_path = out_dir / "tokenizer.json"
    tokenizer.save(str(out_path))
    print(f"Saved tokenizer → {out_path} (vocab_size={tokenizer.get_vocab_size()})")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE on TinyStories")
    add_config_args(parser)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Use only first N stories for quick debug (default: all)",
    )
    args = parser.parse_args()
    cfg = parse_config_from_args(args)
    train_bpe(cfg, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
