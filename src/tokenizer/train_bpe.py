"""
train_bpe.py — 在通用语料上训练 BPE 词表

运行：
  python -m src.tokenizer.train_bpe --config configs/pretrain.yaml
  python -m src.tokenizer.train_bpe --config configs/pretrain.yaml --max_samples 50000  # 调试
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from datasets import Dataset as HFDataset
from datasets import IterableDataset, load_dataset

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from src.utils.config import add_config_args, parse_config_from_args


def _normalise_sources(tok_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return tokenizer training sources from either legacy or new config shape."""
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
                "streaming": tok_cfg.get("streaming", True),
            }
        ]

    for src in sources:
        src.setdefault("split", tok_cfg.get("split", "train"))
        src.setdefault("text_field", tok_cfg.get("text_field", "text"))
        src.setdefault("streaming", tok_cfg.get("streaming", True))
    return sources


def _load_source(src: dict[str, Any]) -> HFDataset | IterableDataset:
    """Load one HuggingFace dataset source."""
    dataset_name = src.get("dataset") or src.get("name")
    if not dataset_name:
        raise ValueError(f"Dataset source missing 'dataset'/'name': {src}")
    dataset_config = src.get("dataset_config") or src.get("config")
    split = src.get("split", "train")
    streaming = bool(src.get("streaming", True))
    kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    if src.get("data_files") is not None:
        kwargs["data_files"] = src["data_files"]
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, **kwargs)
    else:
        ds = load_dataset(dataset_name, **kwargs)
    if not isinstance(ds, (HFDataset, IterableDataset)):
        raise TypeError(f"Expected Dataset/IterableDataset, got {type(ds)}")
    shuffle_buffer = int(src.get("shuffle_buffer_size", 0) or 0)
    if shuffle_buffer > 0 and isinstance(ds, IterableDataset):
        ds = ds.shuffle(
            seed=int(src.get("seed", 42)),
            buffer_size=shuffle_buffer,
        )
    return ds


def _iter_texts(
    ds: HFDataset | IterableDataset,
    text_field: str,
    max_samples: int | None,
) -> Iterator[str]:
    """Yield non-empty text rows from a HuggingFace dataset."""
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        item = cast(dict[str, Any], row)
        text = item.get(text_field)
        if text is None:
            continue
        text = str(text).strip()
        if text:
            yield text


def train_bpe(cfg: dict, max_samples: int | None = None) -> Path:
    """Train BPE and save to paths.tokenizer_dir."""
    tok_cfg = cfg["tokenizer"]
    out_dir = Path(cfg["paths"]["tokenizer_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = _normalise_sources(tok_cfg)
    max_train_samples = max_samples
    if max_train_samples is None:
        max_train_samples = tok_cfg.get("max_train_samples")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

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
        total_seen = 0
        for src_id, src in enumerate(sources, start=1):
            remaining_global = (
                None
                if max_train_samples is None
                else max(0, int(max_train_samples) - total_seen)
            )
            if remaining_global == 0:
                break
            source_limit = src.get("max_train_samples", src.get("max_samples"))
            if source_limit is not None:
                source_limit = int(source_limit)
                if remaining_global is not None:
                    source_limit = min(source_limit, remaining_global)
            else:
                source_limit = remaining_global

            print(
                "Loading tokenizer source "
                f"{src_id}/{len(sources)}: {src.get('dataset') or src.get('name')}"
                f" config={src.get('dataset_config') or src.get('config')}"
                f" split={src.get('split', 'train')}"
                f" text_field={src.get('text_field', 'text')}"
                f" streaming={src.get('streaming', True)}"
                f" sample_limit={source_limit}"
            )
            ds = _load_source(src)
            source_seen = 0
            for text in _iter_texts(ds, str(src.get("text_field", "text")), source_limit):
                batch.append(text)
                total_seen += 1
                source_seen += 1
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            print(f"  consumed {source_seen:,} tokenizer rows from source {src_id}")
        if batch:
            yield batch

    print("Training BPE ...")
    if max_train_samples is not None:
        print(f"  (sample cap: max_train_samples={max_train_samples})")
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
