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
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.sft import SFTDataset, collate_sft_batch
from src.model.llama import build_llama_from_config
from src.tokenizer import ProjectTokenizer
from src.utils.checkpoint import load_model_weights, save_checkpoint
from src.utils.config import add_config_args, parse_config_from_args
from src.utils.logging import init_logger


def train(cfg: dict) -> None:
    device = torch.device(cfg["project"]["device"] if torch.cuda.is_available() else "cpu")
    logger = init_logger(cfg, run_name=cfg["project"]["name"])

    tokenizer = ProjectTokenizer.load(cfg["paths"]["tokenizer_dir"])
    model = build_llama_from_config(cfg).to(device)
    load_model_weights(cfg["paths"]["pretrain_ckpt"], model)

    n_layers = cfg["sft"].get("trainable_layers", "last_2")
    if n_layers == "last_2":
        model.freeze_all_but_last_n_layers(2)

    trainable, total = model.count_trainable_params()
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    dataset = SFTDataset(
        tokenizer,
        max_samples=cfg["data"].get("max_samples"),
        max_length=cfg["data"]["max_seq_length"],
        dataset_name=cfg["data"]["dataset"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["micro_batch_size"],
        shuffle=True,
        collate_fn=collate_sft_batch,
        num_workers=cfg["data"].get("num_workers", 0),
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # TODO: epoch loop, forward with labels, save best
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"SFT skeleton ready: {len(dataset)} samples, {len(loader)} batches/epoch")
    logger.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT on Alpaca-Cleaned")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = parse_config_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
