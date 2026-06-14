"""Export a slim checkpoint for download/evaluation.

Training checkpoints include optimizer state, which can be 2-3x larger than the
model weights. This script keeps the fields needed by load_model_weights().

Examples:
  python scripts/export_slim_ckpt.py checkpoints/pretrain_general/best.pt
  python scripts/export_slim_ckpt.py checkpoints/ppo_general/best.pt --output checkpoints/ppo_general/best.slim.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def _size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def export_slim(input_path: Path, output_path: Path) -> None:
    ckpt: dict[str, Any] = torch.load(input_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"{input_path} does not contain model_state_dict")

    slim: dict[str, Any] = {
        "step": ckpt.get("step", 0),
        "model_state_dict": ckpt["model_state_dict"],
        "metrics": ckpt.get("metrics", {}),
        "extra": ckpt.get("extra", {}),
    }
    # Keep PPO value head if present; harmless for non-PPO checkpoints.
    if "value_head_state_dict" in ckpt:
        slim["value_head_state_dict"] = ckpt["value_head_state_dict"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, output_path)
    print(
        f"Saved {output_path} "
        f"({ _size_mb(input_path):.1f} MB -> { _size_mb(output_path):.1f} MB)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip optimizer state from a checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = args.checkpoint.with_name(args.checkpoint.stem + ".slim.pt")
    export_slim(args.checkpoint, output)


if __name__ == "__main__":
    main()
