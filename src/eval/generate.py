"""
generate.py — 文本生成与样例导出

用途：
  - generate_stories(model, tokenizer, n=5) 用于预训练阶段 qualitative eval
  - generate_instruction_response(model, tokenizer, instruction) 用于 SFT eval
  - 保存到 logs/*/samples.txt

运行：
  python -m src.eval.generate --config configs/pretrain.yaml --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tokenizer import ProjectTokenizer
from src.utils.config import add_config_args, parse_config_from_args


def generate_from_prompt(
    model,
    tokenizer: ProjectTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> str:
    """Generate continuation from prompt."""
    # TODO: encode prompt → model.generate → decode
    raise NotImplementedError


def save_samples(samples: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            f.write(f"=== Sample {i+1} ===\n{s}\n\n")
    print(f"Saved {len(samples)} samples → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text samples")
    add_config_args(parser)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    args = parser.parse_args()
    cfg = parse_config_from_args(args)
    print(f"Generate skeleton: prompt={args.prompt!r}")
    # TODO: load model + tokenizer, generate, save


if __name__ == "__main__":
    main()
