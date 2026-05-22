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

import torch

from src.model.llama import build_llama_from_config
from src.tokenizer import ProjectTokenizer
from src.utils.config import add_config_args, parse_config_from_args
from src.utils.checkpoint import load_model_weights


def generate_from_prompt(
    model,
    tokenizer: ProjectTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> str:
    """Generate continuation from prompt."""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    model.eval()
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_id,
    )
    return tokenizer.decode(outputs[0].tolist())


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
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path to load")
    parser.add_argument("--output", type=str, default=None, help="Where to save generated samples")
    args = parser.parse_args()
    cfg = parse_config_from_args(args)
    print(f"Generate skeleton: prompt={args.prompt!r}")

    tokenizer = ProjectTokenizer.load(cfg["paths"]["tokenizer_dir"])
    model = build_llama_from_config(cfg)
    device = torch.device(cfg["project"]["device"])
    model.to(device)
    ckpt_path = args.ckpt or str(Path(cfg["paths"]["checkpoint_dir"]) / "best.pt")
    load_model_weights(ckpt_path, model)
    model.eval()

    output = generate_from_prompt(model, tokenizer, args.prompt)
    print(f"Generated output: {output}")
    output_path = Path(args.output or Path(cfg["paths"]["log_dir"]) / "samples.txt")
    save_samples([output], output_path)


if __name__ == "__main__":
    main()
