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
from datetime import datetime

import torch

from src.data.sft import format_alpaca
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
    new_tokens = outputs[0, input_ids.shape[1] :].tolist()
    return tokenizer.decode(new_tokens)


def save_sample(
    sample: str,
    out_path: Path,
    prompt: str,
    ckpt_path: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> None:
    """Append one generated sample with enough context for later report writing."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(f"=== Sample @ {timestamp} ===\n")
        f.write(f"checkpoint: {ckpt_path}\n")
        f.write(
            f"temperature: {temperature}, top_p: {top_p}, max_new_tokens: {max_new_tokens}\n"
        )
        f.write("prompt:\n")
        f.write(prompt.rstrip() + "\n\n")
        f.write("output:\n")
        f.write(sample.rstrip() + "\n\n")
    print(f"Appended sample → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text samples")
    add_config_args(parser)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path to load")
    parser.add_argument("--output", type=str, default=None, help="Where to save generated samples")
    args = parser.parse_args()
    cfg = parse_config_from_args(args)
    prompt = args.prompt
    if args.instruction is not None:
        _, prompt = format_alpaca(args.instruction, args.input, "")
    print(f"Generate skeleton: prompt={prompt!r}")

    tokenizer = ProjectTokenizer.load(cfg["paths"]["tokenizer_dir"])
    model = build_llama_from_config(cfg)
    device = torch.device(cfg["project"]["device"] if torch.cuda.is_available() else "cpu")
    model.to(device)
    ckpt_path = args.ckpt or str(Path(cfg["paths"]["checkpoint_dir"]) / "best.pt")
    load_model_weights(ckpt_path, model)
    model.eval()

    output = generate_from_prompt(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f"Generated output: {output}")
    output_path = Path(args.output or Path(cfg["paths"]["log_dir"]) / "samples.txt")
    save_sample(
        output,
        output_path,
        prompt=prompt,
        ckpt_path=ckpt_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
