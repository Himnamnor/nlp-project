"""
sft_eval.py — SFT 指令跟随人工/LLM 评估

用途：
  - 固定 50 条指令（从 Alpaca 分层抽样）
  - 模型生成回答，保存 jsonl 供人工 0/1 打分
  - 可选 GPT-4o-mini LLM-judge 交叉验证
  - 目标准确率 > 60%

运行：
  python -m src.eval.sft_eval --config configs/sft.yaml --output logs/sft/eval_samples.jsonl

输出格式 (jsonl):
  {"instruction": "...", "reference": "...", "model_output": "...", "human_score": null}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset

from src.data.sft import format_alpaca
from src.eval.generate import generate_from_prompt
from src.model.llama import build_llama_from_config
from src.tokenizer import ProjectTokenizer
from src.utils.checkpoint import load_model_weights
from src.utils.config import add_config_args, parse_config_from_args

EVAL_SEED = 42
EVAL_N = 50


def _load_eval_examples(cfg: dict, num_samples: int, seed: int) -> list[dict[str, Any]]:
    """Load a reproducible subset from Alpaca-Cleaned."""
    ds = load_dataset(cfg["data"]["dataset"], split="train")
    if not isinstance(ds, HFDataset):
        raise TypeError(f"Expected HuggingFace Dataset, got {type(ds)}")

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    selected = indices[: min(num_samples, len(indices))]
    return [cast(dict[str, Any], ds[i]) | {"_idx": i} for i in selected]


def _clean_output(text: str) -> str:
    """Trim accidental continuation into the next Alpaca prompt."""
    stops = ["\n### Instruction:", "\n### Input:", "\n### Response:"]
    end = len(text)
    for marker in stops:
        pos = text.find(marker)
        if pos != -1:
            end = min(end, pos)
    return text[:end].strip()


def run_eval(
    cfg: dict,
    output_path: Path,
    ckpt_path: str,
    num_samples: int = EVAL_N,
    seed: int = EVAL_SEED,
    max_new_tokens: int = 96,
    temperature: float = 0.5,
    top_p: float = 0.8,
) -> None:
    """Generate model responses for eval set."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg["project"]["device"] if torch.cuda.is_available() else "cpu")
    tokenizer = ProjectTokenizer.load(cfg["paths"]["tokenizer_dir"])
    model = build_llama_from_config(cfg).to(device)
    load_model_weights(ckpt_path, model)
    model.eval()

    examples = _load_eval_examples(cfg, num_samples=num_samples, seed=seed)

    with open(output_path, "w", encoding="utf-8") as f:
        for rank, ex in enumerate(examples, start=1):
            instruction = str(ex["instruction"])
            input_text = str(ex.get("input") or "")
            reference = str(ex["output"])
            _, prompt = format_alpaca(instruction, input_text, "")
            output = generate_from_prompt(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            row = {
                "id": rank,
                "dataset_idx": ex["_idx"],
                "instruction": instruction,
                "input": input_text,
                "reference": reference,
                "model_output": _clean_output(output),
                "human_score": None,
                "notes": "",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{rank}/{len(examples)}] generated dataset_idx={ex['_idx']}")
    print(f"Wrote {len(examples)} SFT eval samples → {output_path}")


def aggregate_scores(jsonl_path: Path) -> float:
    """Compute accuracy from human_score field."""
    scores: list[float] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("human_score") is not None:
                scores.append(float(row["human_score"]))
    acc = sum(scores) / len(scores) if scores else 0.0
    print(f"Scored {len(scores)} samples: accuracy={acc:.2%} ({sum(scores):.1f}/{len(scores)})")
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT instruction-following eval")
    add_config_args(parser)
    parser.add_argument("--ckpt", type=str, default=None, help="SFT checkpoint to evaluate")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=EVAL_N)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--aggregate", type=str, default=None, help="Aggregate a scored jsonl file")
    args = parser.parse_args()
    cfg = parse_config_from_args(args)
    if args.aggregate:
        aggregate_scores(Path(args.aggregate))
        return

    out = Path(args.output or cfg["paths"].get("eval_samples", "logs/sft/eval_samples.jsonl"))
    ckpt_path = args.ckpt or str(Path(cfg["paths"]["checkpoint_dir"]) / "best.pt")
    run_eval(
        cfg,
        out,
        ckpt_path=ckpt_path,
        num_samples=args.num_samples,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
