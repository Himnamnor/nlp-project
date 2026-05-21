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
from pathlib import Path

from src.utils.config import add_config_args, parse_config_from_args

# TODO: 固定 50 条 eval 指令 id 或 seed，保证可复现
EVAL_SEED = 42
EVAL_N = 50


def run_eval(cfg: dict, output_path: Path) -> None:
    """Generate model responses for eval set."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # TODO: load sft model, sample 50 instructions, generate, write jsonl
    placeholder = {
        "instruction": "Explain gravity in simple terms.",
        "reference": "",
        "model_output": "",
        "human_score": None,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(placeholder, ensure_ascii=False) + "\n")
    print(f"SFT eval skeleton → {output_path} (TODO: fill {EVAL_N} samples)")


def aggregate_scores(jsonl_path: Path) -> float:
    """Compute accuracy from human_score field."""
    scores = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("human_score") is not None:
                scores.append(row["human_score"])
    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT instruction-following eval")
    add_config_args(parser)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    cfg = parse_config_from_args(args)
    out = Path(args.output or cfg["paths"].get("eval_samples", "logs/sft/eval_samples.jsonl"))
    run_eval(cfg, out)


if __name__ == "__main__":
    main()
