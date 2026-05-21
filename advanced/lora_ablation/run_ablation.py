"""
run_ablation.py — LoRA 超参自动 sweep（advanced/lora_ablation）

TODO：
  - 遍历 r ∈ {4,8,16} × target_modules 组合
  - 调用 src.train.ppo 或 dpo
  - 汇总 safety_rate, peak_mem, time → logs/ablation/results.csv
"""

from __future__ import annotations

ABLATION_GRID = [
    {"r": 4, "target_modules": ["q_proj", "v_proj"]},
    {"r": 8, "target_modules": ["q_proj", "v_proj"]},
    {"r": 16, "target_modules": ["q_proj", "v_proj"]},
    {"r": 8, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
]


def main() -> None:
    print("LoRA ablation skeleton. Grid:", ABLATION_GRID)
    # TODO: implement sweep


if __name__ == "__main__":
    main()
