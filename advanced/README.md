# advanced/ — 进阶任务

本目录包含课程进阶部分（40 分）的实现与说明。

## 子目录

| 目录 | 内容 |
|------|------|
| [dpo/](dpo/) | DPO 对照实验（无需奖励模型） |
| [lora_ablation/](lora_ablation/) | LoRA rank / target_modules 消融 |

## 与主代码关系

- 核心训练逻辑在 `src/train/dpo.py`，advanced 目录放实验脚本与 README
- 对比指标：安全率、wall-clock、显存峰值、调参难度

## 可选扩展（时间允许）

- RAG：BM25 + prompt 拼接
- 多轮对话：改造 Alpaca 为多轮格式
