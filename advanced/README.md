# advanced/ — 进阶任务

本目录组织课程进阶部分（40 分=创新性 20 分+性能 10 分+报告质量 10 分）的实验。
所有进阶实验复用 `src/` 主代码（无需重复实现 Llama / Tokenizer），通过不同配置触发。

## 实验矩阵

| 实验 | 配置 | 入口 | 目的 |
|------|------|------|------|
| **DPO 对照（主线）** | `configs/dpo.yaml` | `src/train/dpo.py` | 验证不依赖 reward model 的单阶段偏好优化在小模型上是否更稳定 |
| **保守 PPO 消融** | `configs/ppo_conservative.yaml` | `src/train/ppo.py` | 验证更强 KL 约束 + 更小 LR + 短生成能否缓解主 PPO 的语言退化 |
| **LoRA 扩展设计** | — | — | 设计文档：对 100M+ 模型才是显存必要项；本项目作为方法消融 |

## 一键脚本

```bash
# Windows
.\scripts\run_dpo.ps1
.\scripts\run_ppo_conservative.ps1

# Linux / AutoDL
bash scripts/run_dpo.sh
bash scripts/run_ppo_conservative.sh

# 跑全套 + 安全评测
bash scripts/run_advanced.sh
```

## 子目录

| 目录 | 内容 |
|------|------|
| [dpo/](dpo/) | DPO 对照实验说明、调参建议 |
| [ppo_ablation/](ppo_ablation/) | 保守 PPO 消融说明 |
| [lora_ablation/](lora_ablation/) | LoRA 设计文档（不实做） |

## 评估方法

所有进阶模型最终都跑 `src/eval/safety_eval.py`，与第 5 节 SFT baseline / 主 PPO 同一套
prompts（40 条，25 PKU-SafeRLHF unsafe + 15 经典越狱）和同一份启发式预打分。
对照表写入 `report/report.md` 第 6 节。
