# 保守 PPO 消融

## 动机

主 PPO（`configs/ppo.yaml`）训练完成后出现：

- `kl_mean ≈ 4.6`，policy 明显偏离 SFT 分布
- 输出语义退化（大量重复词、无意义模板）
- safety eval 未出现稳定拒答句式

为了验证这并非 PPO 框架本身的 bug，而是超参选择导致的 reward hacking / policy drift，
设计一组只改若干稳定性相关超参的对照实验。

## 配置 diff（只列与主 PPO 的差异）

| 超参 | 主 PPO | 保守 PPO | 改动方向 |
|------|--------|----------|----------|
| `learning_rate`   | 1e-5 | 5e-6 | ↓ 更小步更新，policy drift 更慢 |
| `init_kl_coef`    | 0.1  | 0.3  | ↑ 更强 KL 约束 |
| `total_steps`     | 200  | 50   | ↓ 早停，避免后期退化 |
| `max_new_tokens`  | 96   | 64   | ↓ 短回答更可读 |
| `ppo_epochs`      | 4    | 2    | ↓ 减少对同一份 rollout 的反复利用 |

其余字段沿用主 PPO 配置。

## 运行

```bash
# Windows
.\scripts\run_ppo_conservative.ps1
# Linux / AutoDL
bash scripts/run_ppo_conservative.sh

# 评估
python -m src.eval.safety_eval --config configs/ppo_conservative.yaml \
    --ckpt checkpoints/ppo_conservative/best.pt --label ppo_conservative \
    --output logs/ppo_conservative/safety_eval_ppo_conservative.jsonl
```

输出落在 `checkpoints/ppo_conservative/` 与 `logs/ppo_conservative/`，不覆盖主 PPO 结果。

## 评估维度

记录到 `report/report.md` 第 6.2 节：

- `ppo/reward_mean` 训练曲线，是否单调上升或平稳
- `ppo/kl_mean` 是否始终 < 主 PPO（应当显著降低）
- `ppo/clipfrac` 是否更低（更小的 LR 期待更小的 clipfrac）
- safety eval 输出是否更可读、是否出现拒答关键词

## 预期结论的两种走向

1. **保守 PPO 优于主 PPO**：写"加强 KL 约束 + 缩短生成是当前小模型上 PPO 稳定性的关键"。
2. **保守 PPO 仍退化**：写"对 26M / 弱 SFT / 弱 reward 的设置，PPO 整体方案受限于底座能力，需提升模型规模或预训练 token"。

两种结论都对报告有意义，不需要"训出好结果"才算成功。
