# DPO 对照实验

## 目的

在同一 PKU-SafeRLHF 偏好对上，对比 **PPO（需奖励模型）** 与 **DPO（单阶段）** 的：

- 安全回答率
- 训练 wall-clock
- 调参难度 / 稳定性

## 运行

```bash
python -m src.train.dpo --config configs/dpo.yaml
python -m src.eval.safety_eval --config configs/dpo.yaml --label dpo --ckpt checkpoints/dpo/best
```

## 报告要点

- DPO beta 敏感性（0.05 / 0.1 / 0.2）
- 与 PPO 的安全率对比表
- 分析：DPO 省略 reward model 的代价与收益

## TODO

- [ ] 完成 `src/train/dpo.py` TRL 接入
- [ ] 与 `logs/ppo/` 安全评测结果对比
- [ ] 写入 `report/report.md` 消融章节
