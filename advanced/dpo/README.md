# DPO 对照实验

## 目的

在同一份 PKU-SafeRLHF 偏好对上对照 PPO（reward + 在线 rollout）和 DPO（单阶段，直接在偏好对上优化 policy），考察：

- 安全回答率
- 训练 wall-clock 与显存峰值
- 收敛稳定性
- 调参难度

## 实现说明（要点）

- 完全手写 DPO，不依赖 TRL：`src/train/dpo.py`
- 数据：`src/data/rlhf.py` 的 `DPOPairDataset`，每条样本包含 `(prompt+chosen, prompt+rejected)` 的 ids 与 response_mask
- 损失：

  ```text
  L = -log sigmoid( beta * ( log_ratio_chosen - log_ratio_rejected ) )
  log_ratio = sum( log p(y_t | x, y_<t) over response tokens )
  ```

- policy 从 SFT (`checkpoints/sft_smol_full/best.pt`) 初始化，可训练
- ref model：policy 的深拷贝，冻结，eval()，不进入 optimizer
- AMP：fp16 + GradScaler（与 SFT / Reward / PPO 一致）
- 保存为 `LlamaForCausalLM` 兼容格式，`safety_eval.py` 可直接加载

## 运行

```bash
# Windows
.\scripts\run_dpo.ps1
# Linux / AutoDL
bash scripts/run_dpo.sh

# 评估
python -m src.eval.safety_eval --config configs/dpo.yaml \
    --ckpt checkpoints/dpo/best.pt --label dpo \
    --output logs/dpo/safety_eval_dpo.jsonl
```

## 调参建议（按重要性）

| 超参 | 建议初值 | 备注 |
|------|----------|------|
| `dpo.beta` | 0.1 | 越大越靠近 ref；不稳定就调到 0.05 |
| `train.learning_rate` | 5e-6 | DPO 比 PPO 更敏感，不要超 1e-5 |
| `train.batch_size` | 4 | 每步 4 次前向（policy×2 + ref×2），显存 ≈ 2×SFT |
| `data.max_length` | 512 | 与 reward 保持一致；过长易 OOM |

## 报告对照表（待训练后填）

| 模型 | 训练时间 | KL/退化迹象 | 安全率 | 是否需 reward model |
|------|----------|-------------|--------|----------------------|
| SFT baseline | — | — | TODO | — |
| 主 PPO | 0.08 h | KL 后期 4.6, 退化 | 未达目标 | 是 |
| 保守 PPO | TODO | TODO | TODO | 是 |
| DPO | TODO | TODO | TODO | **否** |

最终结论填入 `report/report.md` 第 6.1 节。
