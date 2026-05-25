# LoRA 消融实验（设计文档，本项目不实做）

## 为什么本项目不实做

原 PLAN 把 LoRA 作为 RLHF 显存优化的核心路径，但当前实际状况是：

- 模型仅约 26M 参数，policy + ref + reward 三模型同驻 V100 16GB 仍有大量余量
- 主 PPO 实测峰值显存远低于 10GB，全参更新完全够用
- LoRA 在如此小的模型上既不省显存也不省时间，反而增加实现 / 校验成本

因此本项目把 LoRA 调整为「方法对照设计」，不在 26M 模型上跑数据型消融，避免堆叠对结论无增益的实验。

## 如果在 100M+ 模型上做（推荐扩展方向）

设计如下消融，记录显存与安全率：

| 实验 | 可训练参数 | 期望显存 | 期望稳定性 |
|------|------------|----------|------------|
| Full PPO | 100% | 高 | 易退化 |
| Last-2 layer PPO | ~25% | 中 | 中 |
| LoRA r=8, q/v | <1% | 低 | 高 |
| LoRA r=16, q/k/v/o | ~1% | 低 | 高 |
| LoRA r=8, q/k/v/o + mlp | ~2% | 低 | 中-高 |

实现可基于 PEFT：

```python
from peft import LoraConfig, get_peft_model
lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","v_proj"], bias="none", task_type="CAUSAL_LM",
)
policy = get_peft_model(policy, lora_cfg)
```

并把 `src/train/ppo.py` 的 `policy.parameters()` 替换为 `filter(lambda p: p.requires_grad, policy.parameters())`。

## 已保留的占位脚本

`run_ablation.py` 保留为扫参 skeleton（4 组配置），实际运行需要先完成 LoRA 接入。

## 结论（写入 report 6.3）

- 本项目规模下 LoRA 非必需；
- 一旦扩展到 100M+，LoRA 是 RLHF 阶段在 24GB 卡上能否同时容纳 policy/ref/reward 的关键；
- 因此把 LoRA 作为"未来工作 / 扩展方案"列在报告中，配合"200M 模型训练成本估算"一节。
