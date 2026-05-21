# 从零训练大语言模型（NLP 课程项目）

231880135 陈祖希 · Llama3 风格小模型 · 预训练 → SFT → RLHF

## 项目结构

```
Project/
├── configs/          # 各阶段 YAML 超参
├── src/
│   ├── model/        # Llama3 手写 + HF 包装 + Reward head
│   ├── tokenizer/    # BPE 训练与加载
│   ├── data/         # TinyStories / Alpaca / SafeRLHF 数据管线
│   ├── train/        # pretrain / sft / reward / ppo / dpo
│   ├── eval/         # PPL、生成、SFT 准确率、安全率
│   └── utils/        # config、日志、checkpoint
├── scripts/          # 本地 & 云端一键脚本
├── advanced/         # DPO 对照、LoRA 消融（进阶任务）
├── tests/            # 单元测试
├── report/           # 实验报告
├── checkpoints/      # 权重（gitignored）
└── logs/             # W&B / TensorBoard（gitignored）
```

## 快速开始

### 1. 环境

```powershell
cd "大三下/NLP/Project"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 预训练流水线

```powershell
# 训练 BPE tokenizer
python -m src.tokenizer.train_bpe --config configs/pretrain.yaml

# 打包 TinyStories → .bin
python -m src.data.pretrain --config configs/pretrain.yaml

# smoke test（1 step）
python -m src.train.pretrain --config configs/pretrain.yaml --max_steps 2

# 正式预训练
python -m src.train.pretrain --config configs/pretrain.yaml
```

### 3. SFT → RLHF

见 [PLAN.md](PLAN.md) 各阶段说明；云端部署见 [scripts/setup_cloud.sh](scripts/setup_cloud.sh)。

## 评估目标

| 阶段   | 指标              | 目标    |
|--------|-------------------|---------|
| 预训练 | 验证集 PPL        | < 40    |
| SFT    | 50 条指令准确率   | > 60%   |
| RLHF   | 安全回答率        | > 80%   |

## 参考文献

见 [report/report.md](report/report.md)。
