# 实验报告（NLP 课程项目）

231880135 陈祖希 · 从零训练大语言模型

> 训练完成后填写各章节；骨架保留 TODO 提示。

---

## 1. 引言

- 项目目标：预训练 → SFT → RLHF 三阶段
- 模型：Llama3 风格 mini（6L / d=512 / GQA）
- 硬件：RTX 3060 12GB（本地）/ 可选云端 4090

## 2. 模型架构

<!-- TODO: 参数量、模块表、与 GPT-2 差异 -->

| 组件 | 配置 |
|------|------|
| Layers | 6 |
| d_model | 512 |
| Heads / KV heads | 8 / 4 |
| FFN | SwiGLU ~1408 |
| Positional | RoPE θ=10000 |
| Norm | RMSNorm |

## 3. 预训练

### 3.1 数据与 Tokenizer

<!-- TODO: TinyStories 规模、BPE 词表 16k -->

### 3.2 训练配置

<!-- TODO: LR、batch、steps -->

### 3.3 结果

| 指标 | 目标 | 实际 |
|------|------|------|
| Val PPL | < 40 | TODO |

<!-- TODO: 插入 PPL 曲线图 logs/pretrain/ -->

### 3.4 生成样例

<!-- TODO: 粘贴 3-5 条故事 -->

## 4. 指令微调 (SFT)

### 4.1 数据

<!-- Alpaca-Cleaned ≥2k -->

### 4.2 冻结策略

<!-- 仅最后 2 层；讨论 tied embedding -->

### 4.3 结果

| 指标 | 目标 | 实际 |
|------|------|------|
| 50 条指令准确率 | > 60% | TODO |

## 5. RLHF

### 5.1 奖励模型

<!-- pairwise acc ≥ 65% -->

### 5.2 PPO

<!-- KL、reward 曲线 -->

### 5.3 安全评估

| 模型 | 安全率 |
|------|--------|
| SFT baseline | TODO |
| PPO + LoRA | TODO (>80%) |

## 6. 进阶任务

### 6.1 DPO vs PPO

<!-- TODO -->

### 6.2 LoRA 消融

<!-- TODO -->

## 7. 踩坑与思考

<!-- loss spike / OOM / KL 失控 / 奖励 hacking 等 -->

## 8. 参考文献

1. Radford et al. Language Models are Unsupervised Multitask Learners (GPT-2)
2. Touvron et al. LLaMA / Llama 2 / Llama 3
3. Ouyang et al. Training language models to follow instructions with human feedback (InstructGPT)
4. Schulman et al. Proximal Policy Optimization Algorithms
5. Rafailov et al. Direct Preference Optimization (DPO)
6. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models
7. Eldan & Li. TinyStories
8. Taori et al. Alpaca
9. PKU-Alignment. PKU-SafeRLHF
10. HuggingFace TRL documentation
