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

### 7.1 V100 + fp16 预训练发散

**问题情形。** 预训练迁移到 AutoDL V100 后，由于 V100 不支持 bf16，配置改为 `float16`。初始训练在 step 1000 左右验证集 PPL 已经降到 38 附近，但继续训练后出现 loss spike：训练 loss 从稳定下降状态反弹，后续接近 9，表现为数值不稳定和疑似发散。

**思考过程。** 3060/4090 上可以用 bf16，动态范围较大，不需要 loss scaling；但 V100 只能用 fp16。fp16 动态范围更窄，在大词表交叉熵和 AdamW 更新中容易出现梯度下溢或溢出。原训练循环只用了 autocast，没有使用 GradScaler，因此在 LR warmup 到峰值附近更容易触发不稳定。同时，早期日志中 train loss 的统计方式多乘了 `grad_accum`，导致观察到的训练 loss 偏大，容易和真实发散混淆。

**解决方法。**

- 在 `src/train/pretrain.py` 中加入 `torch.cuda.amp.GradScaler`，只在 `dtype=float16` 且 CUDA 可用时启用；bf16 训练保持不启用。
- 按 AMP 标准顺序执行：`scaler.scale(loss).backward()` → `scaler.unscale_(optimizer)` → `clip_grad_norm_` → `scaler.step(optimizer)` → `scaler.update()`。
- 修正训练 loss 日志：按 optimizer step 聚合真实 CE loss，避免再额外乘 `grad_accum`。
- 将峰值 LR 从 `1.0e-4` 降到 `6.0e-5`，`min_lr` 降到 `1.0e-5`，并把 `grad_clip` 恢复到 `1.0`。

**最终结果。** 修复后训练稳定推进，验证集 PPL 持续下降；最终在 step 7000 得到 `val/ppl = 5.45`，显著优于课程目标 `PPL < 40`。因此提前终止预训练，保留 `checkpoints/pretrain/best.pt` 作为后续 SFT 初始化权重。

### 7.2 ByteLevel BPE 解码出现 `Ġ`

**问题情形。** 使用预训练 checkpoint 生成样例时，模型输出中出现大量 `Ġ` 字符，例如：

```text
I 'm ĠTony ĠSt ark , Ġit 's Ġa Ġspecial Ġtreat Ġfor Ġyou !"
```

该输出并非完全乱码，模型能延续 prompt，但空格被显示成 ByteLevel BPE 的内部标记，不能直接作为报告中的生成样例。

**思考过程。** tokenizer 训练时使用了 `ByteLevel(add_prefix_space=False)` 作为 pre-tokenizer。ByteLevel BPE 会用类似 `Ġ` 的符号表示词前空格；如果保存的 tokenizer 没有配置对应的 ByteLevel decoder，`decode()` 时就不会把这些内部符号还原成普通空格。该问题属于 tokenizer 序列化配置缺失，不是模型权重或 PPL 指标异常。由于只缺 decoder，词表和 token id 映射不需要改变，不能重训 tokenizer，否则会破坏与已训练 `best.pt` 的兼容性。

**解决方法。**

- 在 `src/tokenizer/train_bpe.py` 中为未来训练出的 tokenizer 增加：

```python
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

tokenizer.decoder = ByteLevelDecoder()
```

- 对现有 `tokenizer/tokenizer.json` 进行原地修复：加载已有 tokenizer，补上 `ByteLevelDecoder()`，再保存回同一路径。该操作只改变 decode 规则，不改变词表和 token id，因此与现有预训练 checkpoint 兼容。

**最终结果。** 修复后，`ProjectTokenizer.decode()` 可以把 `Ġ` 正常还原为空格。后续生成样例可重新运行 `python -m src.eval.generate --config configs/pretrain.yaml --prompt "Once upon a time"`，将无 `Ġ` 的文本放入报告生成样例部分。

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
