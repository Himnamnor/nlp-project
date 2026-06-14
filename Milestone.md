# Milestone 231880135 陈祖希
## 项目概况
单人组队 231880135 陈祖希

选题：从0训练语言模型

模型：Llama3风格, 6层, d_model=512, head=8, 4KV heads, GQA, RMSNorm模型

数据：
- 预训练：TinyStories, 由于在Alpaca上表现不佳，用SmolLM-Corpus子集进行continued pretrain
- SFT: Alpaca-Cleaned
- RLHF: PKU-Safe-RLHF

平台：租用云端V100训练+SFT+RLHF，得到权重后本地RTX 3060 Laptop验证

## 当前进展
- 已完成基于pytorch的Llama3-like模型从0构建

- 成功进行预训练，在TinyStories上7000轮训练最优困惑度5.45，符合要求；continued-pretrain 3000轮最优困惑度27.09，符合要求。

- 成功在last 2 layers和全量参数上SFT

- 成功基于SFT的模型训练reward model，从0实现PPO并RLHF训练，并人工进行安全评估

- 进阶任务采用DPO+LORA和保守的PPO进行消融实验，目前已完成

## 初步成果
- pretrain最优困惑度5.45<40，continued-pretrain最优困惑度27.09<40
- SFT(last-2)在50条随机抽取的Alpaca上表现为满足要求的回答仅占6%(严格0-1评分)，宽松评分(0,0.5,1分层打分)仅13.8%。上述结果均为SmolLM continued-pretrain后结果，（实际上在TinyStories训练之后效果更差），估计是底座数据集信息太局限（儿童故事）导致的，而Alpaca SFT数据集上有许多世界知识和编码问题，超出底座范围了
- 接着SFT做PPO，大量无关回答，这跟SFT效果太差有关系，因此参考价值不大
- DPO也做了，但正如前面所说，参考价值不大

## 下一步计划
这个6层512d的模型感觉在TinyStories上的训练差不多到头了，得增强参数量，换更泛化的预训练数据集才行。等课程服务器下发薅一波学校羊毛再从头训吧（），尽量提升SFT和RLHF表现，不然以现在的模型开展任何进一步的工作都挺无厘头