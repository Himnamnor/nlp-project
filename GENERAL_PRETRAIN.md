# General Pretraining Plan

This branch adds a from-scratch general pretraining path intended for a single RTX 5090.

## Default Choice

Use `configs/pretrain_general.yaml` first.

- Dataset mix: `HuggingFaceTB/smollm-corpus`
  - `cosmopedia-v2`: 45%
  - `fineweb-edu-dedup`: 45%
  - `python-edu`: 10%
- Model: 10 layers, `d_model=768`, 12 attention heads, 4 KV heads, 32k vocab, context 1024.
- Size: roughly 90M parameters.
- First run budget: 300M packed tokens, 6000 optimizer steps.

Why this route: the current 26M TinyStories model learns language form but lacks general world/code/instruction coverage. SmolLM-Corpus is designed for small language models and gives a better match for later Alpaca SFT and SafeRLHF than pure children's stories.

`configs/pretrain_general_fineweb.yaml` is the broader FineWeb-Edu 100BT-shuffled backup. Use it if the SmolLM mix still feels too synthetic or narrow.

## 5090 Setup

```bash
git switch general-pretrain
bash scripts/setup_cloud.sh
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

If bf16 is not supported by the installed PyTorch/CUDA stack, change `project.dtype` in the general configs from `bfloat16` to `float16`.

## Smoke Test

Run this before spending real GPU time:

```bash
BPE_MAX_SAMPLES=5000 DATA_MAX_TOKENS=2000000 \
  bash scripts/run_general_pretrain.sh configs/pretrain_general.yaml --max_steps 20
```

Expected: tokenizer is created under `tokenizer_general/`, token bins under `data/processed_general/`, and the 20-step training loop finishes without non-finite loss.

## Full Pretraining

```bash
bash scripts/run_general_pretrain.sh configs/pretrain_general.yaml
```

If tokenizer and `.bin` files already exist, resume only training:

```bash
RUN_TOKENIZER=0 RUN_DATA=0 \
  bash scripts/run_general_pretrain.sh configs/pretrain_general.yaml
```

If data exists but tokenizer should be kept:

```bash
RUN_TOKENIZER=0 \
  bash scripts/run_general_pretrain.sh configs/pretrain_general.yaml
```

## Continued Pretraining

Use this when `/root/autodl-tmp/Project/checkpoints/pretrain_general/best.pt`
and `tokenizer_general/` already exist, but the previous continued run produced
no usable artifacts.

If only `best.slim.pt` is available, it can still initialize training because it
keeps `model_state_dict`; place it at
`/root/autodl-tmp/Project/checkpoints/pretrain_general/best.pt` or change
`paths.init_from` in `configs/pretrain_general_continued.yaml`.

First prepare fresh continued data and train:

```bash
mkdir -p /root/autodl-tmp/Project/logs/pretrain_general_continued
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_TOKENIZER=0 DATA_MAX_TOKENS=300000000 \
  bash scripts/run_general_pretrain.sh configs/pretrain_general_continued.yaml \
  2>&1 | tee /root/autodl-tmp/Project/logs/pretrain_general_continued/train.log
```

The continued config writes new data, checkpoints, logs, and samples under
`/root/autodl-tmp/Project/` to avoid filling the system disk.

If `/root/autodl-tmp/Project/data/processed_general_continued/train.bin` and
`val.bin` already exist, restart only the model training:

```bash
mkdir -p /root/autodl-tmp/Project/logs/pretrain_general_continued
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_TOKENIZER=0 RUN_DATA=0 \
  bash scripts/run_general_pretrain.sh configs/pretrain_general_continued.yaml \
  2>&1 | tee -a /root/autodl-tmp/Project/logs/pretrain_general_continued/train.log
```

The continued config saves under
`/root/autodl-tmp/Project/checkpoints/pretrain_general_continued/` every 250
optimizer steps, so an interrupted run should still leave recent `step_*.pt`
files and a `best.pt` once the first eval has completed.

## Generate Pretrain Samples

```bash
python -m src.eval.generate \
  --config configs/pretrain_general.yaml \
  --ckpt checkpoints/pretrain_general/best.pt \
  --prompt "Explain why the moon changes shape in the night sky." \
  --output logs/pretrain_general/samples.txt
```

## SFT and RLHF

After `checkpoints/pretrain_general/best.pt` exists:

```bash
bash scripts/run_general_alignment.sh
```

Or run step by step:

```bash
python -m src.train.sft --config configs/sft_general.yaml
python -m src.eval.sft_eval --config configs/sft_general.yaml \
  --ckpt checkpoints/sft_general_full/best.pt \
  --output logs/sft_general_full/eval_samples.jsonl

python -m src.train.reward --config configs/reward_general.yaml
python -m src.train.ppo --config configs/ppo_general.yaml
python -m src.train.dpo --config configs/dpo_general.yaml
```

## Success Criteria

Track these against the old TinyStories/Smol continued-pretrain run:

- Pretrain validation PPL is stable and decreasing.
- Samples can answer general explanatory prompts without collapsing into TinyStories style.
- `logs/sft_general_full/eval_samples.jsonl` scores materially better than the old 6% strict SFT score.
- Reward pairwise accuracy moves above the old best `0.6100`.
- PPO/DPO safety outputs become meaningful refusals or safe alternatives, not semantic drift.
