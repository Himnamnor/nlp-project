"""
rlhf.py — PKU-SafeRLHF 偏好对 / PPO prompt 采样

数据：PKU-Alignment/PKU-SafeRLHF
  字段：prompt / response_0 / response_1 / safer_response_id / is_response_0_safe / is_response_1_safe / better_response_id

用途：
  - reward 训练：把 (prompt, chosen, rejected) 拼接成完整序列后做 pairwise loss
  - PPO 训练：只对 prompt 做生成，rollout 内自己拼回 response
  - safety_eval：从带 unsafe 标签的样本里抽 prompt

prompt 模板复用 SFT 阶段的 Alpaca：
    ### Instruction:\n{prompt}\n\n### Input:\n\n\n### Response:\n
这样 PPO/SFT 共享同一种格式，policy 不需要额外学新格式。
"""

from __future__ import annotations

import random
from typing import Any, cast

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.sft import format_alpaca
from src.tokenizer import ProjectTokenizer


def format_rlhf_prompt(prompt: str) -> str:
    """Wrap a raw question in the Alpaca prefix (empty Input / Response)."""
    _, prefix = format_alpaca(prompt, "", "")
    return prefix


def load_safe_rlhf(dataset_name: str, split: str = "train") -> HFDataset:
    ds = load_dataset(dataset_name, split=split)
    if not isinstance(ds, HFDataset):
        raise TypeError(f"Expected HuggingFace Dataset, got {type(ds)}")
    return ds


def build_preference_pair(example: dict[str, Any]) -> dict[str, str]:
    """Convert one SafeRLHF row to {prompt, chosen, rejected} by safer_response_id."""
    prompt = str(example.get("prompt") or "")
    r0 = str(example.get("response_0") or "")
    r1 = str(example.get("response_1") or "")
    safer = int(example.get("safer_response_id", 0) or 0)
    if safer == 0:
        chosen, rejected = r0, r1
    else:
        chosen, rejected = r1, r0
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


# ---------------------------------------------------------------------------
# Reward (pairwise) dataset
# ---------------------------------------------------------------------------


class PreferencePairDataset(Dataset):
    """Tokenized (prompt+chosen) / (prompt+rejected) pairs for reward training."""

    def __init__(
        self,
        tokenizer: ProjectTokenizer,
        dataset_name: str,
        split: str = "train",
        max_samples: int | None = None,
        max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        ds = load_safe_rlhf(dataset_name, split=split)
        if max_samples is not None and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def _encode(self, prompt_text: str, response: str) -> list[int]:
        # 与 SFT/pretrain 一致：不加 BOS，仅末尾 EOS
        full = prompt_text + response
        ids = self.tokenizer.encode(full, add_bos=False, add_eos=True)
        ids = ids[: self.max_length]
        if len(ids) < 2:
            ids = ids + [self.tokenizer.eos_id or 0]
        return ids

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = cast(dict[str, Any], self.ds[idx])
        pair = build_preference_pair(ex)
        prompt_text = format_rlhf_prompt(pair["prompt"])
        chosen_ids = self._encode(prompt_text, pair["chosen"])
        rejected_ids = self._encode(prompt_text, pair["rejected"])
        return {
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
        }


def collate_preference_batch(
    batch: list[dict[str, torch.Tensor]], pad_id: int = 0
) -> dict[str, torch.Tensor]:
    """Right-pad chosen/rejected to a common length within the batch."""
    max_len = max(
        max(b["chosen_ids"].size(0) for b in batch),
        max(b["rejected_ids"].size(0) for b in batch),
    )

    def pad(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad_len = max_len - t.size(0)
        padded = torch.cat([t, torch.full((pad_len,), pad_id, dtype=torch.long)])
        mask = torch.cat(
            [
                torch.ones(t.size(0), dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ]
        )
        return padded, mask

    chosen_ids, chosen_mask, rejected_ids, rejected_mask = [], [], [], []
    for b in batch:
        cp, cm = pad(b["chosen_ids"])
        rp, rm = pad(b["rejected_ids"])
        chosen_ids.append(cp)
        chosen_mask.append(cm)
        rejected_ids.append(rp)
        rejected_mask.append(rm)
    return {
        "chosen_ids": torch.stack(chosen_ids),
        "chosen_mask": torch.stack(chosen_mask),
        "rejected_ids": torch.stack(rejected_ids),
        "rejected_mask": torch.stack(rejected_mask),
    }


def build_reward_dataloaders(
    cfg: dict, tokenizer: ProjectTokenizer
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for reward training."""
    data_cfg = cfg["data"]
    full = PreferencePairDataset(
        tokenizer=tokenizer,
        dataset_name=data_cfg["dataset"],
        split=data_cfg.get("split", "train"),
        max_samples=data_cfg.get("max_samples"),
        max_length=data_cfg.get("max_length", 512),
    )
    val_ratio = float(data_cfg.get("val_ratio", 0.05))
    n_val = max(1, int(len(full) * val_ratio))
    n_train = len(full) - n_val
    seed = int(cfg["project"]["seed"])
    train_ds, val_ds = random_split(
        full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    pad_id = tokenizer.pad_id if tokenizer.pad_id is not None else 0
    collate = lambda batch: collate_preference_batch(batch, pad_id=pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=int(data_cfg.get("num_workers", 0)),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Prompt sampling for PPO / safety_eval
# ---------------------------------------------------------------------------


def sample_prompts(
    cfg: dict,
    n: int,
    unsafe_only: bool = False,
    seed: int = 42,
) -> list[str]:
    """Sample unique prompts from PKU-SafeRLHF.

    If unsafe_only is True, return only prompts that elicited at least one unsafe response.
    Otherwise mix unsafe / safe in the ratio dictated by cfg["data"]["unsafe_ratio"].
    """
    data_cfg = cfg["data"]
    ds = load_safe_rlhf(data_cfg["dataset"], split=data_cfg.get("split", "train"))
    rng = random.Random(seed)

    n_total = len(ds)
    indices = list(range(n_total))
    rng.shuffle(indices)

    unsafe_ratio = float(data_cfg.get("unsafe_ratio", 0.7))
    n_unsafe_target = n if unsafe_only else int(round(n * unsafe_ratio))
    n_safe_target = 0 if unsafe_only else n - n_unsafe_target

    prompts: list[str] = []
    seen: set[str] = set()
    n_unsafe = n_safe = 0
    for i in indices:
        ex = cast(dict[str, Any], ds[i])
        prompt = str(ex.get("prompt") or "")
        if not prompt or prompt in seen:
            continue
        is_unsafe = (not bool(ex.get("is_response_0_safe", True))) or (
            not bool(ex.get("is_response_1_safe", True))
        )
        if is_unsafe and n_unsafe < n_unsafe_target:
            prompts.append(prompt)
            seen.add(prompt)
            n_unsafe += 1
        elif (not is_unsafe) and (not unsafe_only) and n_safe < n_safe_target:
            prompts.append(prompt)
            seen.add(prompt)
            n_safe += 1
        if len(prompts) >= n:
            break

    if len(prompts) < n:
        # 兜底：unsafe 不够时用剩余 prompt 补齐
        for i in indices:
            if len(prompts) >= n:
                break
            ex = cast(dict[str, Any], ds[i])
            prompt = str(ex.get("prompt") or "")
            if not prompt or prompt in seen:
                continue
            prompts.append(prompt)
            seen.add(prompt)
    return prompts


class PromptDataset(Dataset):
    """Prompts only, tokenized to the SFT-style Alpaca prefix. Used by PPO rollouts."""

    def __init__(
        self,
        tokenizer: ProjectTokenizer,
        prompts: list[str],
        max_prompt_length: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_prompt_length = max_prompt_length

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        prompt = self.prompts[idx]
        prompt_text = format_rlhf_prompt(prompt)
        # 与 SFT/pretrain 一致：不加 BOS，也不加 EOS（让 policy 继续生成）
        ids = self.tokenizer.encode(prompt_text, add_bos=False, add_eos=False)
        # 保留末尾（含 `### Response:\n`），防止 prompt 过长被前向截断
        ids = ids[-self.max_prompt_length :]
        return {
            "prompt_text": prompt,
            "prompt_formatted": prompt_text,
            "input_ids": torch.tensor(ids, dtype=torch.long),
        }


def build_prompt_dataset(cfg: dict, tokenizer: ProjectTokenizer) -> PromptDataset:
    data_cfg = cfg["data"]
    n = int(data_cfg.get("num_prompts", 500))
    prompts = sample_prompts(cfg, n, unsafe_only=False, seed=int(cfg["project"]["seed"]))
    return PromptDataset(
        tokenizer,
        prompts,
        max_prompt_length=int(data_cfg.get("max_prompt_length", 256)),
    )


# ---------------------------------------------------------------------------
# DPO dataset (preference pairs + response-only token mask)
# ---------------------------------------------------------------------------


class DPOPairDataset(Dataset):
    """Preference pairs with explicit response masks for DPO.

    Each item gives (prompt+chosen) and (prompt+rejected) token ids, plus a binary mask
    marking which tokens belong to the response (i.e. count toward the DPO log-prob sum).
    The mask follows the convention: response_mask[t] = 1 iff input_ids[t] is a response token.
    During training we only sum log-probs over positions where response_mask shifted by 1 is on.
    """

    def __init__(
        self,
        tokenizer: ProjectTokenizer,
        dataset_name: str,
        split: str = "train",
        max_samples: int | None = None,
        max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        ds = load_safe_rlhf(dataset_name, split=split)
        if max_samples is not None and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def _encode_pair(
        self, prompt_text: str, response: str
    ) -> tuple[list[int], list[int]]:
        """Return (full_ids, response_mask) for prompt+response."""
        prompt_ids = self.tokenizer.encode(prompt_text, add_bos=False, add_eos=False)
        full_ids = self.tokenizer.encode(prompt_text + response, add_bos=False, add_eos=True)
        full_ids = full_ids[: self.max_length]
        # 至少保留 1 个 response 位置，否则 DPO loss 没有信号
        prompt_len = min(len(prompt_ids), max(self.max_length - 1, 1))
        if len(full_ids) < prompt_len + 1:
            full_ids = full_ids + [self.tokenizer.eos_id or 0]
        response_mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)
        return full_ids, response_mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = cast(dict[str, Any], self.ds[idx])
        pair = build_preference_pair(ex)
        prompt_text = format_rlhf_prompt(pair["prompt"])
        chosen_ids, chosen_resp_mask = self._encode_pair(prompt_text, pair["chosen"])
        rejected_ids, rejected_resp_mask = self._encode_pair(prompt_text, pair["rejected"])
        return {
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "chosen_response_mask": torch.tensor(chosen_resp_mask, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "rejected_response_mask": torch.tensor(rejected_resp_mask, dtype=torch.long),
        }


def collate_dpo_batch(
    batch: list[dict[str, torch.Tensor]], pad_id: int = 0
) -> dict[str, torch.Tensor]:
    """Right-pad both branches separately; keep attention masks + response masks aligned."""
    max_len = max(
        max(b["chosen_ids"].size(0) for b in batch),
        max(b["rejected_ids"].size(0) for b in batch),
    )

    def pad_branch(
        ids_list: list[torch.Tensor], resp_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids_out, attn_out, resp_out = [], [], []
        for ids, resp in zip(ids_list, resp_list):
            pad_len = max_len - ids.size(0)
            pads = torch.full((pad_len,), pad_id, dtype=torch.long)
            zeros = torch.zeros(pad_len, dtype=torch.long)
            ids_out.append(torch.cat([ids, pads]))
            attn_out.append(
                torch.cat([torch.ones(ids.size(0), dtype=torch.long), zeros])
            )
            resp_out.append(torch.cat([resp, zeros]))
        return torch.stack(ids_out), torch.stack(attn_out), torch.stack(resp_out)

    chosen_ids, chosen_attn, chosen_resp = pad_branch(
        [b["chosen_ids"] for b in batch],
        [b["chosen_response_mask"] for b in batch],
    )
    rejected_ids, rejected_attn, rejected_resp = pad_branch(
        [b["rejected_ids"] for b in batch],
        [b["rejected_response_mask"] for b in batch],
    )
    return {
        "chosen_ids": chosen_ids,
        "chosen_attn": chosen_attn,
        "chosen_response_mask": chosen_resp,
        "rejected_ids": rejected_ids,
        "rejected_attn": rejected_attn,
        "rejected_response_mask": rejected_resp,
    }


def build_dpo_dataloaders(
    cfg: dict, tokenizer: ProjectTokenizer
) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    full = DPOPairDataset(
        tokenizer=tokenizer,
        dataset_name=data_cfg["dataset"],
        split=data_cfg.get("split", "train"),
        max_samples=data_cfg.get("max_samples"),
        max_length=int(data_cfg.get("max_length", 512)),
    )
    val_ratio = float(data_cfg.get("val_ratio", 0.05))
    n_val = max(1, int(len(full) * val_ratio))
    n_train = len(full) - n_val
    seed = int(cfg["project"]["seed"])
    train_ds, val_ds = random_split(
        full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    pad_id = tokenizer.pad_id if tokenizer.pad_id is not None else 0
    collate = lambda batch: collate_dpo_batch(batch, pad_id=pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=int(data_cfg.get("num_workers", 0)),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    return train_loader, val_loader
