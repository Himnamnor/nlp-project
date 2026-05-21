"""
config.py — YAML 配置加载与合并

用途：
  - 从 configs/*.yaml 读取超参，转为 dict / OmegaConf
  - 支持 CLI 覆盖（如 --max_steps 2 用于 smoke test）
  - 统一 project root 路径解析（本地 Windows / 云端 Linux）

TODO：
  - 实现 load_config(path, overrides) -> dict
  - 实现 resolve_paths(cfg, project_root) 将相对路径转为绝对路径
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return Project/ root (parent of src/)."""
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Load config and optionally merge CLI overrides.

    Example overrides: {"train.max_steps": 2, "project.seed": 0}
    """
    cfg = load_yaml(config_path)
    root = get_project_root()

    # Resolve relative paths under project root
    if "paths" in cfg:
        for key, val in cfg["paths"].items():
            if isinstance(val, str) and not val.startswith(("/", "http")):
                cfg["paths"][key] = str(root / val)

    if overrides:
        _deep_update(cfg, overrides)

    cfg["_project_root"] = str(root)
    return cfg


def _deep_update(base: dict, updates: dict) -> None:
    """Merge flat dotted keys or nested dict into base."""
    for key, val in updates.items():
        if "." in key:
            parts = key.split(".")
            d = base
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = val
        elif isinstance(val, dict) and key in base and isinstance(base[key], dict):
            base[key].update(val)
        else:
            base[key] = val


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add --config and common override flags."""
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--max_steps", type=int, default=None, help="Override train.max_steps")
    parser.add_argument("--seed", type=int, default=None, help="Override project.seed")


def parse_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build config from argparse namespace."""
    overrides: dict[str, Any] = {}
    if args.max_steps is not None:
        overrides["train.max_steps"] = args.max_steps
    if args.seed is not None:
        overrides["project.seed"] = args.seed
    return load_config(args.config, overrides)
