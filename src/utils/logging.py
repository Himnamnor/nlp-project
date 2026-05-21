"""
logging.py — 训练日志（W&B / TensorBoard / stdout）

用途：
  - 统一 init_logger(cfg) 初始化
  - log_metrics(step, metrics) 写入 scalar
  - 可选：log_text / log_table 用于 SFT 样例、安全评测

TODO：
  - 实现 W&B 与 TensorBoard 双写或按 cfg 切换
  - 实现 log_generation_samples(path, samples)
"""

from __future__ import annotations

from typing import Any


class TrainingLogger:
    """Thin wrapper around W&B and/or TensorBoard."""

    def __init__(self, cfg: dict[str, Any], run_name: str | None = None) -> None:
        self.cfg = cfg.get("logging", {})
        self.run_name = run_name
        self._wandb = None
        self._writer = None
        # TODO: if self.cfg.get("use_wandb"): import wandb; wandb.init(...)
        # TODO: if self.cfg.get("use_tensorboard"): SummaryWriter(...)

    def log_metrics(self, step: int, metrics: dict[str, float], prefix: str = "") -> None:
        """Log scalar metrics at training step."""
        named = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
        # TODO: wandb.log(named, step=step)
        # TODO: writer.add_scalar(...)
        print(f"[step {step}] {named}")

    def finish(self) -> None:
        """Close loggers."""
        # TODO: wandb.finish(); writer.close()
        pass


def init_logger(cfg: dict[str, Any], run_name: str | None = None) -> TrainingLogger:
    return TrainingLogger(cfg, run_name=run_name)
