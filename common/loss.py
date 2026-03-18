"""Loss definitions and factories."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    """Base interface for custom losses."""

    @abstractmethod
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute scalar loss from logits and targets."""
        raise NotImplementedError


class CrossEntropyLossWrapper(BaseLoss):
    """Standard cross entropy loss wrapper."""

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


class LabelSmoothingCrossEntropy(BaseLoss):
    """Alias wrapper for readability in configs."""

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)



def build_loss(cfg: dict[str, Any] | None) -> BaseLoss:
    """Build loss from config dictionary."""
    cfg = cfg or {}
    name = str(cfg.get("name", "cross_entropy")).lower()

    if name in {"cross_entropy", "ce"}:
        return CrossEntropyLossWrapper(label_smoothing=float(cfg.get("label_smoothing", 0.0)))

    if name in {"label_smoothing", "label_smoothing_ce"}:
        return LabelSmoothingCrossEntropy(smoothing=float(cfg.get("label_smoothing", 0.1)))

    raise ValueError(f"Unsupported loss name: {name}")
