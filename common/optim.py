"""Optimizer and scheduler factories."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn



def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """Build optimizer from config.

    Supported: Adam, AdamW, SGD.
    """
    name = str(cfg.get("name", "adamw")).lower()
    lr = float(cfg.get("lr", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer name: {name}")



def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict[str, Any]):
    """Build scheduler from config.

    Supported: cosine, step, none.
    """
    name = str(cfg.get("name", "none")).lower()

    if name in {"none", "null"}:
        return None

    if name == "cosine":
        t_max = int(cfg.get("t_max", 10))
        eta_min = float(cfg.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    if name == "step":
        step_size = int(cfg.get("step_size", 5))
        gamma = float(cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    raise ValueError(f"Unsupported scheduler name: {name}")
