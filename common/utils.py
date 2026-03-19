"""General utility helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml



def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {cfg_path} must be a mapping")
    return data



def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries without mutating inputs."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged



def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target



def get_device(device_str: str = "auto") -> torch.device:
    """Resolve torch device from config string."""
    device_str = device_str.lower()
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but CUDA is not available")
    return torch.device(device_str)



def save_config(cfg: dict[str, Any], output_dir: Path) -> None:
    """Save config to output directory for experiment tracking."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_backup_dir = output_dir / "configs"
    config_backup_dir.mkdir(parents=True, exist_ok=True)

    config_name = f"config_{timestamp}.yaml"
    config_path = config_backup_dir / config_name

    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return config_path
