"""Logging utilities."""

from __future__ import annotations

import logging
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard is optional
    SummaryWriter = None  # type: ignore[assignment]



def build_logger(name: str, log_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Create a logger with stream and file handlers."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger



def build_tensorboard_writer(log_dir: str | Path):
    """Create a SummaryWriter if tensorboard is available, else return None."""
    if SummaryWriter is None:
        return None
    return SummaryWriter(log_dir=str(log_dir))
