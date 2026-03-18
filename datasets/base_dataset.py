"""Base dataset abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


class BaseImageDataset(Dataset, ABC):
    """Abstract base class for image datasets."""

    def __init__(self, transform: Any = None) -> None:
        self.transform = transform

    @staticmethod
    def load_rgb_image(image_path: str | Path) -> Image.Image:
        """Load a PIL image and convert it to RGB with clear errors."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to open image: {path}") from exc

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError
