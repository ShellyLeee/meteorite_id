"""Lightweight placeholder for ViT classifier."""

from __future__ import annotations

import torch

try:
    from meteorite_id.models.base_model import BaseClassifier
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    from models.base_model import BaseClassifier


class ViTClassifier(BaseClassifier):
    """Placeholder class to keep imports stable.

    Selection of ViT in model factory intentionally raises NotImplementedError.
    """

    def __init__(self, model_name: str = "vit_base_patch16_224", num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__(num_classes=num_classes)
        self.model_name = model_name
        self.pretrained = pretrained

    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("ViT feature extraction is not implemented in this baseline")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("ViT forward pass is not implemented in this baseline")
