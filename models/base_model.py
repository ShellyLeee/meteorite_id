"""Base model abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseClassifier(nn.Module, ABC):
    """Base class for image classifiers that output logits."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate feature representations."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits."""
        raise NotImplementedError

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convert logits to class probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict_label(self, x: torch.Tensor) -> torch.Tensor:
        """Convert logits to predicted class labels."""
        probs = self.predict_proba(x)
        return probs.argmax(dim=1)
