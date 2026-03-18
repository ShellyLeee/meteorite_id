"""ResNet classifier implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

try:
    from meteorite_id.models.base_model import BaseClassifier
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    from models.base_model import BaseClassifier


class ResNetClassifier(BaseClassifier):
    """ResNet-based classifier supporting resnet18 and resnet50."""

    def __init__(self, model_name: str = "resnet18", num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__(num_classes=num_classes)
        self.model_name = model_name.lower()

        if self.model_name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = resnet18(weights=weights)
        elif self.model_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled feature vectors before final FC."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for logits."""
        return self.backbone(x)
