"""ResNet classifier implementation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

try:
    from meteorite_id.models.base_model import BaseClassifier
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    from models.base_model import BaseClassifier


class ResNetClassifier(BaseClassifier):
    """ResNet-based classifier supporting resnet18 and resnet50."""

    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = False,
        pretrained_path: str | None = None,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self.model_name = model_name.lower()

        # Step 1: always build model with weights=None first
        if self.model_name == "resnet18":
            self.backbone = resnet18(weights=None)
        elif self.model_name == "resnet50":
            self.backbone = resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")

        # Step 2: load local pretrained weights if provided
        if pretrained_path:
            ckpt_path = Path(pretrained_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

            state_dict = torch.load(ckpt_path, map_location="cpu")

            # Some checkpoints may store weights under "state_dict"
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Remove classifier head to avoid mismatch with num_classes=2
            state_dict.pop("fc.weight", None)
            state_dict.pop("fc.bias", None)

            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"Loaded local pretrained weights from: {ckpt_path}")
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")

        # Step 3: fallback to torchvision pretrained weights only if no local path
        elif pretrained:
            if self.model_name == "resnet18":
                self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            elif self.model_name == "resnet50":
                self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Step 4: replace classifier head
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