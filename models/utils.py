"""Model factory utilities."""

from __future__ import annotations

from typing import Any

try:
    from meteorite_id.models.backbones.resnet import ResNetClassifier
    from meteorite_id.models.base_model import BaseClassifier
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    from models.backbones.resnet import ResNetClassifier
    from models.base_model import BaseClassifier



def build_model(cfg: dict[str, Any]) -> BaseClassifier:
    """Build model from top-level config."""
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "resnet18")).lower()
    pretrained = bool(model_cfg.get("pretrained", True))
    num_classes = int(cfg.get("num_classes", 2))
    pretrained_path = model_cfg.get("pretrained_path")

    if model_name in {"resnet18", "resnet50"}:
        return ResNetClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
        )

    if model_name.startswith("vit"):
        raise NotImplementedError("ViT is not implemented in this baseline. Use resnet18 or resnet50.")

    raise ValueError(f"Unsupported model name: {model_name}")
