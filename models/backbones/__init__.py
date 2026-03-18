"""Backbone implementations."""

try:
    from meteorite_id.models.backbones.resnet import ResNetClassifier
    from meteorite_id.models.backbones.vit import ViTClassifier
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    from models.backbones.resnet import ResNetClassifier
    from models.backbones.vit import ViTClassifier

__all__ = ["ResNetClassifier", "ViTClassifier"]
