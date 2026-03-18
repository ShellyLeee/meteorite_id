"""Model package."""

try:
    from meteorite_id.models.base_model import BaseClassifier
    from meteorite_id.models.utils import build_model
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    from models.base_model import BaseClassifier
    from models.utils import build_model

__all__ = ["BaseClassifier", "build_model"]
