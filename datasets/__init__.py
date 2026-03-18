"""Dataset package."""

try:
    from meteorite_id.datasets.base_dataset import BaseImageDataset
    from meteorite_id.datasets.meteorite_dataset import MeteoriteDataset
    from meteorite_id.datasets.transforms import build_transforms
    from meteorite_id.datasets.utils import build_dataloaders, build_datasets
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    from datasets.base_dataset import BaseImageDataset
    from datasets.meteorite_dataset import MeteoriteDataset
    from datasets.transforms import build_transforms
    from datasets.utils import build_dataloaders, build_datasets

__all__ = [
    "BaseImageDataset",
    "MeteoriteDataset",
    "build_transforms",
    "build_datasets",
    "build_dataloaders",
]
