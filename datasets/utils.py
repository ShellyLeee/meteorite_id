from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

try:
    from meteorite_id.datasets.meteorite_dataset import MeteoriteDataset
    from meteorite_id.datasets.transforms import build_transforms
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    from datasets.meteorite_dataset import MeteoriteDataset
    from datasets.transforms import build_transforms


def _resolve_image_dir(path: Path) -> Path:
    """Resolve image directory, allowing one extra nested folder layer.

    Examples:
        train_images/
        train_images/train_images/
    """
    if not path.exists():
        raise FileNotFoundError(f"Image directory not found: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {path}")

    files = [p for p in path.iterdir() if p.is_file() and not p.name.startswith(".")]
    if files:
        return path

    subdirs = [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if len(subdirs) == 1:
        nested = subdirs[0]
        nested_files = [p for p in nested.iterdir() if p.is_file() and not p.name.startswith(".")]
        if nested_files:
            print(f"[Data] Detected nested image directory, using: {nested}")
            return nested

    return path


def _resolve_data_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    data_root = Path(cfg["data_root"])
    train_csv = data_root / cfg["train_csv"]
    train_image_dir = _resolve_image_dir(data_root / cfg["train_image_dir"])
    test_image_dir = _resolve_image_dir(data_root / cfg["test_image_dir"])
    sample_submission_path = data_root / cfg["sample_submission_path"]

    return {
        "data_root": data_root,
        "train_csv": train_csv,
        "train_image_dir": train_image_dir,
        "test_image_dir": test_image_dir,
        "sample_submission_path": sample_submission_path,
    }


def build_datasets(cfg: dict[str, Any], include_test: bool = True) -> dict[str, MeteoriteDataset]:
    """Build dataset instances."""
    paths = _resolve_data_paths(cfg)

    train_transform = build_transforms(mode="train", cfg=cfg)
    eval_transform = build_transforms(mode="val", cfg=cfg)

    val_ratio = float(cfg.get("val_ratio", 0.2))
    seed = int(cfg.get("seed", 42))

    datasets: dict[str, MeteoriteDataset] = {
        "train": MeteoriteDataset(
            mode="train",
            csv_path=paths["train_csv"],
            image_dir=paths["train_image_dir"],
            transform=train_transform,
            val_ratio=val_ratio,
            seed=seed,
        ),
        "val": MeteoriteDataset(
            mode="val",
            csv_path=paths["train_csv"],
            image_dir=paths["train_image_dir"],
            transform=eval_transform,
            val_ratio=val_ratio,
            seed=seed,
        ),
    }

    if include_test:
        datasets["test"] = MeteoriteDataset(
            mode="test",
            csv_path=paths["train_csv"],  # keep for now if your dataset class requires it
            image_dir=paths["test_image_dir"],
            transform=build_transforms(mode="test", cfg=cfg),
            val_ratio=val_ratio,
            seed=seed,
        )

    return datasets


def build_dataloaders(cfg: dict[str, Any], include_test: bool = True):
    """Build train, val, and optionally test dataloaders."""
    datasets = build_datasets(cfg, include_test=include_test)

    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 4))
    pin_memory = True

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = None
    if include_test:
        test_loader = DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader