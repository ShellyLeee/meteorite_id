"""Meteorite dataset implementation for train/val/test."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from meteorite_id.datasets.base_dataset import BaseImageDataset
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    from datasets.base_dataset import BaseImageDataset


class MeteoriteDataset(BaseImageDataset):
    """Dataset supporting train/val split, full-train mode, and test inference."""

    def __init__(
        self,
        mode: str,
        csv_path: str | Path | None,
        image_dir: str | Path,
        transform=None,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        super().__init__(transform=transform)
        self.mode = mode
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.image_dir = Path(image_dir)
        self.val_ratio = val_ratio
        self.seed = seed

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        if self.mode in {"train", "val"}:
            self.samples = self._build_train_val_samples()
        elif self.mode == "full_train":
            self.samples = self._build_full_train_samples()
        elif self.mode == "test":
            self.samples = self._build_test_samples()
        else:
            raise ValueError("mode must be one of {'train', 'val', 'full_train', 'test'}")

    def _build_train_val_samples(self) -> list[tuple[Path, int]]:
        if self.csv_path is None:
            raise ValueError("csv_path is required for train/val mode")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Training CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        required_cols = {"id", "label"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {sorted(missing_cols)}")

        df["id"] = df["id"].astype(str)
        df["label"] = df["label"].astype(int)

        stratify = df["label"] if df["label"].nunique() > 1 else None
        train_df, val_df = train_test_split(
            df,
            test_size=self.val_ratio,
            random_state=self.seed,
            shuffle=True,
            stratify=stratify,
        )
        split_df = train_df if self.mode == "train" else val_df

        samples: list[tuple[Path, int]] = []
        missing_files: list[str] = []

        for _, row in split_df.iterrows():
            img_path = self.image_dir / row["id"]
            if not img_path.exists():
                missing_files.append(row["id"])
                continue
            samples.append((img_path, int(row["label"])))

        if missing_files:
            preview = ", ".join(missing_files[:5])
            raise FileNotFoundError(
                f"{len(missing_files)} images listed in CSV were not found in {self.image_dir}. "
                f"Examples: {preview}"
            )

        if not samples:
            raise RuntimeError(f"No usable samples found for mode='{self.mode}'")

        return samples

    def _build_full_train_samples(self) -> list[tuple[Path, int]]:
        if self.csv_path is None:
            raise ValueError("csv_path is required for full_train mode")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Training CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        required_cols = {"id", "label"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {sorted(missing_cols)}")

        df["id"] = df["id"].astype(str)
        df["label"] = df["label"].astype(int)

        samples: list[tuple[Path, int]] = []
        missing_files: list[str] = []

        for _, row in df.iterrows():
            img_path = self.image_dir / row["id"]
            if not img_path.exists():
                missing_files.append(row["id"])
                continue
            samples.append((img_path, int(row["label"])))

        if missing_files:
            preview = ", ".join(missing_files[:5])
            raise FileNotFoundError(
                f"{len(missing_files)} images listed in CSV were not found in {self.image_dir}. "
                f"Examples: {preview}"
            )

        if not samples:
            raise RuntimeError("No usable samples found for mode='full_train'")

        return samples

    def _build_test_samples(self) -> list[Path]:
        image_files = sorted([p for p in self.image_dir.iterdir() if p.is_file() and not p.name.startswith(".")])
        if not image_files:
            raise RuntimeError(f"No test images found in: {self.image_dir}")
        return image_files

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if self.mode in {"train", "val", "full_train"}:
            img_path, label = self.samples[index]
            image = self.load_rgb_image(img_path)
            if self.transform is not None:
                image = self.transform(image)
            return image, label

        img_path = self.samples[index]
        image = self.load_rgb_image(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, str(img_path)
