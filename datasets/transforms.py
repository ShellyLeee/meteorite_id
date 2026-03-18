"""Transform factory for train/val/test splits."""

from __future__ import annotations

from typing import Any

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]



def build_transforms(mode: str, cfg: dict[str, Any]):
    """Build torchvision transforms for a split."""
    image_size = int(cfg.get("image_size", 224))
    aug_cfg = cfg.get("aug", {})

    if mode == "train":
        flip_p = float(aug_cfg.get("train_flip_p", 0.5))
        rotation = float(aug_cfg.get("train_rotation", 10))
        jitter_strength = float(aug_cfg.get("train_color_jitter", 0.2))

        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=flip_p),
                transforms.RandomRotation(degrees=rotation),
                transforms.ColorJitter(
                    brightness=jitter_strength,
                    contrast=jitter_strength,
                    saturation=jitter_strength,
                    hue=min(0.1, jitter_strength / 2),
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    if mode in {"val", "test"}:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    raise ValueError(f"Unsupported transform mode: {mode}")
