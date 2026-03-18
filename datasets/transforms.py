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
        scale_min = float(aug_cfg.get("train_rrc_scale_min", 0.7))
        scale_max = float(aug_cfg.get("train_rrc_scale_max", 1.0))
        flip_p = float(aug_cfg.get("train_flip_p", 0.5))
        rotation = float(aug_cfg.get("train_rotation", 15))
        brightness = float(aug_cfg.get("train_jitter_brightness", 0.1))
        contrast = float(aug_cfg.get("train_jitter_contrast", 0.1))
        saturation = float(aug_cfg.get("train_jitter_saturation", 0.1))
        hue = float(aug_cfg.get("train_jitter_hue", 0.02))

        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(scale_min, scale_max)),
                transforms.RandomHorizontalFlip(p=flip_p),
                transforms.RandomRotation(degrees=rotation),
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    if mode in {"val", "test"}:
        resize_size = int(aug_cfg.get("eval_resize_size", int(image_size * 256 / 224)))
        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    raise ValueError(f"Unsupported transform mode: {mode}")
