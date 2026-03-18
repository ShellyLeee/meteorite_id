"""Inference entrypoint for final submission generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from meteorite_id.common.utils import get_device, load_yaml
    from meteorite_id.datasets.utils import build_dataloaders
    from meteorite_id.models.utils import build_model
    from meteorite_id.trainers.predict import make_submission, predict
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    from common.utils import get_device, load_yaml
    from datasets.utils import build_dataloaders
    from models.utils import build_model
    from trainers.predict import make_submission, predict



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run inference and generate Kaggle submission")
    parser.add_argument("--config", type=str, default="cfgs/resnet50_v1.yaml", help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_path", type=str, default="submission.csv", help="Output CSV path")
    return parser.parse_args()



def main() -> None:
    """Run inference pipeline and generate submission CSV."""
    args = parse_args()
    cfg = load_yaml(args.config)

    device = get_device(str(cfg.get("device", "auto")))
    _, _, test_loader = build_dataloaders(cfg, include_test=True)
    if test_loader is None:
        raise RuntimeError("Failed to build test dataloader")

    model = build_model(cfg).to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint missing 'model_state_dict'")

    model.load_state_dict(checkpoint["model_state_dict"])
    id_to_pred = predict(model=model, loader=test_loader, device=device)

    data_root = Path(cfg["data_root"])
    template_csv_path = data_root / cfg["sample_submission_path"]

    make_submission(
        id_to_pred=id_to_pred,
        template_csv_path=template_csv_path,
        output_path=args.output_path,
    )

    print(f"Submission saved to: {args.output_path}")


if __name__ == "__main__":
    main()
