"""Inference entrypoint for final submission generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

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


def find_latest_config(output_dir: Path) -> Path | None:
    """Find the most recent config file in the experiment's configs directory."""
    configs_dir = output_dir / "configs"
    if not configs_dir.exists():
        return None

    config_files = sorted(configs_dir.glob("config_*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
    return config_files[0] if config_files else None


def find_best_checkpoint(output_dir: Path) -> Path | None:
    """Find the best model checkpoint in the experiment directory."""
    best_ckpt = output_dir / "best_model.pt"
    if best_ckpt.exists():
        return best_ckpt
    return None


def resolve_exp_args(exp_name: str | None) -> tuple[Path | None, Path | None, Path | None]:
    """Resolve config, checkpoint, and output path from experiment name.
    
    Args:
        exp_name: Either a full path to an experiment directory or a relative name
                  like 'resnet50/baseline' which will be resolved against output_dir.
    
    Returns:
        Tuple of (config_path, checkpoint_path, output_dir)
    """
    if exp_name is None:
        return None, None, None

    exp_path = Path(exp_name)
    if not exp_path.is_absolute():
        base_output_dir = Path("./outputs")
        exp_path = base_output_dir / exp_path

    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_path}")

    config_path = find_latest_config(exp_path)
    checkpoint_path = find_best_checkpoint(exp_path)
    
    return config_path, checkpoint_path, exp_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run inference and generate Kaggle submission")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="Path to config yaml (use with --checkpoint)")
    group.add_argument("--exp", type=str, help="Experiment name or path (auto-resolves config, checkpoint, and output)")
    
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint (required if using --config)")
    parser.add_argument("--output_path", type=str, default=None, help="Output CSV path (default: <exp_dir>/submission.csv)")
    return parser.parse_args()


def main() -> None:
    """Run inference pipeline and generate submission CSV."""
    args = parse_args()

    if args.exp is not None:
        config_path, checkpoint_path, output_dir = resolve_exp_args(args.exp)
        
        if config_path is None:
            raise RuntimeError(f"No config found in experiment directory: {output_dir}")
        if checkpoint_path is None:
            raise RuntimeError(f"No checkpoint found in experiment directory: {output_dir}")
        
        if args.output_path is None:
            output_path = output_dir / "submission.csv"
        else:
            output_path = Path(args.output_path)
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when using --config")
        
        config_path = Path(args.config)
        checkpoint_path = Path(args.checkpoint)
        output_path = Path(args.output_path) if args.output_path else Path("submission.csv")
        output_dir = None

    cfg = load_yaml(config_path)

    device = get_device(str(cfg.get("device", "auto")))
    _, _, test_loader = build_dataloaders(cfg, include_test=True)
    if test_loader is None:
        raise RuntimeError("Failed to build test dataloader")

    model = build_model(cfg).to(device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint missing 'model_state_dict'")

    model.load_state_dict(checkpoint["model_state_dict"])
    print("Using fixed threshold: 0.50")

    id_to_pred = predict(model=model, loader=test_loader, device=device, threshold=0.5)

    data_root = Path(cfg["data_root"])
    template_csv_path = data_root / cfg["sample_submission_path"]

    make_submission(
        id_to_pred=id_to_pred,
        template_csv_path=template_csv_path,
        output_path=output_path,
    )

    print(f"Submission saved to: {output_path}")


if __name__ == "__main__":
    main()
