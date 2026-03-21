"""Inference entrypoint for final submission generation."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from meteorite_id.common.utils import get_device, load_yaml
    from meteorite_id.datasets.utils import build_dataloaders
    from meteorite_id.trainers.predict_ensemble import resolve_prediction_sources, run_ensemble_submission
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    from common.utils import get_device, load_yaml
    from datasets.utils import build_dataloaders
    from trainers.predict_ensemble import resolve_prediction_sources, run_ensemble_submission


def find_latest_config(output_dir: Path) -> Path | None:
    """Find the most recent config file in experiment root or fold subdirectories."""
    configs_dir = output_dir / "configs"
    candidates: list[Path] = []

    if configs_dir.exists():
        candidates.extend(configs_dir.glob("config_*.yaml"))

    if not candidates:
        candidates.extend(output_dir.glob("fold_*/configs/config_*.yaml"))

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_exp_args(exp_name: str | None) -> tuple[Path | None, Path | None]:
    """Resolve config path and experiment output directory from experiment name.
    
    Args:
        exp_name: Either a full path to an experiment directory or a relative name
                  like 'resnet50/baseline' which will be resolved against output_dir.
    
    Returns:
        Tuple of (config_path, output_dir)
    """
    if exp_name is None:
        return None, None

    exp_path = Path(exp_name)
    if not exp_path.is_absolute():
        base_output_dir = Path("./outputs")
        exp_path = base_output_dir / exp_path

    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_path}")

    config_path = find_latest_config(exp_path)
    return config_path, exp_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run inference and generate Kaggle submission")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="Path to config yaml (use with --checkpoint)")
    group.add_argument("--exp", type=str, help="Experiment name or path (auto-resolves config, checkpoint, and output)")
    
    parser.add_argument("--checkpoint", type=str, help="Optional single checkpoint override for prediction")
    parser.add_argument("--output_path", type=str, default=None, help="Output CSV path (default: <exp_dir>/submission.csv)")
    return parser.parse_args()


def main() -> None:
    """Run inference pipeline and generate submission CSV."""
    args = parse_args()

    if args.exp is not None:
        config_path, output_dir = resolve_exp_args(args.exp)
        
        if config_path is None:
            raise RuntimeError(f"No config found in experiment directory: {output_dir}")
        
        if args.output_path is None:
            output_path = output_dir / "submission.csv"
        else:
            output_path = Path(args.output_path)
    else:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        output_path = Path(args.output_path) if args.output_path else Path("submission.csv")
        output_dir = None

    cfg = load_yaml(config_path)

    device = get_device(str(cfg.get("device", "auto")))
    _, _, test_loader = build_dataloaders(cfg, include_test=True)
    if test_loader is None:
        raise RuntimeError("Failed to build test dataloader")

    prediction_cfg = cfg.get("prediction", {})
    threshold = float(prediction_cfg.get("threshold", 0.5))
    aggregation = str(prediction_cfg.get("aggregation", "mean")).lower()

    cli_checkpoint = Path(args.checkpoint) if args.checkpoint else None
    prediction_sources = resolve_prediction_sources(
        cfg=cfg,
        config_path=config_path,
        exp_dir=output_dir,
        cli_checkpoint=cli_checkpoint,
    )
    print(f"Resolved {len(prediction_sources)} prediction source(s)")
    print(f"Aggregation: {aggregation}, threshold: {threshold:.2f}")

    data_root = Path(cfg["data_root"]).expanduser()
    template_csv_path = data_root / cfg["sample_submission_path"]

    run_ensemble_submission(
        cfg=cfg,
        loader=test_loader,
        device=device,
        prediction_sources=prediction_sources,
        template_csv_path=template_csv_path,
        output_path=output_path,
        threshold=threshold,
        aggregation=aggregation,
    )

    print(f"Submission saved to: {output_path}")


if __name__ == "__main__":
    main()
