"""Ensemble inference utilities for test prediction and submission generation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

try:
    from meteorite_id.models.utils import build_model
    from meteorite_id.trainers.predict import make_submission
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    from models.utils import build_model
    from trainers.predict import make_submission


def _logits_to_positive_proba(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to positive-class probability for binary classification."""
    if logits.ndim == 1:
        return torch.sigmoid(logits)

    if logits.ndim != 2:
        raise RuntimeError(f"Expected logits with 1D/2D shape, got: {tuple(logits.shape)}")

    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    if logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]

    raise RuntimeError(
        "Binary inference expects model output dim=1 or dim=2. "
        f"Got num_classes={logits.shape[1]}"
    )


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract model state_dict from common checkpoint formats."""
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]

        tensor_like = [k for k, v in checkpoint.items() if torch.is_tensor(v)]
        if tensor_like:
            return checkpoint

    raise RuntimeError("Unsupported checkpoint format. Expect state_dict/model_state_dict in checkpoint.")


def _resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base_dir / path).resolve()


def _predict_source_probabilities(model, loader, device) -> dict[str, float]:
    model.eval()
    id_to_prob: dict[str, float] = {}

    with torch.no_grad():
        for images, img_paths in tqdm(loader, desc="Predicting source", leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = _logits_to_positive_proba(logits).detach().cpu().tolist()

            for prob, path in zip(probs, img_paths):
                image_id = Path(str(path)).name
                id_to_prob[image_id] = float(prob)

    return id_to_prob


def resolve_prediction_sources(
    cfg: dict[str, Any],
    config_path: Path | None,
    exp_dir: Path | None = None,
    cli_checkpoint: Path | None = None,
) -> list[dict[str, Any]]:
    """Resolve prediction source list from config, with sensible fallbacks."""
    prediction_cfg = cfg.get("prediction", {})
    raw_sources = prediction_cfg.get("sources", cfg.get("prediction_sources", []))

    if cli_checkpoint is not None:
        base_dir = config_path.parent if config_path is not None else Path.cwd()
        return [
            {
                "name": "single_ckpt",
                "model_name": cfg.get("model", {}).get("name"),
                "checkpoint_path": str(_resolve_path(cli_checkpoint, base_dir)),
                "weight": 1.0,
                "fold": None,
                "seed": None,
            }
        ]

    if raw_sources:
        base_dir = config_path.parent if config_path is not None else Path.cwd()
        sources: list[dict[str, Any]] = []
        for idx, source in enumerate(raw_sources):
            if "checkpoint_path" not in source:
                raise ValueError(f"prediction source at index {idx} missing required key: checkpoint_path")

            resolved_source = dict(source)
            resolved_source.setdefault("name", f"source_{idx}")
            resolved_source.setdefault("weight", 1.0)
            resolved_source.setdefault("model_name", cfg.get("model", {}).get("name"))
            resolved_source.setdefault("fold", None)
            resolved_source.setdefault("seed", None)
            resolved_source["checkpoint_path"] = str(_resolve_path(source["checkpoint_path"], base_dir))
            sources.append(resolved_source)
        return sources

    if exp_dir is not None:
        fold_ckpts = sorted(exp_dir.glob("fold_*/best_model.pt"))
        if fold_ckpts:
            sources = []
            for fold_ckpt in fold_ckpts:
                fold_name = fold_ckpt.parent.name
                fold_idx = int(fold_name.split("_")[-1]) if "_" in fold_name else None
                sources.append(
                    {
                        "name": fold_name,
                        "model_name": cfg.get("model", {}).get("name"),
                        "checkpoint_path": str(fold_ckpt.resolve()),
                        "weight": 1.0,
                        "fold": fold_idx,
                        "seed": None,
                    }
                )
            return sources

        single_ckpt = exp_dir / "best_model.pt"
        if single_ckpt.exists():
            return [
                {
                    "name": "single_ckpt",
                    "model_name": cfg.get("model", {}).get("name"),
                    "checkpoint_path": str(single_ckpt.resolve()),
                    "weight": 1.0,
                    "fold": None,
                    "seed": None,
                }
            ]

    raise RuntimeError(
        "No prediction sources resolved. Please provide prediction.sources / prediction_sources in config, "
        "or pass --checkpoint, or use --exp with available best_model.pt checkpoints."
    )


def ensemble_predict(
    cfg: dict[str, Any],
    loader,
    device,
    prediction_sources: list[dict[str, Any]],
    threshold: float = 0.5,
    aggregation: str = "mean",
) -> tuple[dict[str, float], dict[str, int]]:
    """Run source-wise inference and aggregate probabilities."""
    if aggregation != "mean":
        raise ValueError(f"Unsupported aggregation method: {aggregation}")

    accumulated_prob: dict[str, float] = {}
    accumulated_weight: dict[str, float] = {}

    for idx, source in enumerate(prediction_sources):
        source_name = str(source.get("name", f"source_{idx}"))
        checkpoint_path = Path(str(source["checkpoint_path"]))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"[{source_name}] checkpoint not found: {checkpoint_path}")

        source_cfg = deepcopy(cfg)
        source_model_name = source.get("model_name")
        if source_model_name:
            source_cfg.setdefault("model", {})
            source_cfg["model"]["name"] = source_model_name

        model = build_model(source_cfg).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = _extract_state_dict(checkpoint)
        model.load_state_dict(state_dict)

        weight = float(source.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"[{source_name}] weight must be > 0, got {weight}")

        id_to_prob = _predict_source_probabilities(model=model, loader=loader, device=device)
        if not id_to_prob:
            raise RuntimeError(f"[{source_name}] got empty predictions")

        for image_id, prob in id_to_prob.items():
            accumulated_prob[image_id] = accumulated_prob.get(image_id, 0.0) + prob * weight
            accumulated_weight[image_id] = accumulated_weight.get(image_id, 0.0) + weight

    ensemble_prob = {k: accumulated_prob[k] / accumulated_weight[k] for k in accumulated_prob}
    ensemble_pred = {k: int(v >= threshold) for k, v in ensemble_prob.items()}
    return ensemble_prob, ensemble_pred


def run_ensemble_submission(
    cfg: dict[str, Any],
    loader,
    device,
    prediction_sources: list[dict[str, Any]],
    template_csv_path: str | Path,
    output_path: str | Path,
    threshold: float = 0.5,
    aggregation: str = "mean",
) -> None:
    _, id_to_pred = ensemble_predict(
        cfg=cfg,
        loader=loader,
        device=device,
        prediction_sources=prediction_sources,
        threshold=threshold,
        aggregation=aggregation,
    )
    make_submission(id_to_pred=id_to_pred, template_csv_path=template_csv_path, output_path=output_path)
