"""Inference and submission helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader



def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, int]:
    """Run test inference and return id-to-label predictions.

    The dataloader must yield `(images, img_paths)` where `img_paths` are strings.
    """
    model.eval()
    id_to_pred: dict[str, int] = {}

    with torch.no_grad():
        for images, img_paths in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1).detach().cpu().tolist()

            for img_path, pred in zip(img_paths, preds):
                image_id = Path(str(img_path)).name
                id_to_pred[image_id] = int(pred)

    return id_to_pred



def make_submission(id_to_pred: dict[str, int], template_csv_path: str | Path, output_path: str | Path) -> None:
    """Generate submission CSV based on sample_submission order."""
    template_csv_path = Path(template_csv_path)
    output_path = Path(output_path)

    if not template_csv_path.exists():
        raise FileNotFoundError(f"Template submission file not found: {template_csv_path}")

    submission_df = pd.read_csv(template_csv_path)
    if "id" not in submission_df.columns:
        raise RuntimeError("sample_submission.csv must contain an 'id' column")

    id_to_pred_by_basename = {Path(str(k)).name: int(v) for k, v in id_to_pred.items()}
    labels = submission_df["id"].astype(str).map(lambda x: id_to_pred_by_basename.get(Path(x).name))
    missing_mask = labels.isna()
    if missing_mask.any():
        missing_ids = submission_df.loc[missing_mask, "id"].astype(str).tolist()
        examples = ", ".join(missing_ids[:5])
        raise RuntimeError(
            f"Missing predictions for {len(missing_ids)} ids from sample submission. Examples: {examples}"
        )

    submission_df["label"] = labels.astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
