"""Inference and submission helpers."""

from __future__ import annotations
from pathlib import Path
import os

import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

def predict(model, loader, device, threshold: float = 0.5):
    model.eval()
    id_to_pred = {}

    with torch.no_grad():
        for images, img_paths in tqdm(loader, desc="Predicting", leave=False):
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            if outputs.shape[1] == 2:
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = (probs >= threshold).long().cpu().numpy().tolist()
            else:
                preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()

            for pred, path in zip(preds, img_paths):
                image_id = os.path.basename(path)
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
