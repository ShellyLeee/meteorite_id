"""Training entrypoint."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from sklearn.model_selection import KFold, StratifiedKFold

try:
    from meteorite_id.common.logging import build_logger, build_tensorboard_writer
    from meteorite_id.common.loss import build_loss
    from meteorite_id.common.optim import build_optimizer, build_scheduler
    from meteorite_id.common.utils import ensure_dir, get_device, load_yaml, save_config, set_seed
    from meteorite_id.datasets.utils import build_cv_datasets, build_dataloaders, build_fold_dataloaders
    from meteorite_id.models.utils import build_model
    from meteorite_id.trainers.base_trainer import BaseTrainer
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    from common.logging import build_logger, build_tensorboard_writer
    from common.loss import build_loss
    from common.optim import build_optimizer, build_scheduler
    from common.utils import ensure_dir, get_device, load_yaml, save_config, set_seed
    from datasets.utils import build_cv_datasets, build_dataloaders, build_fold_dataloaders
    from models.utils import build_model
    from trainers.base_trainer import BaseTrainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train meteorite classifier")
    parser.add_argument("--config", type=str, default="cfgs/config.yaml", help="Path to config yaml")
    return parser.parse_args()


def _run_training(
    cfg: dict[str, Any],
    output_dir: Path,
    device,
    logger_name: str,
    train_loader,
    val_loader,
) -> BaseTrainer:
    """Run a single train/val training session and return the trainer."""
    logger = build_logger(name=logger_name, log_dir=output_dir)
    config_path = save_config(cfg, output_dir)
    logger.info("Config saved to: %s", config_path)
    logger.info(
        "Loaded data: train=%d samples, val=%d samples",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    writer = build_tensorboard_writer(output_dir / "tb")

    model = build_model(cfg).to(device)
    criterion = build_loss(cfg.get("loss", {})).to(device)
    optimizer = build_optimizer(model, cfg.get("optimizer", {}))
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}))

    train_cfg = cfg.get("train", {})
    trainer = BaseTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        logger=logger,
        early_stopping_patience=int(train_cfg.get("early_stopping_patience", 5)),
        monitor=str(train_cfg.get("monitor", "f1_tuned")),
        writer=writer,
    )

    epochs = int(train_cfg.get("epochs", 10))
    trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=epochs)
    logger.info("Best checkpoint saved at: %s", trainer.state.best_ckpt_path)
    return trainer


def _run_single_split(cfg: dict[str, Any], output_dir: Path, device) -> None:
    """Run default single train/val split training."""
    train_loader, val_loader, _ = build_dataloaders(cfg, include_test=False)
    _run_training(
        cfg=cfg,
        output_dir=output_dir,
        device=device,
        logger_name="train",
        train_loader=train_loader,
        val_loader=val_loader,
    )


def _run_cross_validation(cfg: dict[str, Any], output_dir: Path, device) -> None:
    """Run optional K-Fold cross validation training."""
    cv_cfg = cfg.get("cv", {})
    n_folds = int(cv_cfg.get("n_folds", 5))
    stratified = bool(cv_cfg.get("stratified", True))
    shuffle = bool(cv_cfg.get("shuffle", True))
    cv_seed = int(cv_cfg.get("seed", cfg.get("seed", 42)))
    target_fold = int(cv_cfg.get("fold", -1))

    if n_folds < 2:
        raise ValueError("cv.n_folds must be >= 2 when cv.enabled=true")

    train_dataset_full, val_dataset_full, labels = build_cv_datasets(cfg)
    indices = list(range(len(labels)))

    if stratified:
        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=shuffle,
            random_state=cv_seed if shuffle else None,
        )
        split_iter = splitter.split(indices, labels)
    else:
        splitter = KFold(
            n_splits=n_folds,
            shuffle=shuffle,
            random_state=cv_seed if shuffle else None,
        )
        split_iter = splitter.split(indices)

    if target_fold >= n_folds:
        raise ValueError(f"cv.fold={target_fold} is out of range for n_folds={n_folds}")

    fold_scores: list[float] = []
    fold_indices: list[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        if target_fold != -1 and fold_idx != target_fold:
            continue

        print(f"[CV] Fold {fold_idx + 1}/{n_folds}")
        fold_output_dir = ensure_dir(output_dir / f"fold_{fold_idx}")
        train_loader, val_loader = build_fold_dataloaders(
            cfg=cfg,
            train_dataset_full=train_dataset_full,
            val_dataset_full=val_dataset_full,
            train_indices=train_idx.tolist(),
            val_indices=val_idx.tolist(),
        )

        trainer = _run_training(
            cfg=cfg,
            output_dir=fold_output_dir,
            device=device,
            logger_name=f"train_fold_{fold_idx}",
            train_loader=train_loader,
            val_loader=val_loader,
        )

        fold_score = float(trainer.state.best_metric)
        fold_scores.append(fold_score)
        fold_indices.append(fold_idx)
        print(f"[CV] Fold {fold_idx + 1}/{n_folds} Best F1: {fold_score:.4f}")

    if not fold_scores:
        raise RuntimeError("No folds were executed. Check cv.fold and cv.n_folds settings.")

    mean_f1 = float(statistics.mean(fold_scores))
    std_f1 = float(statistics.pstdev(fold_scores)) if len(fold_scores) > 1 else 0.0

    summary = {
        "fold_indices": fold_indices,
        "fold_scores": fold_scores,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
    }
    summary_path = output_dir / "cv_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[CV] Mean F1: {mean_f1:.4f}")
    print(f"[CV] Std F1: {std_f1:.4f}")
    print(f"[CV] Summary saved to: {summary_path}")


def main() -> None:
    """Run training pipeline."""
    args = parse_args()
    cfg = load_yaml(args.config)

    output_dir = ensure_dir(cfg.get("output_dir", "./outputs/default"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = get_device(str(cfg.get("device", "auto")))
    print(f"Using device: {device}")

    cv_cfg = cfg.get("cv", {})
    cv_enabled = bool(cv_cfg.get("enabled", False))

    if not cv_enabled:
        _run_single_split(cfg=cfg, output_dir=output_dir, device=device)
    else:
        _run_cross_validation(cfg=cfg, output_dir=output_dir, device=device)


if __name__ == "__main__":
    main()
