"""Baseline trainer implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from meteorite_id.common.metrics import compute_classification_metrics
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    from common.metrics import compute_classification_metrics


@dataclass
class TrainerState:
    """Simple container for trainer progress."""

    epoch: int = 0
    best_metric: float = float("-inf")
    best_ckpt_path: str = ""
    early_stopping_counter: int = 0


class BaseTrainer:
    """Trainer with train/val loop, checkpointing, and early stopping.

    Best checkpointing and early stopping are always based on validation F1.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        output_dir: str | Path,
        logger: logging.Logger,
        early_stopping_patience: int = 5,
        monitor: str = "f1",
        writer: Any = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.early_stopping_patience = early_stopping_patience
        self.monitor = monitor
        self.writer = writer
        self.state = TrainerState()

    def train_one_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """Train for one epoch and return aggregated metrics."""
        self.model.train()

        running_loss = 0.0
        y_true: list[int] = []
        y_pred: list[int] = []

        progress = tqdm(train_loader, desc="Train", leave=False)
        for images, labels in progress:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)

            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

        epoch_loss = running_loss / len(train_loader.dataset)
        metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred)
        metrics["loss"] = float(epoch_loss)
        return metrics

    def validate_one_epoch(self, val_loader: DataLoader) -> dict[str, float]:
        """Validate for one epoch and return aggregated metrics."""
        self.model.eval()

        running_loss = 0.0
        y_true: list[int] = []
        y_pred: list[int] = []

        with torch.no_grad():
            progress = tqdm(val_loader, desc="Val", leave=False)
            for images, labels in progress:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                running_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)

                y_true.extend(labels.detach().cpu().tolist())
                y_pred.extend(preds.detach().cpu().tolist())

        epoch_loss = running_loss / len(val_loader.dataset)
        metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred)
        metrics["loss"] = float(epoch_loss)
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> None:
        """Run full training loop with checkpointing and early stopping."""
        self.logger.info("Training started: monitor=val_f1, epochs=%d", epochs)

        for epoch in range(1, epochs + 1):
            self.state.epoch = epoch
            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.validate_one_epoch(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            monitor_value = float(val_metrics["f1"])
            improved = monitor_value > self.state.best_metric
            if improved:
                self.state.best_metric = monitor_value
                self.state.early_stopping_counter = 0
                best_ckpt_path = self.output_dir / "best_model.pt"
                self.save_checkpoint(best_ckpt_path)
                self.state.best_ckpt_path = str(best_ckpt_path)
            else:
                self.state.early_stopping_counter += 1

            current_lr = float(self.optimizer.param_groups[0]["lr"])
            self.logger.info(
                "Epoch %d/%d | lr=%.6f | "
                "train_loss=%.4f train_acc=%.4f train_prec=%.4f train_rec=%.4f train_f1=%.4f | "
                "val_loss=%.4f val_acc=%.4f val_prec=%.4f val_rec=%.4f val_f1=%.4f | "
                "best_val_f1=%.4f",
                epoch,
                epochs,
                current_lr,
                train_metrics["loss"],
                train_metrics["accuracy"],
                train_metrics["precision"],
                train_metrics["recall"],
                train_metrics["f1"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
                self.state.best_metric,
            )

            if self.writer is not None:
                self.writer.add_scalar("train/loss", train_metrics["loss"], epoch)
                self.writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
                self.writer.add_scalar("train/precision", train_metrics["precision"], epoch)
                self.writer.add_scalar("train/recall", train_metrics["recall"], epoch)
                self.writer.add_scalar("train/f1", train_metrics["f1"], epoch)
                self.writer.add_scalar("val/loss", val_metrics["loss"], epoch)
                self.writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
                self.writer.add_scalar("val/precision", val_metrics["precision"], epoch)
                self.writer.add_scalar("val/recall", val_metrics["recall"], epoch)
                self.writer.add_scalar("val/f1", val_metrics["f1"], epoch)
                self.writer.add_scalar("train/lr", current_lr, epoch)

            if self.state.early_stopping_counter >= self.early_stopping_patience:
                self.logger.info(
                    "Early stopping at epoch %d (patience=%d)",
                    epoch,
                    self.early_stopping_patience,
                )
                break

        last_ckpt = self.output_dir / "last_model.pt"
        self.save_checkpoint(last_ckpt)

        if self.writer is not None:
            self.writer.close()

        self.logger.info("Training finished. Best checkpoint: %s", self.state.best_ckpt_path)

    def save_checkpoint(self, ckpt_path: str | Path) -> None:
        """Save model, optimizer, and trainer state to a checkpoint."""
        ckpt_path = Path(ckpt_path)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.state.epoch,
            "best_metric": self.state.best_metric,
            "best_ckpt_path": self.state.best_ckpt_path,
            "monitor": "f1",
        }
        torch.save(payload, ckpt_path)

    def load_checkpoint(self, ckpt_path: str | Path) -> None:
        """Load model and optimizer states from checkpoint."""
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.state.epoch = int(checkpoint.get("epoch", 0))
        self.state.best_metric = float(checkpoint.get("best_metric", float("-inf")))
        self.state.best_ckpt_path = str(checkpoint.get("best_ckpt_path", ""))
