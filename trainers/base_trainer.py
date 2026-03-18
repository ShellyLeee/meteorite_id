"""Baseline trainer implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
    history: dict[str, list[float]] = field(default_factory=lambda: {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
    })


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

            # Record history for plotting
            self.state.history["train_loss"].append(train_metrics["loss"])
            self.state.history["val_loss"].append(val_metrics["loss"])
            self.state.history["train_f1"].append(train_metrics["f1"])
            self.state.history["val_f1"].append(val_metrics["f1"])

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

        # Plot and save curves
        self._plot_curves()

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
            "history": self.state.history,
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
        if "history" in checkpoint:
            self.state.history = checkpoint["history"]

    def _plot_curves(self) -> None:
        """Plot and save training curves for loss and F1 score."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not installed, skipping curve plotting")
            return

        history = self.state.history
        epochs = list(range(1, len(history["train_loss"]) + 1))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curve
        ax1 = axes[0]
        ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
        ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # F1 curve
        ax2 = axes[1]
        ax2.plot(epochs, history["train_f1"], "b-", label="Train F1", linewidth=2)
        ax2.plot(epochs, history["val_f1"], "r-", label="Val F1", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("F1 Score")
        ax2.set_title("Training & Validation F1 Score")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        curve_path = self.output_dir / "training_curves.png"
        plt.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Log final metrics
        final_train_loss = history["train_loss"][-1]
        final_val_loss = history["val_loss"][-1]
        final_train_f1 = history["train_f1"][-1]
        final_val_f1 = history["val_f1"][-1]

        self.logger.info("Final Training Loss: %.4f", final_train_loss)
        self.logger.info("Final Validation Loss: %.4f", final_val_loss)
        self.logger.info("Final Training F1: %.4f", final_train_f1)
        self.logger.info("Final Validation F1: %.4f", final_val_f1)
        self.logger.info("Best Validation F1: %.4f", self.state.best_metric)
        self.logger.info("Training curves saved to: %s", curve_path)
