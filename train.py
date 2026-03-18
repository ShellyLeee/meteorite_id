"""Training entrypoint."""

from __future__ import annotations

import argparse

try:
    from meteorite_id.common.logging import build_logger, build_tensorboard_writer
    from meteorite_id.common.loss import build_loss
    from meteorite_id.common.optim import build_optimizer, build_scheduler
    from meteorite_id.common.utils import ensure_dir, get_device, load_yaml, set_seed
    from meteorite_id.datasets.utils import build_dataloaders
    from meteorite_id.models.utils import build_model
    from meteorite_id.trainers.base_trainer import BaseTrainer
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    from common.logging import build_logger, build_tensorboard_writer
    from common.loss import build_loss
    from common.optim import build_optimizer, build_scheduler
    from common.utils import ensure_dir, get_device, load_yaml, set_seed
    from datasets.utils import build_dataloaders
    from models.utils import build_model
    from trainers.base_trainer import BaseTrainer



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train meteorite classifier")
    parser.add_argument("--config", type=str, default="cfgs/config.yaml", help="Path to config yaml")
    return parser.parse_args()



def main() -> None:
    """Run training pipeline."""
    args = parse_args()
    cfg = load_yaml(args.config)

    output_dir = ensure_dir(cfg.get("output_dir", "./outputs/default"))
    logger = build_logger(name="train", log_dir=output_dir)
    writer = build_tensorboard_writer(output_dir / "tb")

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = get_device(str(cfg.get("device", "auto")))
    logger.info("Using device: %s", device)

    train_loader, val_loader, _ = build_dataloaders(cfg, include_test=False)
    logger.info(
        "Loaded data: train=%d samples, val=%d samples",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

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
        monitor="f1",
        writer=writer,
    )

    epochs = int(train_cfg.get("epochs", 10))
    trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=epochs)

    logger.info("Best checkpoint saved at: %s", trainer.state.best_ckpt_path)


if __name__ == "__main__":
    main()
