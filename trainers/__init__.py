"""Training and inference orchestration."""

try:
    from meteorite_id.trainers.base_trainer import BaseTrainer
    from meteorite_id.trainers.predict import make_submission, predict
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    from trainers.base_trainer import BaseTrainer
    from trainers.predict import make_submission, predict

__all__ = ["BaseTrainer", "predict", "make_submission"]
