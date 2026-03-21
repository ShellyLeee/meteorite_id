"""Training and inference orchestration."""

try:
    from meteorite_id.trainers.base_trainer import BaseTrainer
    from meteorite_id.trainers.predict import make_submission, predict
    from meteorite_id.trainers.predict_ensemble import ensemble_predict, resolve_prediction_sources
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    from trainers.base_trainer import BaseTrainer
    from trainers.predict import make_submission, predict
    from trainers.predict_ensemble import ensemble_predict, resolve_prediction_sources

__all__ = ["BaseTrainer", "predict", "make_submission", "ensemble_predict", "resolve_prediction_sources"]
