"""Inference logic for the prediction API."""

from pathlib import Path
from typing import Any

from src.modelling.predicting import DEFAULT_MODEL_PATH, load_model, predict_single

from .models import AbaloneFeatures

# Last loaded model and the modification time of its file
_model_cache: dict[str, Any] = {"mtime": None, "model": None}


def get_model(path: Path = DEFAULT_MODEL_PATH) -> Any:
    """Load and cache the pickled model.

    The model is reloaded only when ``model.pkl`` changes on disk, e.g. after
    the scheduled retraining flow has saved a new model.
    """
    mtime = path.stat().st_mtime
    if _model_cache["mtime"] != mtime:
        _model_cache["model"] = load_model(path)
        _model_cache["mtime"] = mtime
    return _model_cache["model"]


def predict_abalone(features: AbaloneFeatures) -> float:
    """Return the predicted age (in years) for one abalone."""
    return predict_single(get_model(), features.to_model_row())
