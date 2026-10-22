"""Inference logic for the prediction API."""

from functools import lru_cache
from typing import Any

from src.modelling.predicting import load_model, predict_single

from .models import AbaloneFeatures


@lru_cache(maxsize=1)
def get_model() -> Any:
    """Load and cache the pickled model.

    Cached per process: the first call reads ``model.pkl`` from
    ``src/web_service/local_objects/``, subsequent calls reuse the same object.
    """
    return load_model()


def predict_abalone(features: AbaloneFeatures) -> float:
    """Return the predicted age (in years) for one abalone."""
    return predict_single(get_model(), features.to_model_row())
