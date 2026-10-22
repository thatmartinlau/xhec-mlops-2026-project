"""Inference: load a pickled model and predict abalone age."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import unpickle_object

# Columns the trained pipeline expects, in the order the model was fitted.
FEATURE_COLUMNS = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
]

# Default model location: `src/web_service/local_objects/model.pkl`.
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "web_service"
    / "local_objects"
    / "model.pkl"
)


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    """Load a pickled model.

    Defaults to the model pickled by ``src.modelling.main`` in the web
    service's ``local_objects`` folder.
    """
    return unpickle_object(path)


def predict(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return predicted ages (in years) for a batch of abalone measurements.

    ``features`` must contain every column in :data:`FEATURE_COLUMNS`.
    """
    return model.predict(features[FEATURE_COLUMNS])


def predict_single(model: Any, features: dict) -> float:
    """Return the predicted age (in years) for a single abalone.

    ``features`` is a mapping with one entry per column in
    :data:`FEATURE_COLUMNS`, e.g.::

        predict_single(model, {
            "Sex": "M",
            "Length": 0.455,
            "Diameter": 0.365,
            "Height": 0.095,
            "Whole weight": 0.514,
            "Shucked weight": 0.2245,
            "Viscera weight": 0.101,
            "Shell weight": 0.15,
        })
    """
    return float(model.predict(pd.DataFrame([features])[FEATURE_COLUMNS])[0])
