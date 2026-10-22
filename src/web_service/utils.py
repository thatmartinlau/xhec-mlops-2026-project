"""Utilities for the web service."""

import pickle
from pathlib import Path
from typing import Any


def load_object(path: str | Path) -> Any:
    """Deserialize and return the object stored at ``path``."""
    with Path(path).open("rb") as f:
        return pickle.load(f)
