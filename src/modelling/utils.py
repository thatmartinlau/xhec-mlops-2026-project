"""Generic utilities for the modelling package."""

import pickle
from pathlib import Path
from typing import Any


def pickle_object(obj: Any, path: str | Path) -> None:
    """Serialize ``obj`` to ``path`` using pickle.

    Parent directories are created if they do not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def unpickle_object(path: str | Path) -> Any:
    """Deserialize and return the object stored at ``path``."""
    with Path(path).open("rb") as f:
        return pickle.load(f)
