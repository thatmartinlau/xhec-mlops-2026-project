"""Build, train and evaluate the linear regression pipeline.

The preprocessor is packed inside the pipeline, so the trained object is a
single artifact that can be pickled, loaded and called directly on raw data.
"""

import numpy as np
import pandas as pd
from prefect import task
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from ..utils import pickle_object


@task(name="Build pipeline")
def build_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Wrap the (fitted) preprocessor and a linear regression into one pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


@task(name="Train model")
def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit the pipeline on the training data and return it."""
    pipeline.fit(X_train, y_train)
    return pipeline


@task(name="Evaluate model")
def evaluate(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """Compute RMSE, MAE and R² (all in years) on the given data."""
    y_pred = model.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }


@task(name="Save model")
def save_model(model: Pipeline, path: str | Path) -> None:
    """Save the trained model pipeline to disk as a pickle file."""
    pickle_object(model, path)
