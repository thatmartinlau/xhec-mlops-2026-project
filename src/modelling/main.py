# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path

from prefect import flow

from .predicting import DEFAULT_MODEL_PATH
from .preprocessing import (
    construct_target_variable,
    extract_X_y,
    fit_preprocessor,
    load_data,
    remove_invalid_rows,
    split_train_test,
)
from .training import build_pipeline, evaluate, train_model, save_model


@flow(name="Train model", log_prints=True)
def train_model_workflow(
    trainset_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, float]:
    """Train the model on ``trainset_path``, pickle it to ``model_path``,
    and return the test metrics.
    """
    df = load_data(trainset_path)
    df = remove_invalid_rows(df)
    df = construct_target_variable(df)
    X, y = extract_X_y(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    preprocessor = fit_preprocessor(X_train)
    pipeline = build_pipeline(preprocessor)
    train_model(pipeline, X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)
    save_model(pipeline, model_path)

    print(f"Model saved to {model_path}")
    print(f"Test metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the abalone age model on the data at the given path."
    )
    parser.add_argument("trainset_path", type=str, help="Path to the training CSV file")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help=f"Where to save the pickled model (default: {DEFAULT_MODEL_PATH})",
    )
    args = parser.parse_args()
    train_model_workflow(args.trainset_path, model_path=args.model_path)
