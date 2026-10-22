"""Clean the raw abalone data, build X/y, and transform the features.

The preprocessor is fitted on the training set only and packed into the
model pipeline (see ``training.build_pipeline``), so it is pickled with the
model and applied identically at prediction time.
"""

from pathlib import Path

import pandas as pd
from prefect import task
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RINGS_COLUMN = "Rings"
TARGET_COLUMN = "Age"
CATEGORICAL_FEATURES = ["Sex"]
# Age in years: the number of shell rings + 1.5 (standard convention for abalone)
RINGS_TO_AGE_OFFSET = 1.5
TEST_SIZE = 0.2
RANDOM_STATE = 42


@task(name="Load data")
def load_data(path: str | Path) -> pd.DataFrame:
    """Load the abalone dataset from a CSV file."""
    return pd.read_csv(path)


@task(name="Remove invalid rows")
def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with physically impossible values.

    A row is invalid if its ``Height`` is 0, or if a part (shucked, viscera or
    shell) is heavier than the whole abalone.
    """
    invalid_mask = (
        (df["Height"] == 0)
        | (df["Shucked weight"] > df["Whole weight"])
        | (df["Viscera weight"] > df["Whole weight"])
        | (df["Shell weight"] > df["Whole weight"])
    )
    return df[~invalid_mask].reset_index(drop=True)


@task(name="Construct target variable")
def construct_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``Age`` target column, computed from the number of rings."""
    df = df.copy()
    df[TARGET_COLUMN] = df[RINGS_COLUMN] + RINGS_TO_AGE_OFFSET
    return df


@task(name="Extract X and y")
def extract_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the data into features (X) and target (y).

    ``Rings`` is dropped from the features since the target is derived from it.
    """
    X = df.drop(columns=[RINGS_COLUMN, TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


@task(name="Split train and test")
def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into train and test sets (X_train, X_test, y_train, y_test)."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


@task(name="Fit preprocessor")
def fit_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit the preprocessor on the training set.

    It one-hot encodes the categorical features and standardizes the numerical ones.
    """
    numerical_features = [
        col for col in X_train.columns if col not in CATEGORICAL_FEATURES
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numerical", StandardScaler(), numerical_features),
        ],
        verbose_feature_names_out=False,
    )
    # Output DataFrames with readable column names (e.g. `Sex_M`, `Length`)
    preprocessor.set_output(transform="pandas")
    preprocessor.fit(X_train)
    return preprocessor


@task(name="Transform features")
def transform_features(
    X: pd.DataFrame, preprocessor: ColumnTransformer
) -> pd.DataFrame:
    """Transform the features with the fitted preprocessor."""
    return preprocessor.transform(X)
