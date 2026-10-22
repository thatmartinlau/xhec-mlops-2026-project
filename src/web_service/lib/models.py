"""Pydantic schemas for the abalone prediction API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AbaloneFeatures(BaseModel):
    """Physical measurements of a single abalone.

    Field names use ``snake_case`` for a clean JSON API; ``to_model_row``
    maps them to the column names the trained pipeline expects.
    As in the training data, continuous values are scaled: millimetres and grams
    divided by 200 (e.g. a length of 91 mm is sent as 0.455).
    """

    # Example request body shown in the interactive docs (/docs)
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "sex": "M",
                    "length": 0.455,
                    "diameter": 0.365,
                    "height": 0.095,
                    "whole_weight": 0.514,
                    "shucked_weight": 0.2245,
                    "viscera_weight": 0.101,
                    "shell_weight": 0.15,
                }
            ]
        }
    )

    sex: Literal["M", "F", "I"] = Field(
        ..., description="M = male, F = female, I = infant"
    )
    length: float = Field(..., gt=0, description="Longest shell measurement (mm)")
    diameter: float = Field(..., gt=0, description="Perpendicular to length (mm)")
    height: float = Field(..., gt=0, description="Height with meat in shell (mm)")
    whole_weight: float = Field(..., gt=0, description="Whole abalone (g)")
    shucked_weight: float = Field(..., gt=0, description="Weight of meat (g)")
    viscera_weight: float = Field(
        ..., gt=0, description="Gut weight after bleeding (g)"
    )
    shell_weight: float = Field(..., gt=0, description="Shell weight after drying (g)")

    @model_validator(mode="after")
    def _check_physical_consistency(self) -> "AbaloneFeatures":
        """Reject values that would have been dropped during training."""
        for part in ("shucked_weight", "viscera_weight", "shell_weight"):
            if getattr(self, part) > self.whole_weight:
                raise ValueError(f"{part} cannot exceed whole_weight")
        return self

    def to_model_row(self) -> dict:
        """Return a mapping compatible with ``predicting.FEATURE_COLUMNS``."""
        return {
            "Sex": self.sex,
            "Length": self.length,
            "Diameter": self.diameter,
            "Height": self.height,
            "Whole weight": self.whole_weight,
            "Shucked weight": self.shucked_weight,
            "Viscera weight": self.viscera_weight,
            "Shell weight": self.shell_weight,
        }


class PredictionResponse(BaseModel):
    """Predicted age of the abalone, in years."""

    predicted_age: float = Field(
        ..., description="Predicted age in years (number of shell rings + 1.5)"
    )
