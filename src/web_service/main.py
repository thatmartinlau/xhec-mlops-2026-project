"""FastAPI prediction service for the abalone age model."""

from fastapi import FastAPI

from .lib.inference import predict_abalone
from .lib.models import AbaloneFeatures, PredictionResponse


app = FastAPI(
    title="Abalone Age Prediction API",
    description=(
        "Predict the age of an abalone (in years) from its physical "
        "measurements, instead of counting shell rings under a microscope."
    ),
    version="0.1.0",
)


@app.get("/")
def home() -> dict:
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=PredictionResponse, status_code=201)
def predict(payload: AbaloneFeatures) -> PredictionResponse:
    """Predict the age (in years) of a single abalone."""
    return PredictionResponse(predicted_age=predict_abalone(payload))
