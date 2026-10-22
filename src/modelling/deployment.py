"""Deploy the training flow so the model is retrained on a regular schedule.
Run from the project root (the command keeps running to serve the scheduled runs)::
    uv run python -m src.modelling.deployment
"""

from pathlib import Path

from .main import train_model_workflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINSET_PATH = PROJECT_ROOT / "data" / "abalone.csv"
# Cron expression: every Monday at 00:00
RETRAINING_SCHEDULE = "0 0 * * 1"


if __name__ == "__main__":
    train_model_workflow.serve(
        name="weekly-retraining",
        cron=RETRAINING_SCHEDULE,
        parameters={"trainset_path": str(TRAINSET_PATH)},
        tags=["training"],
    )
