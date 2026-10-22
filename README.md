<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

## 🎯 Project Overview

This **group 7**'s submission for the X-HEC MLOps course with Artefact.

Group members:
- Martin Lau (martin.lau@hec.edu)
- Yuqi Xin (yuqi.xin@hec.edu)
- Julian Fix (julian.fix@hec.edu)
- Scott Mayette (scott.mayette@hec.edu)

This project includes:
- Notebook exploration (EDA and a first model tracked with MLflow)
- A training pipeline orchestrated with Prefect (`src/modelling`)
- A Prefect deployment that retrains the model every week
- A FastAPI prediction API, packaged with the Prefect server in a Docker container


## 🚀 Quick Start

### Setup Steps

1. **Set up your development environment**:
   ```bash
   # Create and activate a virtual environment
   uv sync
   source .venv/bin/activate # on Windows (Git Bash): source .venv/Scripts/activate

   # Install pre-commit hooks for code quality
   uv run pre-commit install
   ```

2. **Download the data**: get `abalone.csv` from the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) and save it as `data/abalone.csv` (the `data/` folder is git-ignored).

### Training pipeline with Prefect

The training code in `src/modelling` is a Prefect **flow** (`Training pipeline`, in `main.py`) made of **tasks**, one per step: load data, remove invalid rows, construct the target, extract X and y, split train/test, build the preprocessor, build the pipeline, train (fits the preprocessor and the model together), evaluate and save the model. The trained pipeline is saved to `src/web_service/local_objects/model.pkl`.

All commands below are run from the project root.

1. **Point Prefect to a local server** (only needed once):
   ```bash
   uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
   ```

2. **Start the Prefect server and UI** in a first terminal (it keeps running):
   ```bash
   uv run prefect server start
   ```
   Open the UI at http://127.0.0.1:4200. To reset the database: `uv run prefect server database reset`.

3. **Run the training flow once** in a second terminal:
   ```bash
   uv run python -m src.modelling.main data/abalone.csv
   ```
   The run appears in the UI under **Runs**: click on it to see each task, its logs and the test metrics (RMSE, MAE, R²).

4. **Deploy the weekly retraining** in the second terminal (it keeps running to serve the scheduled runs):
   ```bash
   uv run python -m src.modelling.deployment
   ```
   The `weekly-retraining` deployment appears in the UI under **Deployments**, scheduled every Monday at 00:00 (cron `0 0 * * 1`, set in `src/modelling/deployment.py`). To trigger a run now, click **Run** in the UI or use, from a third terminal:
   ```bash
   uv run prefect deployment run "Training pipeline/weekly-retraining"
   ```
   Stopping the command (`Ctrl+C`) stops the scheduled runs; the deployment stays visible in the UI.


### Prediction API

The API (`src/web_service`) loads the trained pipeline from `src/web_service/local_objects/model.pkl` and exposes:

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predicted age (in years) of one abalone |

Inputs are validated with Pydantic: `sex` must be `M`, `F` or `I`, measurements must be positive, and no part can weigh more than the whole abalone (otherwise the API answers `422`). As in the dataset, measurements are **scaled**: millimetres and grams divided by 200 (a length of 91 mm is sent as `0.455`). The API reloads the model automatically when `model.pkl` is updated by a retraining run.

#### Run the API locally

Train a model first (see above), then:

```bash
uv run uvicorn src.web_service.main:app --port 8001
```

The interactive documentation is at http://127.0.0.1:8001/docs.

#### Run everything with Docker

The container starts the Prefect server, trains a first model, serves the weekly retraining deployment and starts the API (see `bin/run_services.sh`). From the project root, with the data in `data/abalone.csv`:

```bash
docker build -f Dockerfile.app -t abalone-api .
docker run -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 abalone-api
```

- API documentation: http://localhost:8000/docs
- Prefect UI (training runs and the `weekly-retraining` deployment): http://localhost:4200

To retrain on an updated dataset without rebuilding the image, mount the data folder: add `-v "$(pwd)/data:/app/data"` to `docker run`.

#### Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sex": "M", "length": 0.455, "diameter": 0.365, "height": 0.095,
       "whole_weight": 0.514, "shucked_weight": 0.2245,
       "viscera_weight": 0.101, "shell_weight": 0.15}'
```

Response (`201`):

```json
{"predicted_age": 10.67}
```

Use port `8001` instead of `8000` when the API runs locally.

## Project Structure

```
├── bin/run_services.sh        # Container entrypoint: Prefect server, training, deployment, API
├── Dockerfile.app             # Docker image of the API and the Prefect server
├── notebooks/                 # EDA and first model (tracked with MLflow)
└── src/
    ├── modelling/             # Prefect training flow
    │   ├── main.py            # Flow: load, clean, split, train, evaluate, save
    │   ├── deployment.py      # Weekly retraining deployment
    │   ├── preprocessing.py   # Cleaning, target, train/test split, preprocessor
    │   ├── training.py        # Pipeline, training and evaluation
    │   ├── predicting.py      # Model loading and prediction
    │   └── utils.py           # Pickle helpers
    └── web_service/           # FastAPI app
        ├── main.py            # Routes
        ├── lib/models.py      # Pydantic request and response schemas
        ├── lib/inference.py   # Model loading (reloaded on change) and prediction
        └── local_objects/     # Trained model (model.pkl, git-ignored)
```

## Development Tips

### Managing Dependencies

Dependencies are managed with uv (`pyproject.toml` and `uv.lock`). Add a package with:

```bash
uv add <package>
```

### MLflow Tracking

- **Issue**: with MLflow 3.x (locked at 3.17.2), the file-based tracking store (`./mlruns`, the default) is in maintenance mode and raises an error, so the usual `mlflow ui --backend-store-uri mlruns` command fails.
- **Resolution**: runs are stored in a local SQLite database, `mlflow.db` at the project root, and artifacts (models) go to the project-level `mlruns/` folder. The notebook sets both.
- **View the runs** from the project root:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

- `mlflow.db`, `mlruns/` and `data/` are git-ignored and must not be pushed.

### Code Quality

- The pre-commit hooks (ruff lint and format, nbstripout, file checks) run on every commit, and on every push and pull request in the CI.
- Run them on all files with `uv run pre-commit run --all-files`.
