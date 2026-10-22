#!/bin/bash
# Entrypoint of the Docker container (see Dockerfile.app), run from the project root:
# 1. start the Prefect server (UI + API) in the background,
# 2. train a first model so that the API has a `model.pkl` to load,
# 3. serve the weekly retraining deployment in the background,
# 4. start the FastAPI app in the foreground.

set -e

prefect server start --host 0.0.0.0 --port 4201 &

echo "Waiting for the Prefect server to be ready..."
until python -c "import urllib.request; urllib.request.urlopen('${PREFECT_API_URL}/health')" 2>/dev/null; do
    sleep 1
done

python -m src.modelling.main data/abalone.csv

python -m src.modelling.deployment &

uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001
