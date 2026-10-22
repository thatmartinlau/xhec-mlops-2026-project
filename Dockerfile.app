# Image running the prediction API (port 8001) and the Prefect server (port 4201).
# The model is trained when the container starts (see bin/run_services.sh).
#
# Build and run from the project root, after downloading the data to data/abalone.csv:
#   docker build -f Dockerfile.app -t abalone-api .
#   docker run -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 abalone-api

FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /bin/uv

WORKDIR /app

# Install the runtime dependencies only (no dev group: notebooks, MLflow, linters),
# first, so this layer is cached when only the code changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache --no-dev

COPY src/ src/
COPY bin/ bin/
COPY data/ data/

# Use the project virtual environment by default
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    # Training flows and deployments report to the Prefect server of the container
    PREFECT_API_URL=http://127.0.0.1:4201/api \
    # The Prefect UI runs in the browser of the host, where port 4201 is bound to 4200
    PREFECT_UI_API_URL=http://localhost:4200/api

EXPOSE 8001 4201

CMD ["bash", "bin/run_services.sh"]
