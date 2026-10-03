# ML Deploy Demo

**Status: work in progress.** This is an initial technical demo.

An educational example of deploying a scikit-learn classifier as a FastAPI service, with Docker, automated tests and Prometheus metrics.

## Overview

The project uses the breast-cancer dataset bundled with scikit-learn to demonstrate a complete training-to-serving workflow. `train.py` fits a pipeline containing a standard scaler and logistic regression, then saves the pipeline together with the feature and target names.

The example focuses on model packaging, validation of API inputs and automated checks.

## Training and evaluation

Training uses a reproducible, stratified 80/20 train/test split. The script prints accuracy, precision, recall, F1 and ROC AUC, and saves the measured values in `model/metrics.json`. The serving artifact is written to `model/model.pkl`; preprocessing is stored inside the pipeline used by the API.

## Run locally

Use Python 3.11, matching the Docker image and CI environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
uvicorn app:app --reload
```

Open [the interactive API documentation](http://127.0.0.1:8000/docs) to explore the endpoints. On Windows, activate the virtual environment using the activation command for your shell.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Check service status and whether the model is loaded |
| `POST /predict` | Classify an input containing 30 numeric features |
| `GET /metrics` | Expose Prometheus request counters and prediction latency |

The prediction request has a `features` array in the order returned by `sklearn.datasets.load_breast_cancer().feature_names`. The response contains the class index, its label and its probability. Requests with the wrong feature count receive HTTP 422.

## Run with Docker

```bash
docker build -t ml-deploy-demo .
docker run --rm -p 8000:8000 ml-deploy-demo
```

The image trains the model during the build and starts the API on port 8000 by default.

## Tests and CI

```bash
python -m pip install -r requirements-dev.txt
python train.py
pytest tests/ -v
```

Tests cover health checks, prediction responses, invalid feature counts and the metrics endpoint. GitHub Actions installs the dependencies, trains the model and runs these tests on pushes and pull requests.

## Repository contents

- [`train.py`](train.py): training, evaluation and artifact creation.
- [`app.py`](app.py): API endpoints and monitoring.
- [`Dockerfile`](Dockerfile): container build and startup.
- [`tests/test_api.py`](tests/test_api.py): API tests.
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml): automated checks.
- [`requirements.txt`](requirements.txt) and [`requirements-dev.txt`](requirements-dev.txt): runtime and test dependencies.

## Author

Antonio Verde
