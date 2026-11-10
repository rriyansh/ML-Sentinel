



  # ML-Sentinel

  ### Automated, Self-Healing MLOps Platform with Real-Time Drift Detection

  [![CI/CD](https://img.shields.io/github/actions/workflow/status/your-org/ml-sentinel/ci-cd.yaml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white&label=CI/CD)](https://github.com/your-org/ml-sentinel/actions)
  [![Security](https://img.shields.io/github/actions/workflow/status/your-org/ml-sentinel/security-scan.yaml?branch=main&style=for-the-badge&logo=githubsecurity&logoColor=white&label=Security)](https://github.com/your-org/ml-sentinel/actions)
  [![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

  <p align="center">
    <a href="#quick-start">Quick Start</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#features">Features</a> •
    <a href="#serving-the-model">API Docs</a> •
    <a href="#observability">Observability</a>
  </p>

</div>

---

<div align="center">

### Tech Stack & Tooling

![Feast](https://img.shields.io/badge/Feast-000000?style=flat-square&logo=feast&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![Evidently AI](https://img.shields.io/badge/Evidently_AI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Apache Airflow](https://img.shields.io/badge/Apache_Airflow-017CEE?style=flat-square&logo=apacheairflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=flat-square&logo=prometheus&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Ruff](https://img.shields.io/badge/Code_Style-Ruff-000000?style=flat-square&logo=python&logoColor=white)

</div>

---

**ML-Sentinel** is a production-grade, self-healing MLOps platform that automates the entire machine learning lifecycle — from feature management to drift-aware retraining and secured model serving. Built with **Feast**, **Evidently**, **MLflow**, **Airflow**, and **FastAPI**.

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Project Layout](#project-layout)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Serving the Model](#serving-the-model)
- [Security Hardening](#security-hardening)
- [Observability](#observability)
- [Docker Deployment](#docker-deployment)
- [Testing & Quality](#testing--quality)
- [CI/CD Workflow](#cicd-workflow)
- [Contributing](#contributing)
- [License](#license)

---

## Architecture


```

┌─────────────────────────────────────────────────────────────────────────────┐
│                              ML-Sentinel Platform                           │
│                                                                             │
│   ┌──────────┐    ┌─────────────┐    ┌───────────────┐    ┌──────────────┐  │
│   │   Raw    │───▶│   Feast     │───▶│   MLflow      │───▶│   Model      │  │
│   │   Data   │    │   Feature   │    │   Training +  │    │   Registry   │  │
│   │   Lake   │    │   Store     │    │   Lineage     │    │  (Champion/  │  │
│   └──────────┘    └─────────────┘    └───────────────┘    │  Challenger) │  │
│        │                 │                   │            └──────┬───────┘  │
│        │                 │                   │                   │          │
│        ▼                 ▼                   ▼                   ▼          │
│   ┌────────────────────────────────────────────────────────┐   ┌─────────┐  │
│   │                 Apache Airflow (DAG)                   │   │   API   │  │
│   │  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐  │   │ FastAPI │  │
│   │  │  Drift  │─▶│ Retrain  │─▶│ Quality │─▶│  Promote │──┼──▶│ Serving │  │
│   │  │  Detect │  │          │  │   Gate  │  │  to Prod │  │   └────┬────┘  │
│   │  └─────────┘  └──────────┘  └─────────┘  └──────────┘  │        │       │
│   └────────────────────────────────────────────────────────┘        │       │
│                                                                     ▼       │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │  Observability: Prometheus Metrics · Structured Logs (JSON)          │  │
│   │          Drift Audit Trail (PostgreSQL) · Alerting                   │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

## Features

| Category | Feature | Tech / Implementation |
|---|---|---|
| **Feature Store** | Consistent features across train & serve | **Feast** + MLflow Lineage |
| **Experimentation** | Experiment tracking & Model Registry | **MLflow** (Champion vs Challenger) |
| **Drift Monitoring**| Real-time statistical drift detection | **Evidently AI** (PSI, KS tests) |
| **Orchestration** | Automated retrain pipelines on drift | **Apache Airflow** TaskFlow API |
| **Quality Gate** | Auto-promotion policy comparison | Metric-based Gatekeeper |
| **API Security** | Auth, Rate-limiting & Hardening | **JWT**, API-Keys, SlowAPI, CORS |
| **Metrics & Logs** | Full system observability | **Prometheus**, `structlog`, PostgreSQL |
| **Deployment** | Hardened container runtime | Docker Multi-stage, non-root user |

---

## Quick Start

### 1. Prerequisites

* **Python**: `3.10+`
* **Containers**: Docker & Docker Compose
* **Build Tool**: `make` (recommended)

### 2. Local Setup (Development)

```bash
# Clone the repository
git clone [https://github.com/your-org/ml-sentinel.git](https://github.com/your-org/ml-sentinel.git)
cd ml-sentinel

# Initialize environment
make setup-env

# Install & initialize pipeline
make install-dev
make sample-data
make feast-apply
make init-db
make train
make drift-check
make serve

```

> **API Access**: `http://localhost:8000` | **Swagger Docs**: `http://localhost:8000/docs`

### 3. Full Stack with Docker

```bash
make docker-up

```

#### Exposed Dashboards & Services

* **FastAPI Serving Engine**: `http://localhost:8000`
* **MLflow Tracking Server**: `http://localhost:5000`
* **Airflow Dashboard**: `http://localhost:8080`
* **Prometheus Metrics**: `http://localhost:8000/metrics`

---

## Project Layout

```
ml-sentinel/
├── airflow/dags/          # Airflow DAGs (Drift -> Retrain -> Gate -> Promote)
├── config/                # Pipeline, drift & logging YAML/JSON configs
├── docker/                # Multi-stage Dockerfiles (Serving, Training, Airflow)
├── feast_repo/            # Feature definitions & store specs
├── mlflow/                # Model training code & lineage logic
├── monitoring/            # Evidently drift detection suites
├── scripts/               # Setup, seeding & utility scripts
├── serving/               # FastAPI endpoint definitions & middleware
├── src/ml_sentinel/       # Shared core library (Security, DB, Metrics)
├── tests/                 # Unit & Integration test suite
├── docker-compose.yaml    # Infrastructure composition
├── Makefile               # Command automation
├── pyproject.toml         # Dependencies & tool settings
└── README.md

```

---

## Configuration

Environment variables drive all execution settings (refer to `.env.example`).

| Config File | Description |
| --- | --- |
| `config/drift_config.yaml` | Statistical thresholds, monitored features, drift test algorithms |
| `config/model_promotion_policy.json` | Champion/Challenger rules for automated promotion |
| `config/logging.yaml` | Structured JSON log formats (`structlog`) |

---

## Running the Pipeline

### Manual Execution

```bash
# Generate data and register features
make sample-data && make feast-apply

# Train and log model to MLflow
make train

# Run drift inspection
make drift-check

```

### Automated Orchestration (Airflow)

1. Launch Airflow UI at `http://localhost:8080`.
2. Login with credentials specified in your `.env`.
3. Enable and trigger `ml_sentinel_production_pipeline` DAG:
* **Fetch Data** (Reference vs Current)
* **Evaluate Drift** via Evidently
* **Trigger Retraining** (if drift detected)
* **Evaluate Quality Gate** (Champion vs Challenger)
* **Promote Model** to `Production` stage



---

## Serving the Model

### 1. Launch Server

```bash
make serve

```

### 2. Generate Auth Token

```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"api_key":"dev-key-abc123"}'

```

### 3. Request Inference

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer <YOUR_JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"conv_rate": 0.7, "acc_rate": 0.9}'

```

**Response Payload:**

```json
{
  "predictions": [42.0],
  "model_name": "MLSentinelModel",
  "model_version": "3",
  "request_id": "b3b1c2e6-817a-4c28-98e2-04fa3a90890d",
  "latency_ms": 12.4
}

```

### Health Check Matrix

| Endpoint | Target System | Purpose |
| --- | --- | --- |
| `GET /health/liveness` | Kubernetes Probe | Pod vitality status |
| `GET /health/readiness` | Kubernetes Probe | Verifies DB connection & loaded model |
| `GET /health/startup` | Kubernetes Probe | Verifies container boot success |
| `GET /metrics` | Prometheus Scraping | Exposes system & performance metrics |

---

## Security Hardening

* **JWT & API Key Auth**: HS256 signed bearer tokens with configurable expiration.
* **Key Rotation Support**: Zero-downtime key rotation using comma-separated API keys.
* **Rate Limiting**: Built-in per-IP rate limit protection via `SlowAPI`.
* **HTTP Headers**: Enforces `X-Frame-Options`, `X-Content-Type-Options`, `HSTS`, and `CORS`.
* **Security Audit**: Scanned using `bandit` and `pip-audit` inside CI/CD workflows.
* **Container Safety**: Runs under non-root user (`mlsentinel`) with read-only root filesystems.

---

## Observability

Prometheus metrics available at `/metrics`:

* `ml_sentinel_predictions_total{status}` — Inference counter
* `ml_sentinel_prediction_latency_seconds` — Latency distribution
* `ml_sentinel_model_loaded` — Model load health state (`1` / `0`)
* `ml_sentinel_http_requests_total` — API request metrics
* `ml_sentinel_drift_events_total` — Total drift alerts triggered

---

## Testing & Quality

```bash
make test        # Run unit/integration tests with coverage
make lint        # Code formatting check with Ruff
make type-check  # Type verification via MyPy
make security    # Static security check (Bandit + Pip-audit)

```

> **Quality Standard**: Test coverage is enforced at **>= 70%** via `pyproject.toml`.

---

## Contributing

1. Fork the repo and create your branch: `git checkout -b feat/my-new-feature`
2. Set up environment: `make install-dev`
3. Verify test compliance: `make test && make lint`
4. Submit a Pull Request

Please follow [Conventional Commits](https://www.conventionalcommits.org/) standards.

---

## License

Distributed under the **MIT License**. See `LICENSE` for more information.



```

```
