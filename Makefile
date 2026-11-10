.PHONY: help install install-dev setup-env sample-data feast-apply init-db \
        train drift-check test lint type-check security serve docker-build \
        docker-up docker-down clean

PY ?= python
PIP ?= $(PY) -m pip

help:
	@echo "ML-Sentinel — Available targets:"
	@echo "  install         Install runtime dependencies"
	@echo "  install-dev     Install dev dependencies (pytest, ruff, mypy, bandit)"
	@echo "  setup-env       Copy .env.example to .env (if missing)"
	@echo "  sample-data     Generate synthetic Feast sample data"
	@echo "  feast-apply     Apply Feast feature definitions"
	@echo "  init-db         Create SQLAlchemy tables in the configured DB"
	@echo "  train           Run the training pipeline end-to-end"
	@echo "  drift-check     Run drift detection against reference data"
	@echo "  test            Run the test suite with coverage"
	@echo "  lint            Run ruff linter"
	@echo "  type-check      Run mypy static typing"
	@echo "  security        Run bandit + pip-audit security scans"
	@echo "  serve           Start FastAPI serving layer locally"
	@echo "  docker-build    Build all docker images"
	@echo "  docker-up       Start the full stack via docker-compose"
	@echo "  docker-down     Stop and remove the stack"
	@echo "  clean           Remove caches and transient artifacts"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -e ".[dev]"
	$(PIP) install pre-commit

setup-env:
	@test -f .env || (cp .env.example .env && echo ".env created — edit secrets before running")

sample-data:
	$(PY) scripts/generate_sample_data.py

feast-apply: sample-data
	cd feast_repo && feast apply

init-db:
	$(PY) scripts/init_db.py

train: feast-apply
	$(PY) -m mlflow.train

drift-check:
	$(PY) -c "from monitoring.drift_check import run_manual_drift_check; run_manual_drift_check()"

test:
	$(PY) -m pytest tests/ -v

lint:
	$(PY) -m ruff check .

type-check:
	$(PY) -m mypy src serving monitoring mlflow

security:
	$(PY) -m bandit -r src serving monitoring mlflow -ll
	$(PY) -m pip-audit -r requirements.txt

serve:
	$(PY) -m uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -f docker/Dockerfile.serving -t ml-sentinel-serving:2.0.0 .
	docker build -f docker/Dockerfile.train   -t ml-sentinel-train:2.0.0 .
	docker build -f docker/Dockerfile.airflow -t ml-sentinel-airflow:2.0.0 .

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage mlruns monitoring/reports
