.PHONY: help install install-dev setup-env sample-data feast-apply init-db \
        train drift-check explain fairness model-card test test-unit test-integration \
        test-load lint type-check arch-check security sbom serve docker-build \
        docker-up docker-down k8s-apply helm-install clean

PY ?= python
PIP ?= $(PY) -m pip

help:
	@echo "ML-Sentinel v3.0 targets:"
	@echo "  install          Install runtime deps"
	@echo "  install-dev      Install dev deps"
	@echo "  setup-env        Copy .env.example -> .env"
	@echo "  sample-data      Generate synthetic data"
	@echo "  feast-apply      Apply Feast definitions"
	@echo "  init-db          Create DB tables"
	@echo "  train            Train + register model"
	@echo "  drift-check      Run drift detection"
	@echo "  explain          Generate SHAP explanations"
	@echo "  fairness         Run fairness analysis"
	@echo "  model-card       Generate model card"
	@echo "  test             Run all tests"
	@echo "  test-unit        Unit tests only"
	@echo "  test-integration Integration tests"
	@echo "  test-load        Load tests (Locust)"
	@echo "  lint             Ruff"
	@echo "  type-check       Mypy"
	@echo "  arch-check       Import linter (hexagonal rules)"
	@echo "  security         bandit + pip-audit"
	@echo "  sbom             Generate SBOM"
	@echo "  serve            Run FastAPI locally"
	@echo "  docker-build     Build all images"
	@echo "  docker-up        Full docker-compose stack"
	@echo "  docker-down      Stop stack"
	@echo "  k8s-apply        Apply raw k8s manifests"
	@echo "  helm-install     Deploy via Helm"
	@echo "  clean            Remove caches"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -e ".[dev]"

setup-env:
	@test -f .env || (cp .env.example .env && echo "Edit .env before running")

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

explain:
	$(PY) -c "from monitoring.explainability import run_shap_explanations; run_shap_explanations()"

fairness:
	$(PY) -c "from monitoring.fairness import run_fairness_check; run_fairness_check()"

model-card:
	$(PY) scripts/generate_model_card.py

test:
	$(PY) -m pytest tests/ -v

test-unit:
	$(PY) -m pytest tests/unit -v

test-integration:
	$(PY) -m pytest tests/integration -v -m integration

test-load:
	$(PY) -m locust -f tests/load/locustfile.py --headless -u 50 -r 5 -t 30s --host http://localhost:8000

lint:
	$(PY) -m ruff check .

type-check:
	$(PY) -m mypy src serving monitoring mlflow

arch-check:
	$(PY) -m importlinter.cli lint

security:
	$(PY) -m bandit -r src serving monitoring mlflow -ll
	$(PY) -m pip-audit -r requirements.txt

sbom:
	$(PY) -m cyclonedx_py requirements requirements.txt -o sbom.json

serve:
	$(PY) -m uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -f docker/Dockerfile.serving -t ml-sentinel-serving:3.0.0 .
	docker build -f docker/Dockerfile.train -t ml-sentinel-train:3.0.0 .
	docker build -f docker/Dockerfile.airflow -t ml-sentinel-airflow:3.0.0 .

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down -v

k8s-apply:
	kubectl apply -f deploy/kubernetes/

helm-install:
	helm upgrade --install ml-sentinel deploy/helm/ml-sentinel -f deploy/helm/ml-sentinel/values.production.yaml --namespace ml-sentinel --create-namespace

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache .hypothesis htmlcov .coverage mlruns monitoring/reports
