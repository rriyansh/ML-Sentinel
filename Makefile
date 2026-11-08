.PHONY: setup test run-serving feast-apply

setup:
	pip install -e .[dev]

test:
	pytest tests/ -v --cov=src

feast-apply:
	cd feast_repo && feast apply

run-serving:
	uvicorn serving.app:app --reload

run-airflow:
	airflow standalone
