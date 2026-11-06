.PHONY: setup test run train serve drift clean

setup:
	pip install -r requirements.txt
	pip install pytest pytest-cov

test:
	pytest tests/unit tests/integration -v

train:
	python mlflow/train.py

serve:
	uvicorn serving.app:app --reload --host 0.0.0.0 --port 8000

drift:
	python monitoring/drift_check.py

clean:
	rm -rf __pycache__ .pytest_cache mlruns logs monitoring_reports
