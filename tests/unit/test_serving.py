from fastapi.testclient import TestClient
from serving.app import app

client = TestClient(app)

def test_liveness():
    response = client.get("/health/liveness")
    assert response.status_code == 200

def test_readiness_not_ready():
    # Model is not loaded in test environment
    response = client.get("/health/readiness")
    assert response.status_code == 503
