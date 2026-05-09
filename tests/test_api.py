from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_endpoint():
    response = client.get("/models")

    assert response.status_code == 200

    data = response.json()

    assert "available_models" in data
    assert len(data["available_models"]) > 0