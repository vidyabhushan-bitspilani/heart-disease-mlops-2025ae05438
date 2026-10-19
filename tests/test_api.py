
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

VALID_INPUT = {
    "age": 55,
    "trestbps": 130,
    "chol": 240,
    "thalach": 150,
    "oldpeak": 1.0,
    "sex": 1,
    "cp": 4,
    "fbs": 0,
    "restecg": 0,
    "exang": 0,
    "slope": 2,
    "ca": 0,
    "thal": 3,
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_valid_input():
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in (0, 1)
    assert 0 <= result["probability_disease"] <= 1
    assert "disclaimer" in result


def test_predict_missing_feature():
    payload = VALID_INPUT.copy()
    payload.pop("age")

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_value():
    payload = VALID_INPUT.copy()
    payload["age"] = 500

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
