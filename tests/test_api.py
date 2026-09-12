from fastapi.testclient import TestClient

from code.deployment.api.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_endpoint_and_validation():
    passenger = {"pclass":1,"sex":"female","age":35,"sibsp":1,"parch":0,"fare":75,"embarked":"C"}
    with TestClient(app) as client:
        response = client.post("/predict", json=passenger)
        invalid = client.post("/predict", json={**passenger, "age": -1})
    assert response.status_code == 200
    assert response.json()["predicted_class"] in (0, 1)
    assert invalid.status_code == 422
