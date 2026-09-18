
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import get_classifier

client = TestClient(app)


def _model_available() -> bool:
    
    try:
        get_classifier()
        return True
    except Exception:
        return False


MODEL_AVAILABLE = _model_available()


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rejects_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Could not load model from Hugging Face Hub")
def test_predict_returns_ranked_predictions():
    response = client.post(
        "/predict",
        json={"text": "My credit card company charged me twice for the same transaction"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "top_prediction" in body
    assert "all_predictions" in body
    assert len(body["all_predictions"]) > 0
    confidences = [p["confidence"] for p in body["all_predictions"]]
    assert confidences == sorted(confidences, reverse=True)


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Could not load model from Hugging Face Hub")
def test_predict_top_prediction_matches_expected_category():
    """Sanity-check the model actually predicts sensible categories, not
    just that the response is well-formed."""
    response = client.post(
        "/predict",
        json={"text": "My credit card company charged me twice for the same transaction"},
    )
    assert response.status_code == 200
    assert response.json()["top_prediction"]["label"] == "Credit card or prepaid card"