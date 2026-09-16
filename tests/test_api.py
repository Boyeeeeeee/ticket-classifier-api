"""
Tests for the API layer.

Note: predict-endpoint tests are marked to skip until a trained model exists
at the path in app/model.py. The health check and validation tests run
immediately and don't need a trained model.
"""
import os

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import MODEL_DIR

client = TestClient(app)

MODEL_AVAILABLE = os.path.isdir(MODEL_DIR)


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


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Trained model not available yet")
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
