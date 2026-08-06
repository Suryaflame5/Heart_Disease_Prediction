import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from main import app

client = TestClient(app)

def test_read_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "best_params" in data or "item_count" in data or "sparsity" in data or "silhouette_score" in data

def test_prediction_history():
    response = client.get("/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
