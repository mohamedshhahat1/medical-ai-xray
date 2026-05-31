"""Integration tests for the API."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from fastapi.testclient import TestClient
from app import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "Medical AI" in data["name"]
    print("✓ Root endpoint works")


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    print("✓ Health endpoint works")


def test_classes_endpoint():
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert len(data["classes"]) > 0
    print("✓ Classes endpoint works")


if __name__ == "__main__":
    test_root_endpoint()
    test_health_endpoint()
    test_classes_endpoint()
    print("\n✅ All API tests passed!")
