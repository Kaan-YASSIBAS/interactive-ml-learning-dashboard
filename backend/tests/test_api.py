from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Interactive ML Learning Dashboard API is running"
    }


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["project"] == "Interactive ML Learning Dashboard"
    assert data["status"] == "running"


def test_decision_tree_endpoint():
    response = client.get("/decision-tree?max_depth=3")

    assert response.status_code == 200

    data = response.json()

    assert data["algorithm"] == "Decision Tree"
    assert data["max_depth"] == 3
    assert "train_accuracy" in data
    assert "test_accuracy" in data
    assert "sample" in data
    assert "rules" in data
