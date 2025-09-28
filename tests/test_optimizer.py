import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/api/v2/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "dfs-api"}


def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "DFS API Service"
    assert data["status"] == "running"
    assert data["version"] == "1.0.0"


def test_optimizer_endpoint_missing_data():
    """Test optimizer endpoint with missing required data"""
    response = client.post("/api/v2/optimize", json={})
    assert response.status_code == 422  # Validation error


def test_optimizer_endpoint_minimal_data():
    """Test optimizer endpoint with minimal valid data"""
    minimal_request = {
        "players": [
            {
                "player_id": 1,
                "slate_player_id": 100,
                "name": "Test Player",
                "salary": 5000,
                "value": 10.5,
                "roster_slots": ["QB"],
                "team_abbr": "TEST",
                "position_abbr": "QB"
            }
        ],
        "options": {
            "max_per_team": 9
        },
        "game": {
            "roster_slots": ["QB"],
            "max_budget": 50000,
            "multipliers": {"QB": 1.0}
        },
        "num_solutions": 1
    }

    response = client.post("/api/v2/optimize", json=minimal_request)
    assert response.status_code == 200

    data = response.json()
    assert "data" in data
    assert "status" in data
    assert data["status"] == 0
    assert "execution_time" in data
    assert "num_lineups" in data
    assert isinstance(data["data"], list)
    assert "datatype" in data
    assert data["datatype"] == "optimizer"