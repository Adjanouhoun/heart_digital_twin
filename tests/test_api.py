"""Tests D4.1 — API REST + WebSocket."""
import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.api.cdt_endpoints import app, sessions


@pytest.fixture
def client():
    sessions.clear()
    return TestClient(app)


class TestAPIHealth:
    def test_health(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_version(self, client):
        r = client.get("/api/v1/health")
        assert "version" in r.json()


class TestTwinCreate:
    def test_create_twin(self, client):
        r = client.post("/api/v1/twin/create", json={"patient_id": "patient001"})
        assert r.status_code == 200
        assert "twin_id" in r.json()
        assert r.json()["state"] == "created"

    def test_create_twin_returns_patient(self, client):
        r = client.post("/api/v1/twin/create", json={"patient_id": "patient003"})
        assert r.json()["patient_id"] == "patient003"


class TestTwinState:
    def test_get_state(self, client):
        r = client.post("/api/v1/twin/create", json={"patient_id": "patient001"})
        tid = r.json()["twin_id"]
        r2 = client.get(f"/api/v1/twin/{tid}/state")
        assert r2.status_code == 200
        assert r2.json()["state"] == "created"

    def test_state_not_found(self, client):
        r = client.get("/api/v1/twin/nonexistent/state")
        assert r.status_code == 404


class TestTwinSimulate:
    def test_simulate_returns_results(self, client):
        r = client.post("/api/v1/twin/create", json={"patient_id": "patient001"})
        tid = r.json()["twin_id"]
        r2 = client.post("/api/v1/twin/simulate", json={"twin_id": tid})
        assert r2.status_code == 200
        data = r2.json()
        assert data["state"] == "completed"
        assert "ef_pct" in data["results"]

    def test_simulate_not_found(self, client):
        r = client.post("/api/v1/twin/simulate", json={"twin_id": "fake"})
        assert r.status_code == 404

    def test_results_after_simulate(self, client):
        r = client.post("/api/v1/twin/create", json={"patient_id": "patient001"})
        tid = r.json()["twin_id"]
        client.post("/api/v1/twin/simulate", json={"twin_id": tid})
        r3 = client.get(f"/api/v1/twin/{tid}/results")
        assert r3.status_code == 200


class TestWebSocket:
    def test_ws_ping(self, client):
        r = client.post("/api/v1/twin/create", json={"patient_id": "patient001"})
        tid = r.json()["twin_id"]
        with client.websocket_connect(f"/api/v1/twin/{tid}/stream") as ws:
            ws.send_json({"action": "ping"})
            data = ws.receive_json()
            assert data["type"] == "pong"

    def test_ws_get_state(self, client):
        r = client.post("/api/v1/twin/create", json={"patient_id": "patient001"})
        tid = r.json()["twin_id"]
        with client.websocket_connect(f"/api/v1/twin/{tid}/stream") as ws:
            ws.send_json({"action": "get_state"})
            data = ws.receive_json()
            assert data["type"] == "state"
            assert data["state"] == "created"

    def test_ws_not_found(self, client):
        with client.websocket_connect("/api/v1/twin/fake/stream") as ws:
            data = ws.receive_json()
            assert "error" in data
