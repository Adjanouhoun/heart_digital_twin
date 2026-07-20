"""Tests D4.1 — API REST + WebSocket."""
import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.api.cdt_endpoints import app, sessions


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Le test API vérifie le contrat HTTP, pas les solveurs scientifiques.
    # Il fournit donc explicitement un maillage minimal et un résultat simulé.
    (tmp_path / "patient001.pts").write_text(
        "4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
    )
    (tmp_path / "patient001.elem").write_text("1\nTt 0 1 2 3 1\n")
    (tmp_path / "patient001_fibers.lon").write_text(
        "1\n1 0 0\n1 0 0\n1 0 0\n1 0 0\n"
    )
    monkeypatch.setenv("CDT_MESH_DIR", str(tmp_path))

    class FakeResult:
        def to_doe_row(self):
            return {
                "ef_pct": 55.0,
                "edv_mL": 120.0,
                "esv_mL": 54.0,
                "p_systolic_mmHg": 120.0,
                "p_diastolic_mmHg": 80.0,
                "cv_ms": 0.6,
                "benchmark_passed": True,
            }

    class FakeCoupledSolver:
        def simulate(self, *args, **kwargs):
            return FakeResult()

    monkeypatch.setattr(
        "app.solver.coupled_solver.CoupledSolver", FakeCoupledSolver
    )
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
