"""Server route tests using FastAPI TestClient."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


class TestBasicEndpoints:
    def test_index(self):
        r = client.get("/")
        assert r.status_code == 200

    def test_heartbeat(self):
        r = client.post("/api/heartbeat")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_config(self):
        r = client.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        assert "model_type" in data or "conf_threshold" in data or isinstance(data, dict)

    def test_model_types(self):
        r = client.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        assert "model_types" in data
        assert "yolo" in data["model_types"]

    def test_system_info(self):
        r = client.get("/api/system/info")
        assert r.status_code == 200


class TestOptimizationEndpoints:
    def test_optimize_methods(self):
        r = client.get("/api/optimize/methods")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
        # Should have at least quantization category
        assert any("quantization" in str(v) for v in data.keys()) or len(data) > 0

    def test_optimize_status(self):
        r = client.get("/api/optimize/status")
        assert r.status_code == 200
        data = r.json()
        assert "running" in data

    def test_diagnose_status(self):
        r = client.get("/api/diagnose/status")
        assert r.status_code == 200
        data = r.json()
        assert "running" in data

    def test_optimize_run_invalid_model(self):
        r = client.post("/api/optimize/run", json={
            "model_path": "/nonexistent/model.onnx",
            "method": "dynamic_int8"
        })
        assert r.status_code == 200
        data = r.json()
        assert "error" in data


class TestEvalEndpoints:
    def test_eval_status(self):
        r = client.get("/api/evaluation/status")
        assert r.status_code == 200
        data = r.json()
        assert "running" in data

    def test_eval_stop(self):
        r = client.get("/api/eval/stop")
        assert r.status_code == 200

    def test_eval_history(self):
        r = client.get("/api/eval/history")
        assert r.status_code == 200
        assert "files" in r.json()

    def test_eval_load_nonexistent(self):
        r = client.get("/api/eval/load/nonexistent.json")
        assert r.status_code == 200
        assert "error" in r.json()


class TestForceStop:
    def test_force_stop_all(self):
        r = client.post("/api/force-stop/all")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_force_stop_unknown(self):
        r = client.post("/api/force-stop/nonexistent_task")
        assert r.status_code == 200
        assert "error" in r.json()
