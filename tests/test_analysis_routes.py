"""Tests for /api/analysis/* routes (track: analysis).

Focus: class-aware FP/FN matching (ANLY-01) and path-safety envelopes
(ANLY-03). Model loading and inference are mocked so the tests stay fast
and don't need real .onnx weights.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server.analysis_routes as ar
from server import app
from server.state import error_analysis_state

client = TestClient(app)


class _FakeModelInfo:
    def __init__(self, names):
        self.names = names


class _FakeResult:
    """Minimal stand-in for run_inference's return value."""

    def __init__(self, boxes, scores, class_ids):
        self.boxes = boxes
        self.scores = scores
        self.class_ids = class_ids
        self.infer_ms = 1.0


def _wait_until_done(status_url: str, timeout_s: float = 10.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        data = client.get(status_url).json()
        if not data.get("running"):
            return data
        time.sleep(0.02)
    raise AssertionError(f"worker did not finish within {timeout_s}s")


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """One image + a GT label dir + a dummy model file, with mocks wired up.

    The single GT box is class 0 covering the whole frame. The caller
    decides what the model predicts via `set_pred`.
    """
    img_dir = tmp_path / "imgs"
    lbl_dir = tmp_path / "lbls"
    img_dir.mkdir()
    lbl_dir.mkdir()
    img_path = img_dir / "a.jpg"
    img_path.write_bytes(b"fake")  # imread is mocked; content irrelevant
    model_path = tmp_path / "m.onnx"
    model_path.write_bytes(b"fake")

    # GT: class 0, full-frame box (YOLO normalized cx cy w h).
    (lbl_dir / "a.txt").write_text("0 0.5 0.5 1.0 1.0\n")

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    monkeypatch.setattr(ar, "imread", lambda _p: frame)
    monkeypatch.setattr(ar, "glob_images", lambda _d: [str(img_path)])

    import core.model_loader
    monkeypatch.setattr(
        core.model_loader, "load_model",
        lambda *a, **k: _FakeModelInfo({0: "cat", 1: "dog"}),
    )

    state = {"result": None}

    def _set_pred(class_id: int):
        # Predicted box fully overlaps the GT box (IoU = 1.0).
        res = _FakeResult(
            boxes=[np.array([0.0, 0.0, 100.0, 100.0])],
            scores=[0.9],
            class_ids=[class_id],
        )
        import core.inference
        monkeypatch.setattr(core.inference, "run_inference", lambda *a, **k: res)

    return {
        "img_dir": str(img_dir),
        "lbl_dir": str(lbl_dir),
        "model_path": str(model_path),
        "set_pred": _set_pred,
    }


def _run_error_analysis(ds) -> dict:
    error_analysis_state.update(running=False)  # ensure no stale "Already running"
    r = client.post("/api/analysis/error-analysis", json={
        "model_path": ds["model_path"],
        "img_dir": ds["img_dir"],
        "label_dir": ds["lbl_dir"],
        "iou_threshold": 0.5,
        "conf": 0.25,
    })
    assert r.status_code == 200, r.text
    assert r.json().get("ok") is True
    data = _wait_until_done("/api/analysis/error-analysis/status")
    return data["results"]


def test_same_class_overlap_is_true_positive(dataset):
    """GT class 0, pred class 0, full overlap -> no FP, no FN."""
    dataset["set_pred"](0)
    results = _run_error_analysis(dataset)
    assert results["fp"]["count"] == 0
    assert results["fn"]["count"] == 0


def test_cross_class_overlap_is_fp_plus_fn(dataset):
    """GT class 0, pred class 1, full overlap -> 1 FP + 1 FN (misclass)."""
    dataset["set_pred"](1)
    results = _run_error_analysis(dataset)
    assert results["fp"]["count"] == 1
    assert results["fn"]["count"] == 1


class TestPathSafetyEnvelopes:
    def test_error_analysis_rejects_missing_dir(self):
        error_analysis_state.update(running=False)
        r = client.post("/api/analysis/error-analysis", json={
            "model_path": "does_not_exist.onnx",
            "img_dir": "/no/such/dir",
            "label_dir": "/no/such/dir",
        })
        assert r.status_code == 400
        body = r.json()
        assert body["code"].startswith("PATH_")

    def test_model_compare_rejects_bad_model_ext(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        bad_model = tmp_path / "m.txt"
        bad_model.write_bytes(b"x")
        r = client.post("/api/analysis/model-compare", json={
            "model_a": str(bad_model),
            "model_b": str(bad_model),
            "img_dir": str(img_dir),
        })
        assert r.status_code == 400
        assert r.json()["code"] == "PATH_BAD_EXT"
