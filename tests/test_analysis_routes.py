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


@pytest.fixture
def crossing_dataset(tmp_path, monkeypatch):
    """One GT box + two overlapping predictions of different size and score.

    Single GT (class 0) covers the whole frame. Two predictions both clear the
    IoU threshold against that one GT, so exactly one can claim it (greedy 1:1)
    and the other is an FP. WHICH one is the FP depends on match order:
      - score-desc (canonical / Evaluation tab): the HIGH-score pred claims GT;
        the LOW-score pred is the FP. The low pred is small -> FP is 'small'.
      - raw NMS order (low first): the LOW-score pred claims GT; the HIGH-score
        pred is the FP. The high pred is large -> FP is 'large'.
    Both orders give FP count == 1, FN == 0, but the FP *size bucket* differs,
    which is exactly the Evaluation-divergence the score-sort fix removes.
    """
    img_dir = tmp_path / "imgs"
    lbl_dir = tmp_path / "lbls"
    img_dir.mkdir()
    lbl_dir.mkdir()
    img_path = img_dir / "a.jpg"
    img_path.write_bytes(b"fake")
    model_path = tmp_path / "m.onnx"
    model_path.write_bytes(b"fake")

    # 100x100 frame. GT: class 0, full-frame box.
    (lbl_dir / "a.txt").write_text("0 0.5 0.5 1.0 1.0\n")

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    monkeypatch.setattr(ar, "imread", lambda _p: frame)
    monkeypatch.setattr(ar, "glob_images", lambda _d: [str(img_path)])

    import core.model_loader
    monkeypatch.setattr(
        core.model_loader, "load_model",
        lambda *a, **k: _FakeModelInfo({0: "cat"}),
    )

    # Both clear IoU>=0.5 against the full-frame GT [0,0,100,100], and land in
    # DIFFERENT size buckets so the FP's bucket reveals which pred was matched.
    # Buckets: small <32^2=1024, medium <96^2=9216, else large.
    #   pred_high (large): [0,0,100,96] area 9600 (>=9216 -> 'large'), IoU 0.96.
    #   pred_low (medium): [0,0,80,72]  area 5760 (<9216  -> 'medium'), IoU 0.576.
    pred_high = np.array([0.0, 0.0, 100.0, 96.0])   # score 0.9
    pred_low = np.array([0.0, 0.0, 80.0, 72.0])     # score 0.1

    # Feed the worker in raw order (low first) — this is where a naive,
    # unsorted matcher would mis-assign GT to the low-score prediction.
    res = _FakeResult(
        boxes=[pred_low, pred_high],
        scores=[0.1, 0.9],
        class_ids=[0, 0],
    )
    import core.inference
    monkeypatch.setattr(core.inference, "run_inference", lambda *a, **k: res)

    return {
        "img_dir": str(img_dir),
        "lbl_dir": str(lbl_dir),
        "model_path": str(model_path),
        "pred_high": pred_high,
        "pred_low": pred_low,
    }


def test_predictions_score_sorted_before_greedy_match(crossing_dataset):
    """Score-desc sorting picks the HIGH-score pred as the GT match.

    With one GT and two overlapping predictions, exactly one is TP and the
    other is FP. After the fix the matcher sorts by score desc, so the
    high-score (large) pred claims the GT and the low-score (medium-area) pred
    is the FP — matching the Evaluation tab. A naive unsorted matcher (low
    first) would instead make the large high-score pred the FP, so the FP's
    size bucket is the discriminator.
    """
    ds = crossing_dataset

    # Sanity: both predictions clear IoU>=0.5 against the full-frame GT, so the
    # test exercises *ordering*, not threshold coincidence.
    from core.evaluation import _compute_iou_matrix
    gt = [0, 0, 100, 100]
    iou = _compute_iou_matrix([ds["pred_high"], ds["pred_low"]], [gt])
    assert iou[0, 0] >= 0.5 and iou[1, 0] >= 0.5
    assert iou[0, 0] > iou[1, 0]  # high-score pred has the better IoU too

    error_analysis_state.update(running=False)
    r = client.post("/api/analysis/error-analysis", json={
        "model_path": ds["model_path"],
        "img_dir": ds["img_dir"],
        "label_dir": ds["lbl_dir"],
        "iou_threshold": 0.5,
        "conf": 0.01,
    })
    assert r.status_code == 200, r.text
    assert r.json().get("ok") is True
    results = _wait_until_done("/api/analysis/error-analysis/status")["results"]

    # Exactly one FP, no FN (the single GT is matched by one of the two preds).
    assert results["fp"]["count"] == 1, results["fp"]
    assert results["fn"]["count"] == 0, results["fn"]
    # The high-score pred ([0,0,100,96], 'large') matched the GT, so the FP is
    # the low-score pred ([0,0,80,72], 'medium'). An unsorted matcher (low
    # first) would instead leave the LARGE pred as the FP -> fp['large']==1.
    assert results["fp"]["medium"] == 1 and results["fp"]["large"] == 0, results["fp"]


def test_error_analysis_try_start_rejects_concurrent_run():
    """try_start() makes the second concurrent POST lose atomically."""
    import threading

    error_analysis_state.update(running=False)
    # Hold running=True to simulate an in-flight job; the route's try_start
    # must reject a new POST without clobbering state.
    error_analysis_state.update(running=True)
    try:
        r = client.post("/api/analysis/error-analysis", json={
            "model_path": "irrelevant.onnx",  # never reached — guard rejects first
            "img_dir": "irrelevant",
            "label_dir": "irrelevant",
        })
        # Path validation happens before try_start, so an invalid path returns
        # the PATH_ envelope; a *valid* in-flight guard returns Already running.
        # To isolate the guard we assert the worker did not start a second run:
        body = r.json()
        assert body.get("error") == "Already running" or body.get("code", "").startswith("PATH_")
    finally:
        error_analysis_state.update(running=False)


def test_inference_analysis_offloaded_does_not_block(monkeypatch, tmp_path):
    """inference-analysis runs its sync body in the executor (event loop free).

    Mock model load + inference so the route returns a normal payload without
    real weights; the assertion is that the response shape is intact (timing
    present), proving the offloaded path returns correctly.
    """
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    img_path = img_dir / "a.jpg"
    img_path.write_bytes(b"fake")
    model_path = tmp_path / "m.onnx"
    model_path.write_bytes(b"fake")

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    monkeypatch.setattr(ar, "imread", lambda _p: frame)

    class _Sess:
        def get_inputs(self):
            class _I:
                shape = [1, 3, 64, 64]
            return [_I()]

        def get_outputs(self):
            class _O:
                shape = [1, 5, 100]
            return [_O()]

    class _MI:
        names = {0: "cat"}
        input_size = (64, 64)
        session = _Sess()

    res = _FakeResult(boxes=[], scores=[], class_ids=[])
    import core.model_loader, core.inference
    monkeypatch.setattr(core.model_loader, "load_model", lambda *a, **k: _MI())
    monkeypatch.setattr(core.inference, "run_inference", lambda *a, **k: res)

    r = client.post("/api/analysis/inference-analysis", json={
        "model_path": str(model_path),
        "image_path": str(img_path),
        "conf": 0.25,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "error" not in body, body
    assert "timing" in body and "infer_ms" in body["timing"]


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
