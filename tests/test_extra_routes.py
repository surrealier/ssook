"""Tests for /api/clip/*, /api/embedder/*, /api/segmentation/* (track: extra_routes).

Covers the Wave-1 P0/P1 fixes in server/extra_routes.py:
- CLIP builds ground truth by scanning class subfolders RECURSIVELY
  (ImageNet-style img_dir/<class>/*.jpg), so accuracy is no longer always 0.
- embedder_compare validates the model path before opening an ORT session.
- segmentation result carries metrics{mIoU,mDice,per_class} plus a top-level
  per-image `images:[{path, overlay}]` list with base64-PNG overlays.

Model loading + inference are mocked so the suite stays fast and needs no
real .onnx weights.
"""
import base64
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server.extra_routes as er
from server import app
from server.state import clip_state, embedder_state, seg_state

client = TestClient(app)


def _wait_until_done(status_url: str, timeout_s: float = 10.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        data = client.get(status_url).json()
        if not data.get("running"):
            return data
        time.sleep(0.02)
    raise AssertionError(f"worker did not finish within {timeout_s}s")


# ── CLIP: recursive GT folder scanning ─────────────────────────────────────
class _FakeCLIPModel:
    """zero_shot_classify ranks the GT label (=parent folder) first, so a
    correctly-scanned dataset yields 100% accuracy. If the GT layout were not
    scanned at all (the old non-recursive bug), label_total stays 0."""

    def __init__(self, *a, **k):
        pass

    def encode_text(self, tokens):
        return np.zeros((1, 4), dtype=np.float32)

    def zero_shot_classify(self, frame, text_embs, labels):
        # Decide the "prediction" from the frame's first pixel, which the
        # fixture sets to the index of the image's true class.
        idx = int(frame[0, 0, 0]) % len(labels)
        ranked = [(labels[idx], 0.9)]
        ranked += [(l, 0.1) for l in labels if l != labels[idx]]
        return ranked


@pytest.fixture
def clip_dataset(tmp_path, monkeypatch):
    """ImageNet-style dir: img_dir/cat/*.jpg, img_dir/dog/*.jpg."""
    img_dir = tmp_path / "imgs"
    (img_dir / "cat").mkdir(parents=True)
    (img_dir / "dog").mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        (img_dir / "cat" / name).write_bytes(b"x")
        (img_dir / "dog" / name).write_bytes(b"x")
    img_enc = tmp_path / "img.onnx"
    txt_enc = tmp_path / "txt.onnx"
    img_enc.write_bytes(b"x")
    txt_enc.write_bytes(b"x")

    labels = ["cat", "dog"]

    def _imread(fp):
        # Encode the true class index (parent folder) into pixel 0 so the fake
        # model can "predict" it correctly.
        parent = os.path.basename(os.path.dirname(fp))
        idx = labels.index(parent) if parent in labels else 0
        return np.full((8, 8, 3), idx, dtype=np.uint8)

    monkeypatch.setattr(er, "imread", _imread)
    import core.clip_inference
    monkeypatch.setattr(core.clip_inference, "CLIPModel", _FakeCLIPModel)
    monkeypatch.setattr(core.clip_inference, "simple_tokenize", lambda s: [0])

    return {"img_dir": str(img_dir), "img_enc": str(img_enc),
            "txt_enc": str(txt_enc)}


def test_clip_scans_class_subfolders_recursively(clip_dataset):
    """The standard img_dir/<class>/*.jpg layout must produce non-zero GT
    totals (the old non-recursive glob made every total 0)."""
    clip_state.update(running=False)
    r = client.post("/api/clip/run", json={
        "image_encoder": clip_dataset["img_enc"],
        "text_encoder": clip_dataset["txt_enc"],
        "img_dir": clip_dataset["img_dir"],
        "labels": "cat,dog",
    })
    assert r.status_code == 200, r.text
    assert r.json().get("ok") is True
    data = _wait_until_done("/api/clip/status")

    results = {row["label"]: row for row in data["results"]}
    assert "cat" in results and "dog" in results
    # Each class has 2 images, all matched and correctly classified.
    assert results["cat"]["total"] == 2
    assert results["dog"]["total"] == 2
    assert results["Overall"]["total"] == 4
    assert results["Overall"]["accuracy"] == 100.0
    # `matched` surfaces the layout-OK signal.
    assert data["matched"] == 4


def test_clip_flat_dir_reports_no_match(tmp_path, monkeypatch):
    """Images directly under img_dir (no class subfolder) report matched=0
    with an explicit warning, not a silent 0%."""
    img_dir = tmp_path / "flat"
    img_dir.mkdir()
    (img_dir / "a.jpg").write_bytes(b"x")
    (img_dir / "b.jpg").write_bytes(b"x")
    img_enc = tmp_path / "img.onnx"
    txt_enc = tmp_path / "txt.onnx"
    img_enc.write_bytes(b"x")
    txt_enc.write_bytes(b"x")

    monkeypatch.setattr(er, "imread", lambda fp: np.zeros((8, 8, 3), dtype=np.uint8))
    import core.clip_inference
    monkeypatch.setattr(core.clip_inference, "CLIPModel", _FakeCLIPModel)
    monkeypatch.setattr(core.clip_inference, "simple_tokenize", lambda s: [0])

    clip_state.update(running=False)
    r = client.post("/api/clip/run", json={
        "image_encoder": str(img_enc), "text_encoder": str(txt_enc),
        "img_dir": str(img_dir), "labels": "cat,dog",
    })
    assert r.status_code == 200, r.text
    data = _wait_until_done("/api/clip/status")
    # Flat dir → images skipped (parent == img_dir) → no class-subfolder images.
    assert "No class-subfolder images" in data["msg"]


def test_clip_concurrent_start_guard(clip_dataset, monkeypatch):
    """try_start: a second POST while running returns the already-running
    error rather than launching a second worker."""
    clip_state.update(running=False)
    # Force the state to look running so the route's try_start returns False.
    assert clip_state.try_start(progress=0) is True
    try:
        r = client.post("/api/clip/run", json={
            "image_encoder": clip_dataset["img_enc"],
            "text_encoder": clip_dataset["txt_enc"],
            "img_dir": clip_dataset["img_dir"],
            "labels": "cat,dog",
        })
        assert r.json() == {"error": "Already running"}
    finally:
        clip_state.update(running=False)


# ── embedder_compare: path validation ──────────────────────────────────────
class TestEmbedderComparePathSafety:
    def test_rejects_path_outside_with_bad_ext(self, tmp_path):
        """A non-.onnx model path must be rejected at the boundary before any
        ORT session is opened (the bug fed the raw request path to ORT)."""
        bad_model = tmp_path / "m.txt"
        bad_model.write_bytes(b"x")
        img_a = tmp_path / "a.jpg"
        img_b = tmp_path / "b.jpg"
        img_a.write_bytes(b"x")
        img_b.write_bytes(b"x")
        r = client.post("/api/embedder/compare", json={
            "model_path": str(bad_model),
            "img_paths": [str(img_a), str(img_b)],
        })
        assert r.status_code == 400
        assert r.json()["code"] == "PATH_BAD_EXT"

    def test_rejects_nonexistent_model(self, tmp_path):
        img_a = tmp_path / "a.jpg"
        img_b = tmp_path / "b.jpg"
        img_a.write_bytes(b"x")
        img_b.write_bytes(b"x")
        r = client.post("/api/embedder/compare", json={
            "model_path": str(tmp_path / "missing.onnx"),
            "img_paths": [str(img_a), str(img_b)],
        })
        assert r.status_code == 400
        assert r.json()["code"] == "PATH_NOT_FOUND"

    def test_validates_model_before_session_open(self, tmp_path, monkeypatch):
        """The ORT session must be opened with the VALIDATED (normalized) path,
        not the raw request string."""
        model = tmp_path / "m.onnx"
        model.write_bytes(b"x")
        img_a = tmp_path / "a.jpg"
        img_b = tmp_path / "b.jpg"
        img_a.write_bytes(b"x")
        img_b.write_bytes(b"x")

        opened_with = {}

        class _FakeInput:
            name = "in"
            shape = [1, 3, 224, 224]

        class _FakeSession:
            def __init__(self, path):
                opened_with["path"] = path

            def get_inputs(self):
                return [_FakeInput()]

            def run(self, _out, _feed):
                return [np.ones((1, 8), dtype=np.float32)]

        import onnxruntime as ort
        monkeypatch.setattr(ort, "InferenceSession", _FakeSession)
        monkeypatch.setattr(er, "imread", lambda fp: np.zeros((8, 8, 3), dtype=np.uint8))

        # Pass a relative-ish path; the handler must normalize it to absolute.
        r = client.post("/api/embedder/compare", json={
            "model_path": str(model),
            "img_paths": [str(img_a), str(img_b)],
        })
        assert r.status_code == 200, r.text
        assert opened_with["path"] == os.path.abspath(str(model))


# ── segmentation: metrics + images result shape ────────────────────────────
class _FakeSegResult:
    def __init__(self, mask):
        self.mask = mask
        self.num_classes = 3
        self.infer_ms = 1.0


@pytest.fixture
def seg_dataset(tmp_path, monkeypatch):
    """3 images + matching GT masks. One image gets a deliberately-wrong pred
    so it ranks lowest and earns an overlay."""
    img_dir = tmp_path / "imgs"
    gt_dir = tmp_path / "gt"
    img_dir.mkdir()
    gt_dir.mkdir()
    model = tmp_path / "seg.onnx"
    model.write_bytes(b"x")

    files = ["good1.jpg", "good2.jpg", "bad.jpg"]
    for name in files:
        (img_dir / name).write_bytes(b"x")
        # GT mask: a (10,10) array with class 1 in the top half.
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[:5, :] = 1
        # cv2.imwrite via the real cv2 so _find_gt_mask + cv2.imread work.
        import cv2
        cv2.imwrite(str(gt_dir / (os.path.splitext(name)[0] + ".png")), gt)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setattr(er, "imread", lambda fp: frame)

    import core.model_loader

    class _FakeMI:
        pass

    monkeypatch.setattr(er, "_load_model", lambda *a, **k: _FakeMI())

    def _fake_run_seg(mi, fr):
        # Caller sets which file is "current" via a closure-tracked counter is
        # awkward; instead key off a module-level toggle in the test.
        return _FakeSegResult(_fake_run_seg.next_mask)

    # Default: perfect prediction (matches GT). The test overrides per-image
    # by patching run_segmentation directly.
    perfect = np.zeros((10, 10), dtype=np.uint8)
    perfect[:5, :] = 1
    _fake_run_seg.next_mask = perfect
    monkeypatch.setattr(er, "run_segmentation", _fake_run_seg)

    return {"img_dir": str(img_dir), "gt_dir": str(gt_dir),
            "model": str(model), "perfect": perfect}


def test_segmentation_result_has_metrics_and_images(seg_dataset, monkeypatch):
    """The status result must carry metrics{mIoU,mDice,per_class} AND a
    top-level images list of {path, overlay} per evaluated image."""
    # Predict perfectly for good1/good2 and badly for bad.jpg so bad ranks
    # lowest and gets an overlay. We drive the mask via glob order.
    perfect = seg_dataset["perfect"]
    wrong = np.zeros((10, 10), dtype=np.uint8)  # all background → 0 IoU vs GT class 1

    order = sorted(["good1.jpg", "good2.jpg", "bad.jpg"])  # glob_images sorts
    call_idx = {"i": 0}

    def _run_seg(mi, fr):
        i = call_idx["i"]
        call_idx["i"] += 1
        name = order[i] if i < len(order) else order[-1]
        return _FakeSegResult(wrong if name == "bad.jpg" else perfect)

    monkeypatch.setattr(er, "run_segmentation", _run_seg)

    seg_state.update(running=False)
    r = client.post("/api/segmentation/run", json={
        "model_path": seg_dataset["model"],
        "img_dir": seg_dataset["img_dir"],
        "gt_mask_dir": seg_dataset["gt_dir"],
        "num_classes": 3,
    })
    assert r.status_code == 200, r.text
    assert r.json().get("ok") is True
    data = _wait_until_done("/api/segmentation/status")

    results = data["results"]
    # metrics block
    assert "metrics" in results
    metrics = results["metrics"]
    assert "mIoU" in metrics and "mDice" in metrics
    assert isinstance(metrics["per_class"], list)
    assert any(pc["class"] == 1 for pc in metrics["per_class"])

    # images block: one entry per evaluated image, each with path + overlay key
    images = data["images"]
    assert len(images) == 3
    for entry in images:
        assert "path" in entry
        assert "overlay" in entry
        assert "iou" in entry

    # bad.jpg has the lowest IoU (the good images score 1.0) and so must carry
    # a non-empty base64-PNG overlay.
    by_file = {e["file"]: e for e in images}
    assert by_file["bad.jpg"]["iou"] < by_file["good1.jpg"]["iou"]
    assert by_file["good1.jpg"]["iou"] == 1.0
    overlay = by_file["bad.jpg"]["overlay"]
    assert overlay  # non-empty
    decoded = base64.b64decode(overlay)
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic


def test_segmentation_concurrent_start_guard(seg_dataset):
    """try_start: a second POST while running returns the already-running
    error envelope rather than double-submitting a worker."""
    seg_state.update(running=False)
    assert seg_state.try_start(progress=0) is True
    try:
        r = client.post("/api/segmentation/run", json={
            "model_path": seg_dataset["model"],
            "img_dir": seg_dataset["img_dir"],
            "gt_mask_dir": seg_dataset["gt_dir"],
            "num_classes": 3,
        })
        assert r.json() == {"error": "Segmentation evaluation already running"}
    finally:
        seg_state.update(running=False)
