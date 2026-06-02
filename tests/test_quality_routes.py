"""Tests for /api/quality/* routes (track: quality).

Covers the Wave-1 fixes in server/quality_routes.py:
  - anomaly: a single malformed label line no longer aborts the scan
    and is counted; missing label files are reported (not silently skipped).
  - leaky: recursive option finds nested cross-split duplicates.
  - all 5 endpoints use TaskState.try_start() as an atomic start guard.

The app under test mounts ONLY the quality router so the suite stays
isolated from other waves concurrently editing the full app. Images are
written as real (tiny) PNGs via cv2 so the leaky/duplicate hashing path
runs end-to-end without mocking imread.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import cv2

from server import quality_routes
from server.state import anomaly_state, leaky_state


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(quality_routes.router)
    return TestClient(app)


def _reset(state) -> None:
    """Clear a TaskState between tests so a prior run's `running` flag or
    cached index can't leak into the next test."""
    state.update(running=False, progress=0, total=0, msg="", results=[])


def _wait_until_done(client: TestClient, status_url: str, timeout_s: float = 10.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        data = client.get(status_url).json()
        if not data.get("running"):
            return data
        time.sleep(0.02)
    raise AssertionError(f"worker did not finish within {timeout_s}s")


def _write_png(path: str, color: int = 0) -> None:
    img = np.full((8, 8, 3), color, dtype=np.uint8)
    cv2.imwrite(path, img)


# ── anomaly: malformed line survives + is counted (P0) ──────────────

def test_malformed_label_line_does_not_abort_scan(client, tmp_path):
    """One non-numeric label line must not discard the whole scan; the valid
    out-of-bounds box on the next line must still be reported, and the
    malformed line must be tallied rather than crash the run."""
    _reset(anomaly_state)
    img_dir = tmp_path / "imgs"
    lbl_dir = tmp_path / "lbls"
    img_dir.mkdir()
    lbl_dir.mkdir()
    # Content is irrelevant — the anomaly tool never decodes the image, it
    # only globs filenames and reads the matching .txt.
    (img_dir / "a.jpg").write_bytes(b"not-a-real-image")
    # Line 1: non-numeric header (would raise ValueError on float()).
    # Line 2: a valid but out-of-bounds box (cx+w/2 = 1.4 > 1.01).
    (lbl_dir / "a.txt").write_text("foo bar baz qux quux\n0 0.9 0.5 1.0 0.2\n")

    resp = client.post("/api/quality/anomaly",
                       json={"img_dir": str(img_dir), "label_dir": str(lbl_dir)})
    assert resp.json() == {"ok": True}

    data = _wait_until_done(client, "/api/quality/anomaly/status")
    assert data["running"] is False
    # The scan completed normally — NOT via the route_errors abort path.
    assert not data["msg"].startswith("Error:")
    assert data["malformed_lines"] == 1
    # The valid OOB anomaly on the second line survived the bad first line.
    types = [r["type"] for r in data["results"]]
    assert any("Out-of-bounds" in t for t in types)


# ── anomaly: missing label files reported (P1) ──────────────────────

def test_missing_label_files_are_counted(client, tmp_path):
    """Images with no matching .txt must be tallied as missing_labels and
    surfaced in the message — a wrong label_dir should not read as 'clean'."""
    _reset(anomaly_state)
    img_dir = tmp_path / "imgs"
    lbl_dir = tmp_path / "lbls"
    img_dir.mkdir()
    lbl_dir.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (img_dir / name).write_bytes(b"x")
    # Only one of the three images has a label file.
    (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    resp = client.post("/api/quality/anomaly",
                       json={"img_dir": str(img_dir), "label_dir": str(lbl_dir)})
    assert resp.json() == {"ok": True}

    data = _wait_until_done(client, "/api/quality/anomaly/status")
    assert data["running"] is False
    assert data["missing_labels"] == 2
    assert "no label file" in data["msg"]


# ── leaky: recursive finds nested cross-split dupes (P0) ────────────

def test_leaky_recursive_finds_nested_duplicates(client, tmp_path):
    """A duplicate that lives under train/sub/ and val/sub/ is only found
    when recursive=True; recursive=False must miss it (proving the flag is
    actually wired through to glob_images)."""
    _reset(leaky_state)
    train_sub = tmp_path / "train" / "sub"
    val_sub = tmp_path / "val" / "sub"
    train_sub.mkdir(parents=True)
    val_sub.mkdir(parents=True)
    # Identical content under nested subdirs of each split → cross-split leak.
    _write_png(str(train_sub / "a.jpg"), color=50)
    _write_png(str(val_sub / "a.jpg"), color=50)

    # recursive=True → the nested duplicate is detected.
    resp = client.post("/api/quality/leaky",
                       json={"train_dir": str(tmp_path / "train"),
                             "val_dir": str(tmp_path / "val"),
                             "threshold": 0, "recursive": True})
    assert resp.json() == {"ok": True}
    data = _wait_until_done(client, "/api/quality/leaky/status")
    assert data["running"] is False
    total_dupes = sum(r["duplicates"] for r in data["results"])
    assert total_dupes >= 1

    # recursive=False → the nested files are never globbed, so 0 leaks.
    _reset(leaky_state)
    resp = client.post("/api/quality/leaky",
                       json={"train_dir": str(tmp_path / "train"),
                             "val_dir": str(tmp_path / "val"),
                             "threshold": 0, "recursive": False})
    assert resp.json() == {"ok": True}
    data = _wait_until_done(client, "/api/quality/leaky/status")
    total_dupes = sum(r["duplicates"] for r in data["results"])
    assert total_dupes == 0


# ── try_start guard: second concurrent POST is rejected ─────────────

def test_anomaly_try_start_rejects_second_run(client, tmp_path):
    """While a run is in flight, a second POST must be rejected by the atomic
    try_start guard. We hold the first run open by marking it running before
    the request (the route's try_start sees running=True and refuses)."""
    _reset(anomaly_state)
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    (img_dir / "a.jpg").write_bytes(b"x")

    # Simulate an in-flight run.
    anomaly_state.update(running=True)
    try:
        resp = client.post("/api/quality/anomaly", json={"img_dir": str(img_dir)})
        assert resp.json() == {"error": "Already running"}
    finally:
        _reset(anomaly_state)
