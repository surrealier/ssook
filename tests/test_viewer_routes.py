"""Viewer route tests for the v1.6.0 hardening findings.

Covers the observable behavior of:
- VIEWER-02: _video_sessions mutations + cleanup scan are RLock-guarded
  (no 'dict changed size during iteration' under concurrent churn).
- VIEWER-03: seek/step reset the active tracker (ID/trajectory invariants).
- VIEWER-04: embedding/vlm models are refused by viewer_start.
"""
import asyncio
import threading

import numpy as np

from server import viewer_routes


# ── VIEWER-02: concurrent session churn vs. cleanup scan ────────────────
def test_cleanup_scan_survives_concurrent_churn():
    """N threads insert/pop _video_sessions while cleanup scans repeatedly.

    Before the RLock guard the cleanup scan iterated a plain dict that other
    threads mutated, raising 'dictionary changed size during iteration'.
    """
    stop = threading.Event()
    errors: list[Exception] = []

    def churn(worker_id: int):
        try:
            for i in range(500):
                sid = f"w{worker_id}-{i}"
                with viewer_routes._sessions_lock:
                    # Minimal session shape; cleanup only reads playing/last_access.
                    viewer_routes._video_sessions[sid] = {
                        "playing": False, "last_access": 0.0, "cap": None,
                    }
                with viewer_routes._sessions_lock:
                    viewer_routes._video_sessions.pop(sid, None)
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)
        finally:
            stop.set()

    def scanner():
        try:
            while not stop.is_set():
                viewer_routes._cleanup_stale_sessions()
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    threads = [threading.Thread(target=churn, args=(w,)) for w in range(8)]
    threads += [threading.Thread(target=scanner) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"concurrency raised: {errors!r}"


# ── VIEWER-03: seek/step reset the tracker ──────────────────────────────
def test_tracker_reset_clears_tracks_and_ids():
    """Growing a ByteTracker then reset() must drop tracks and rewind ids.

    generate()'s seek/step branches call sess['tracker'].reset(); this asserts
    the invariant those branches rely on so a discontinuity does not streak
    trajectories or keep stale IDs.
    """
    from core.tracking import create_tracker

    tracker = create_tracker("bytetrack")
    boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
    scores = np.array([0.9])
    class_ids = np.array([0])
    # min_hits=3 → push a few frames so a track is established.
    for _ in range(5):
        tracker.update(boxes, scores, class_ids)
    assert len(tracker.tracks) > 0
    assert tracker._next_id > 1

    tracker.reset()
    assert tracker.tracks == []
    assert tracker._next_id == 1
    assert tracker.frame_count == 0


# ── VIEWER-04: viewer refuses embedding/vlm models ──────────────────────
class _StubModel:
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.model_type = "yolo"
        self.names = {}


def _start(monkeypatch, task_type: str):
    monkeypatch.setattr(viewer_routes, "safe_model_file", lambda p: p)
    monkeypatch.setattr(viewer_routes, "safe_video_file", lambda p: p)
    monkeypatch.setattr(
        viewer_routes, "ensure_model",
        lambda *a, **k: _StubModel(task_type),
    )
    req = viewer_routes.VideoStartRequest(model_path="m.onnx", video_path="v.mp4")
    return asyncio.run(viewer_routes.viewer_start(req))


def test_viewer_start_refuses_vlm(monkeypatch):
    resp = _start(monkeypatch, "vlm")
    assert resp.get("error") == "vlm_unsupported"
    assert "session_id" not in resp


def test_viewer_start_refuses_embedding(monkeypatch):
    resp = _start(monkeypatch, "embedding")
    assert resp.get("error") == "vlm_unsupported"
    assert "session_id" not in resp
