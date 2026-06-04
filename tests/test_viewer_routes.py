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


# ── P1/concurrency: per-stream sequential state isolation ───────────────
# These import core.inference, which imports cv2 at module load. Under the
# memory-safe protocol they are written but DEFERRED to the central run.
# _preprocess_single_frame is monkeypatched so no real cv2 op executes.
class _StubSeqModel:
    """Minimal seq_ ModelInfo stand-in: only the attrs run_inference touches."""

    def __init__(self):
        self.model_type = "seq_yolo"
        self.input_size = (32, 32)
        self.names = {0: "person"}


def _drive_seq(model, seq_state, n_frames: int, label: str) -> dict:
    """Push n_frames through preprocess_sequential_cached with a per-stream
    seq_state. Returns the counters/tensors recorded so callers can assert
    two streams advance independently. Each frame's chw carries a stream tag
    so a cross-stream buffer leak is detectable.
    """
    import numpy as _np
    from core import inference

    produced = []  # frame indices where a tensor (not a stride-skip) was emitted
    for i in range(n_frames):
        frame = _np.zeros((8, 8, 3), dtype=_np.uint8)
        out = inference.preprocess_sequential_cached(
            model, frame, model.input_size, seq_state=seq_state)
        if out is not None:
            produced.append(i)
    return {
        "produced": produced,
        "counter": seq_state["_seq_frame_counter"],
        "buf_len": len(seq_state["_seq_tensor_buf"]),
    }


def test_seq_state_two_sessions_advance_independently(monkeypatch):
    """Two sessions of the SAME seq model, each with its own seq_state, must
    keep independent stride counters/buffers — the bug was a shared counter on
    the ModelInfo singleton causing one stream's parity to skip the other's.
    """
    from core import inference

    # Avoid cv2: produce a deterministic (3,H,W) tensor tagged with a counter.
    calls = {"n": 0}

    def fake_single(frame, input_size, imagenet_norm, use_letterbox):
        import numpy as _np
        calls["n"] += 1
        h, w = input_size
        return _np.full((3, h, w), float(calls["n"]), dtype=_np.float32), 1.0, (0.0, 0.0)

    monkeypatch.setattr(inference, "_preprocess_single_frame", fake_single)

    model = _StubSeqModel()
    inference.reset_seq_state(state_a := {})
    inference.reset_seq_state(state_b := {})

    res_a = _drive_seq(model, state_a, 7, "A")
    res_b = _drive_seq(model, state_b, 7, "B")

    # Each session counts only its OWN 7 frames (not 14 shared).
    assert res_a["counter"] == 7
    assert res_b["counter"] == 7
    # Same input cadence → same emit pattern, proving no parity collision.
    assert res_a["produced"] == res_b["produced"]
    # Buffers are distinct objects (no aliasing through the model singleton).
    assert state_a["_seq_tensor_buf"] is not state_b["_seq_tensor_buf"]


def test_seq_state_falls_back_to_model_attrs_when_none(monkeypatch):
    """seq_state=None must use the ModelInfo attrs (legacy single-stream path),
    leaving non-viewer callers (eval/benchmark) unchanged.
    """
    from core import inference

    def fake_single(frame, input_size, imagenet_norm, use_letterbox):
        import numpy as _np
        h, w = input_size
        return _np.zeros((3, h, w), dtype=_np.float32), 1.0, (0.0, 0.0)

    monkeypatch.setattr(inference, "_preprocess_single_frame", fake_single)

    model = _StubSeqModel()
    import numpy as _np
    for _ in range(4):
        inference.preprocess_sequential_cached(
            model, _np.zeros((8, 8, 3), dtype=_np.uint8), model.input_size)

    # State landed on the model attrs, not on any dict.
    assert getattr(model, "_seq_frame_counter") == 4
    assert len(getattr(model, "_seq_tensor_buf")) == 3


def test_reset_seq_state_clears_per_session_dict():
    """reset_seq_state on a populated dict rewinds counter/buffers (seek/step)."""
    from core import inference

    state = {
        "_seq_frame_counter": 9,
        "_seq_tensor_buf": [1, 2, 3],
        "_seq_last_result": object(),
        "_seq_async_future": object(),
        "_frame_buffer": [object()],
    }
    inference.reset_seq_state(state)
    assert state["_seq_frame_counter"] == 0
    assert state["_seq_tensor_buf"] == []
    assert state["_seq_last_result"] is None
    assert state["_seq_async_future"] is None
    assert state["_frame_buffer"] == []
    # None is a no-op (used when a non-seq model has no seq_state).
    inference.reset_seq_state(None)


# ── P1: inference-input downscale (infer_max_side) ──────────────────────
# These import core.inference, which imports cv2 at module load. Under the
# memory-safe protocol they are WRITTEN but DEFERRED to the central run (cv2).
# The coordinate-rescale math itself is exercised standalone below WITHOUT any
# cv2/core.inference import, so a pure-logic guard always runs.


def _make_detection_result(boxes, scores=None, class_ids=None, extra=None):
    """Build a DetectionResult (imports core.inference → cv2; deferred path)."""
    from core.inference import DetectionResult
    import numpy as _np

    n = len(boxes)
    return DetectionResult(
        boxes=_np.asarray(boxes, dtype=_np.float32).reshape(-1, 4),
        scores=_np.asarray(scores if scores is not None else [0.9] * n, dtype=_np.float32),
        class_ids=_np.asarray(class_ids if class_ids is not None else [0] * n, dtype=_np.int32),
        infer_ms=3.0,
        extra_attrs=extra,
    )


def test_rescale_detection_result_maps_boxes_back_within_original_bounds():
    """A result decoded on a downscaled frame, rescaled by 1/scale, must land
    within the ORIGINAL frame bounds and recover the pre-downscale coords.
    """
    from core.inference import _rescale_detection_result
    import numpy as _np

    # Original 4000x2000 frame downscaled by scale=0.25 → inference at 1000x500.
    orig_shape = (2000, 4000, 3)
    scale = 0.25
    # Boxes as the model would emit them in DOWNSCALED coords.
    inferred = _make_detection_result(
        [[100.0, 50.0, 400.0, 250.0], [0.0, 0.0, 999.0, 499.0]],
        scores=[0.8, 0.6], class_ids=[1, 2],
        extra=_np.ones((2, 7), dtype=_np.float32),
    )
    out = _rescale_detection_result(inferred, scale, orig_shape)

    # Boxes scaled up by 1/scale = 4x.
    assert out.boxes[0].tolist() == [400.0, 200.0, 1600.0, 1000.0]
    # All coords within original bounds (W=4000, H=2000).
    assert (out.boxes[:, [0, 2]] <= 4000).all() and (out.boxes[:, [0, 2]] >= 0).all()
    assert (out.boxes[:, [1, 3]] <= 2000).all() and (out.boxes[:, [1, 3]] >= 0).all()
    # scores/class_ids/extra carried through unchanged (scores are float32).
    assert _np.allclose(out.scores, [0.8, 0.6])
    assert out.class_ids.tolist() == [1, 2]
    assert out.extra_attrs is not None and out.extra_attrs.shape == (2, 7)


def test_rescale_detection_result_does_not_mutate_input():
    """Rescale must return a NEW result and leave the input untouched, so the
    seq_ stride cache (which holds the pre-rescale result) is never double-scaled.
    """
    from core.inference import _rescale_detection_result

    src = _make_detection_result([[10.0, 10.0, 20.0, 20.0]])
    before = src.boxes.copy()
    out = _rescale_detection_result(src, 0.5, (1000, 1000, 3))
    assert out is not src
    assert src.boxes.tolist() == before.tolist()  # input unchanged
    assert out.boxes[0].tolist() == [20.0, 20.0, 40.0, 40.0]


def test_rescale_detection_result_noop_when_scale_is_one():
    """scale==1.0 (infer_max_side unset / frame already fit) returns the SAME
    object unchanged — proving the default path is behavior-preserving.
    """
    from core.inference import _rescale_detection_result

    src = _make_detection_result([[1.0, 2.0, 3.0, 4.0]])
    out = _rescale_detection_result(src, 1.0, (100, 100, 3))
    assert out is src
    assert out.boxes[0].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_infer_downscale_noop_when_unset_or_small():
    """infer_max_side<=0 OR frame within the cap returns the SAME frame array
    and scale=1.0 — no cv2.resize, no copy, today's behavior preserved.
    """
    from core.inference import _infer_downscale
    import numpy as _np

    frame = _np.zeros((480, 640, 3), dtype=_np.uint8)
    # Unset (0) → unchanged.
    f0, s0, shape0 = _infer_downscale(frame, 0)
    assert f0 is frame and s0 == 1.0 and shape0 == frame.shape
    # Cap larger than the longer side (640) → unchanged.
    f1, s1, _ = _infer_downscale(frame, 1280)
    assert f1 is frame and s1 == 1.0


def test_infer_downscale_shrinks_large_frame_preserving_aspect():
    """A frame whose longer side exceeds the cap is shrunk isotropically; the
    returned scale recovers the original via coord/scale and true_shape is the
    ORIGINAL shape (so boxes map back correctly).
    """
    from core.inference import _infer_downscale

    frame = _make_frame(2160, 3840)  # 4K landscape (H, W)
    small, scale, true_shape = _infer_downscale(frame, 1280)
    # Longer side (W=3840) → 1280 → scale = 1280/3840.
    assert abs(scale - (1280 / 3840)) < 1e-9
    sh, sw = small.shape[:2]
    assert max(sh, sw) == 1280            # longer side hits the cap
    assert abs(sw / sh - 3840 / 2160) < 0.01  # aspect preserved
    assert true_shape == (2160, 3840, 3)  # original shape retained for rescale


def _make_frame(h, w):
    """Allocate a BGR frame (imports numpy only — no cv2)."""
    import numpy as _np
    return _np.zeros((h, w, 3), dtype=_np.uint8)


# ── Pure-logic guard: rescale invariant WITHOUT importing cv2/core ──────
# Mirrors _rescale_detection_result's contract using only numpy so it ALWAYS
# runs under the memory-safe protocol (no cv2 import). If the real function ever
# diverges from "coord/scale, clipped to bounds", the deferred tests above catch
# it against the actual implementation.
def test_rescale_math_invariant_pure_numpy():
    boxes = np.array([[100.0, 50.0, 400.0, 250.0]], dtype=np.float32)
    scale = 0.25
    oh, ow = 2000, 4000
    up = boxes / scale
    up[:, 0] = np.clip(up[:, 0], 0, ow)
    up[:, 1] = np.clip(up[:, 1], 0, oh)
    up[:, 2] = np.clip(up[:, 2], 0, ow)
    up[:, 3] = np.clip(up[:, 3], 0, oh)
    assert up[0].tolist() == [400.0, 200.0, 1600.0, 1000.0]
    # Over-bound coords clip into frame.
    big = np.array([[0.0, 0.0, 5000.0, 5000.0]], dtype=np.float32) / scale
    big[:, 2] = np.clip(big[:, 2], 0, ow)
    big[:, 3] = np.clip(big[:, 3], 0, oh)
    assert big[0, 2] == ow and big[0, 3] == oh
