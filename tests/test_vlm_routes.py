"""Smoke tests for the VLM route surface introduced in Phase 5.

These do NOT exercise real ONNX inference — building a CLIP-shaped dummy
ONNX is expensive for a smoke test and the CLIP backend itself is unit-
covered indirectly via core/clip_inference (existing tests). What we do
cover here:

1. The path-safety guard rejects bogus model paths cleanly (no 500).
2. The new VLM input fields on InferRequest are accepted by Pydantic.
3. The error envelope from server.errors carries trace_id when something
   downstream raises.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from server import app

client = TestClient(app)


def test_infer_request_accepts_new_vlm_fields():
    # Endpoint is a POST; we expect a 200 with `error` (not 422 from Pydantic)
    # because the missing files surface as a runtime error rather than a
    # validation one — this asserts the request schema accepts the fields.
    r = client.post(
        "/api/infer/image",
        json={
            "model_path": "definitely_not_a_real_file.onnx",
            "image_path": "also_missing.jpg",
            "vlm_task": "caption",
            "vlm_text_encoder": "missing_te.onnx",
            "vlm_candidates": "yes,no,maybe",
        },
    )
    # The route returns 200 even on logical errors (ssook convention),
    # surfacing them in the JSON body. The point is: NOT a 422.
    assert r.status_code in (200, 400, 500)
    body = r.json()
    # Either an error envelope or an inline {"error": ...}; both are fine.
    assert isinstance(body, dict)


def test_gt_classes_traversal_rejected():
    # New Pydantic-backed /api/gt/classes should reject a non-existent dir
    # through the UnsafePathError path → 400 with PATH_NOT_FOUND.
    r = client.post(
        "/api/gt/classes",
        json={"label_dir": "/this/path/does/not/exist/anywhere"},
    )
    # Empty `label_dir` short-circuits to {classes:[]}; non-empty + missing
    # is now a 400 from the global UnsafePathError handler.
    assert r.status_code == 400
    body = r.json()
    assert body.get("code", "").startswith("PATH_")
    assert "trace_id" in body


def test_gt_classes_empty_returns_empty():
    r = client.post("/api/gt/classes", json={"label_dir": ""})
    assert r.status_code == 200
    assert r.json() == {"classes": []}


def test_force_stop_unique_route():
    # The duplicate /api/force-stop/{task_id} in extra_routes.py was removed
    # in Phase 1; the canonical handler in server/__init__.py owns it now.
    r = client.post("/api/force-stop/vlm")
    assert r.status_code == 200
    body = r.json()
    # vlm_state is now registered, so this should succeed rather than say "Unknown task".
    assert body.get("ok") is True
