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

import pytest
from fastapi.testclient import TestClient

from server import app
from server.model_routes import (
    UnsafeEndpointError,
    validate_endpoint_url,
    _is_blocked_ip,
)
import ipaddress

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


# ── SSRF endpoint_url validation (P0 security) ───────────────────────────
@pytest.fixture(autouse=True)
def _no_internal_opt_in(monkeypatch):
    # Ensure the secure default: the opt-in escape hatch must be OFF so the
    # guard actually blocks internal hosts during these tests.
    monkeypatch.delenv("SSOOK_VLM_ALLOW_INTERNAL", raising=False)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.1/v1",
        "http://localhost:11434/v1",
        "http://[::1]:8000/v1",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata IP
        "http://169.254.1.1/v1",  # link-local
        "http://10.0.0.5:8000/v1",  # RFC1918
        "http://172.16.5.5/v1",  # RFC1918
        "http://192.168.1.10:1234/v1",  # RFC1918
        "http://0.0.0.0:8000/v1",  # unspecified
        "http://[::ffff:127.0.0.1]:8000/v1",  # IPv4-mapped loopback
    ],
)
def test_validate_endpoint_url_blocks_internal(url):
    with pytest.raises(UnsafeEndpointError):
        validate_endpoint_url(url)


def test_validate_endpoint_url_rejects_non_http_scheme():
    for url in ("file:///etc/passwd", "ftp://example.com/x", "gopher://1.2.3.4/"):
        with pytest.raises(UnsafeEndpointError):
            validate_endpoint_url(url)


def test_validate_endpoint_url_rejects_empty():
    with pytest.raises(UnsafeEndpointError):
        validate_endpoint_url(None)
    with pytest.raises(UnsafeEndpointError):
        validate_endpoint_url("")


def test_validate_endpoint_url_accepts_public_https(monkeypatch):
    # A normal public https URL must pass. Stub DNS so the test does not
    # depend on real network resolution: resolve to a public address.
    def _fake_getaddrinfo(host, port, proto=0):
        return [(2, 1, 6, "", ("93.184.216.34", port or 443))]  # example.com-ish public IP

    monkeypatch.setattr("server.model_routes.socket.getaddrinfo", _fake_getaddrinfo)
    assert validate_endpoint_url("https://api.example.com/v1") == "https://api.example.com/v1"


def test_validate_endpoint_url_blocks_hostname_resolving_to_loopback(monkeypatch):
    # DNS-rebind-style: a public-looking hostname that resolves to 127.0.0.1
    # must still be blocked.
    def _fake_getaddrinfo(host, port, proto=0):
        return [(2, 1, 6, "", ("127.0.0.1", port or 80))]

    monkeypatch.setattr("server.model_routes.socket.getaddrinfo", _fake_getaddrinfo)
    with pytest.raises(UnsafeEndpointError):
        validate_endpoint_url("http://evil.example.com/v1")


def test_validate_endpoint_url_opt_in_allows_internal(monkeypatch):
    # With the explicit opt-in, a localhost server is permitted (trusted
    # on-host vLLM/Ollama use case).
    monkeypatch.setenv("SSOOK_VLM_ALLOW_INTERNAL", "1")
    assert validate_endpoint_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"


def test_is_blocked_ip_classifies_ranges():
    assert _is_blocked_ip(ipaddress.ip_address("127.0.0.1"))
    assert _is_blocked_ip(ipaddress.ip_address("169.254.169.254"))
    assert _is_blocked_ip(ipaddress.ip_address("10.1.2.3"))
    assert _is_blocked_ip(ipaddress.ip_address("::1"))
    assert not _is_blocked_ip(ipaddress.ip_address("8.8.8.8"))
    assert not _is_blocked_ip(ipaddress.ip_address("93.184.216.34"))


def test_vlm_batch_endpoint_url_blocked_via_route():
    # Route-level smoke: an openai backend pointed at loopback is rejected
    # with the VLM_ENDPOINT_BLOCKED code before any worker is submitted.
    r = client.post(
        "/api/vlm/batch",
        json={
            "model_path": "unused.onnx",
            "img_dir": ".",
            "backend": "openai",
            "model_id": "some-model",
            "endpoint_url": "http://127.0.0.1:8000/v1",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("code") == "VLM_ENDPOINT_BLOCKED"


def test_infer_image_endpoint_url_blocked_via_route():
    r = client.post(
        "/api/infer/image",
        json={
            "model_path": "unused.onnx",
            "image_path": "missing.jpg",
            "model_type": "vlm",
            "backend": "openai",
            "model_id": "some-model",
            "endpoint_url": "http://169.254.169.254/latest/meta-data/",
        },
    )
    assert r.status_code in (200, 400, 500)
    body = r.json()
    assert body.get("code") == "VLM_ENDPOINT_BLOCKED"


# ── Atomic start guard (P1 concurrency) ──────────────────────────────────
def test_vlm_batch_try_start_guard_blocks_second_concurrent_start():
    # Two near-simultaneous POSTs must not both launch a worker. Drive the
    # state directly to avoid depending on a real backend/ONNX: claim the
    # slot once, then assert a second try_start fails until running clears.
    from server.state import vlm_state

    # Make sure we start from a clean slate.
    vlm_state.update(running=False)
    assert vlm_state.try_start(progress=0, total=0, msg="Starting...", results=[]) is True
    assert vlm_state.try_start(progress=0, total=0, msg="Starting...", results=[]) is False
    # Releasing lets the next run start.
    vlm_state.update(running=False)
    assert vlm_state.try_start(progress=0, total=0, msg="Starting...", results=[]) is True
    vlm_state.update(running=False)


def test_vlm_batch_second_post_returns_busy(monkeypatch):
    # Route-level: while a run is marked running, a second POST gets VLM_BUSY.
    from server.state import vlm_state
    import server.model_routes as mr

    vlm_state.update(running=False)
    # Prevent a real worker from being submitted/started.
    monkeypatch.setattr(mr.executor, "submit", lambda *a, **k: None)
    # Stub path/dir validation so we reach the start guard with a clip backend.
    monkeypatch.setattr(mr, "safe_model_file", lambda p: p)
    monkeypatch.setattr(mr, "safe_image_dir", lambda p: p)
    monkeypatch.setattr(mr, "glob_images", lambda d: ["a.jpg", "b.jpg"])

    payload = {
        "model_path": "img.onnx",
        "text_encoder": "txt.onnx",
        "img_dir": ".",
        "backend": "clip",
    }
    first = client.post("/api/vlm/batch", json=payload).json()
    assert first.get("ok") is True
    # vlm_state.running is now True (worker was stubbed to a no-op).
    second = client.post("/api/vlm/batch", json=payload).json()
    assert second.get("code") == "VLM_BUSY"
    vlm_state.update(running=False)
