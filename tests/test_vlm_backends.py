"""Tests for the pluggable VLM backend layer (core.vlm_inference) and the
backend-aware route surface (server.model_routes).

We never download multi-GB weights here:
- CLIP is dependency-free and tested through make_backend's validation only
  (constructing a real CLIPBackend needs ONNX fixtures, out of scope).
- The transformers wiring test stubs out the model/processor load via
  monkeypatch, and live inference is double-gated behind importorskip AND
  SSOOK_VLM_LIVE=1 so CI stays offline.
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core import vlm_inference
from core.vlm_inference import list_backends, make_backend
from server import app

client = TestClient(app)


# ── list_backends() ──────────────────────────────────────────────────────
def test_list_backends_includes_clip_available():
    backends = {b["name"]: b for b in list_backends()}
    assert "clip" in backends
    assert backends["clip"]["available"] is True
    assert backends["clip"]["requires_text_encoder"] is True
    assert backends["clip"]["generative"] is False
    assert backends["clip"]["missing_deps"] == []


def test_list_backends_availability_matches_find_spec():
    backends = {b["name"]: b for b in list_backends()}
    tf_expected = (importlib.util.find_spec("transformers") is not None
                   and importlib.util.find_spec("torch") is not None)
    assert backends["transformers"]["available"] is tf_expected
    openai_expected = (importlib.util.find_spec("httpx") is not None
                       or importlib.util.find_spec("requests") is not None)
    assert backends["openai"]["available"] is openai_expected
    # Generative flags are static descriptors.
    assert backends["transformers"]["generative"] is True
    assert backends["openai"]["generative"] is True


# ── make_backend() validation ────────────────────────────────────────────
def test_make_backend_unknown_backend():
    with pytest.raises(ValueError) as exc:
        make_backend({"backend": "does-not-exist"})
    assert "Unknown VLM backend" in str(exc.value)


def test_make_backend_clip_requires_text_encoder():
    with pytest.raises(ValueError) as exc:
        make_backend({"backend": "clip", "model_path": "img_enc.onnx"})
    assert "text_encoder" in str(exc.value)


def test_make_backend_clip_missing_files(tmp_path):
    # Image encoder path does not exist → FileNotFoundError with a clear msg.
    with pytest.raises(FileNotFoundError):
        make_backend({
            "backend": "clip",
            "model_path": str(tmp_path / "img.onnx"),
            "text_encoder": str(tmp_path / "txt.onnx"),
        })


def test_make_backend_transformers_requires_model_id():
    with pytest.raises(ValueError) as exc:
        make_backend({"backend": "transformers"})
    assert "model_id" in str(exc.value)


def test_make_backend_openai_requires_endpoint():
    with pytest.raises(ValueError) as exc:
        make_backend({"backend": "openai", "model_id": "qwen2-vl"})
    assert "endpoint_url" in str(exc.value)


def test_make_backend_openai_requires_model_id():
    with pytest.raises(ValueError) as exc:
        make_backend({"backend": "openai", "endpoint_url": "http://localhost:8000/v1"})
    assert "model_id" in str(exc.value)


# ── get_backend back-compat ──────────────────────────────────────────────
def test_get_backend_requires_text_encoder():
    with pytest.raises(NotImplementedError) as exc:
        vlm_inference.get_backend("img.onnx", text_encoder=None)
    assert "CLIP" in str(exc.value)


# ── Route surface ────────────────────────────────────────────────────────
def test_get_vlm_backends_route():
    r = client.get("/api/vlm/backends")
    assert r.status_code == 200
    body = r.json()
    assert "backends" in body and isinstance(body["backends"], list)
    assert isinstance(body["cuda"], bool)
    names = {b["name"] for b in body["backends"]}
    assert "clip" in names


def test_infer_openai_bogus_endpoint_is_inline_error(tmp_path):
    # A valid-looking image + openai backend pointed at a dead endpoint must
    # surface as an inline {"error": ...} (HTTP 200, ssook convention), not a
    # 422/404 from the framework.
    import cv2
    img = tmp_path / "frame.jpg"
    cv2.imwrite(str(img), np.zeros((8, 8, 3), dtype=np.uint8))
    r = client.post("/api/infer/image", json={
        "model_path": "served-model",
        "image_path": str(img),
        "model_type": "vlm",
        "backend": "openai",
        "model_id": "qwen2-vl",
        "endpoint_url": "http://127.0.0.1:9/v1",  # port 9 = discard, refuses
        "vlm_task": "caption",
    })
    assert r.status_code == 200
    body = r.json()
    assert "error" in body
    assert "VLM inference failed" in body["error"]


def test_infer_transformers_without_model_id_is_validation_error(tmp_path):
    import cv2
    img = tmp_path / "frame.jpg"
    cv2.imwrite(str(img), np.zeros((8, 8, 3), dtype=np.uint8))
    r = client.post("/api/infer/image", json={
        "model_path": "",
        "image_path": str(img),
        "model_type": "vlm",
        "backend": "transformers",
        "vlm_task": "caption",
    })
    assert r.status_code == 200
    body = r.json()
    assert "error" in body
    assert "model_id" in body["error"]


def test_infer_api_key_never_logged(tmp_path, caplog):
    # The api_key must not leak into logs even on the error path.
    import cv2
    img = tmp_path / "frame.jpg"
    cv2.imwrite(str(img), np.zeros((8, 8, 3), dtype=np.uint8))
    secret = "sk-supersecret-do-not-log"
    with caplog.at_level("DEBUG"):
        client.post("/api/infer/image", json={
            "model_path": "served-model",
            "image_path": str(img),
            "model_type": "vlm",
            "backend": "openai",
            "model_id": "qwen2-vl",
            "endpoint_url": "http://127.0.0.1:9/v1",
            "api_key": secret,
            "vlm_task": "caption",
        })
    assert secret not in caplog.text


# ── transformers wiring (stubbed; no weights download) ───────────────────
def test_transformers_backend_wiring_stubbed(monkeypatch):
    """Exercise the transformers code path without loading real weights.

    Skips when transformers/torch are absent. We monkeypatch the lazy model
    + processor load so describe()/answer() return a canned string, proving
    the chat-template → generate → decode plumbing is wired correctly.
    """
    pytest.importorskip("transformers")
    pytest.importorskip("torch")

    class _FakeProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            # Confirm the contract: image + text content blocks.
            content = messages[0]["content"]
            kinds = {c["type"] for c in content}
            assert kinds == {"image", "text"}
            return "PROMPT"

        def __call__(self, text=None, images=None, return_tensors=None):
            return _FakeInputs()

        def batch_decode(self, seqs, skip_special_tokens=True, clean_up_tokenization_spaces=True):
            return ["a fake caption"]

    class _FakeInputs(dict):
        def __init__(self):
            super().__init__(input_ids=[[1, 2, 3]])

        def to(self, device):
            return self

    class _FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            # Returns prompt tokens + one new token so trimming works.
            return [[1, 2, 3, 4]]

    monkeypatch.setattr(vlm_inference.TransformersBackend, "is_available",
                        staticmethod(lambda: True))

    def _fake_init(self, model_id, *, device=None):
        import torch
        self._torch = torch
        self._model_id = model_id
        self._device = "cpu"
        self._processor = _FakeProcessor()
        self._model = _FakeModel()

    monkeypatch.setattr(vlm_inference.TransformersBackend, "__init__", _fake_init)

    backend = vlm_inference.TransformersBackend("fake/model")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    out = backend.describe(frame, "Describe this.")
    assert out == "a fake caption"
    ans = backend.answer(frame, "Is it day?", candidates=["yes", "no"])
    assert ans == "a fake caption"


@pytest.mark.skipif(
    os.environ.get("SSOOK_VLM_LIVE") != "1",
    reason="live transformers inference is opt-in via SSOOK_VLM_LIVE=1",
)
def test_transformers_live_inference():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    import cv2
    # Small VLM so the download stays manageable for an opt-in live run.
    backend = make_backend({"backend": "transformers",
                            "model_id": "Qwen/Qwen2-VL-2B-Instruct"})
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (50, 50), (0, 0, 255), -1)
    out = backend.describe(frame, "Describe this image briefly.")
    assert isinstance(out, str) and out.strip()
