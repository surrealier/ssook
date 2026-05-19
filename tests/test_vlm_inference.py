"""Unit-ish tests for core.vlm_inference (Phase 5 backend).

We can't load real ONNX weights here without committing fixture binaries.
What we *can* test cheaply: the factory rejects misconfiguration and the
CLIPCaptioner constructor raises a clear error when no text encoder is
provided. This catches regressions in the public surface of vlm_inference.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


def test_factory_requires_text_encoder():
    from core.vlm_inference import get_backend

    with pytest.raises(NotImplementedError) as exc:
        get_backend("some_image_encoder.onnx", text_encoder=None)
    assert "CLIP" in str(exc.value)


def test_factory_rejects_missing_files(tmp_path):
    from core.vlm_inference import get_backend

    # Both files missing → FileNotFoundError on the image encoder first.
    with pytest.raises(FileNotFoundError):
        get_backend(str(tmp_path / "img_enc.onnx"),
                    text_encoder=str(tmp_path / "txt_enc.onnx"))


def test_captioner_requires_text_encoder():
    from core.vlm_inference import CLIPCaptioner

    with pytest.raises(ValueError):
        CLIPCaptioner("any_image.onnx", "")
