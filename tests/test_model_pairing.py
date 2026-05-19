"""Unit tests for core.model_pairing — CLIP encoder auto-pairing."""
from core.model_pairing import _flip_role, _my_role, find_partner


def test_flip_image_to_text():
    assert _flip_role("clip_image_encoder") == "clip_text_encoder"


def test_flip_text_to_image():
    assert _flip_role("clip_text_encoder") == "clip_image_encoder"


def test_flip_visual_to_textual():
    assert _flip_role("clip_visual_b32") == "clip_textual_b32"


def test_my_role_text():
    assert _my_role("model_text_encoder") == "text"


def test_my_role_image():
    assert _my_role("vision_encoder_b16") == "image"


def test_my_role_ambiguous():
    assert _my_role("clip_b32") is None


def test_find_partner_by_filename(tmp_path):
    a = tmp_path / "clip_image_encoder.onnx"
    b = tmp_path / "clip_text_encoder.onnx"
    a.write_bytes(b"\x00")
    b.write_bytes(b"\x00")
    r = find_partner(str(a))
    # _classify_from_io returns unknown since these aren't real ONNX,
    # but the *filename* heuristic should still match.
    assert r["partner_path"] is not None
    assert r["partner_path"].lower().endswith("clip_text_encoder.onnx")


def test_find_partner_missing(tmp_path):
    a = tmp_path / "lone_clip_image_encoder.onnx"
    a.write_bytes(b"\x00")
    r = find_partner(str(a))
    assert r["partner_path"] is None


def test_find_partner_nonexistent():
    r = find_partner("/does/not/exist.onnx")
    assert r["partner_path"] is None
    assert "not found" in r["reason"].lower()
