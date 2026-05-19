"""Smoke tests for server.path_safety — the boundary guard introduced in
Phase 2. Verifies traversal/NUL/extension rejection without touching the
network stack.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from server.path_safety import (
    UnsafePathError,
    safe_image_dir,
    safe_image_file,
    safe_label_dir,
    safe_model_file,
    safe_path,
)


def test_empty_rejected():
    with pytest.raises(UnsafePathError) as exc:
        safe_path("")
    assert exc.value.code == "EMPTY"


def test_nul_rejected():
    with pytest.raises(UnsafePathError) as exc:
        safe_path("foo\x00bar")
    assert exc.value.code == "NUL"


def test_must_exist_missing(tmp_path):
    with pytest.raises(UnsafePathError) as exc:
        safe_path(str(tmp_path / "does_not_exist"), must_exist=True)
    assert exc.value.code == "NOT_FOUND"


def test_must_be_file_on_dir(tmp_path):
    with pytest.raises(UnsafePathError) as exc:
        safe_path(str(tmp_path), must_exist=True, must_be_file=True)
    assert exc.value.code == "NOT_FILE"


def test_must_be_dir_on_file(tmp_path):
    fp = tmp_path / "x.txt"
    fp.write_text("hi")
    with pytest.raises(UnsafePathError) as exc:
        safe_path(str(fp), must_exist=True, must_be_dir=True)
    assert exc.value.code == "NOT_DIR"


def test_extension_allowlist(tmp_path):
    fp = tmp_path / "model.pt"
    fp.write_bytes(b"\x00")
    with pytest.raises(UnsafePathError) as exc:
        safe_model_file(str(fp))
    assert exc.value.code == "BAD_EXT"


def test_extension_ok(tmp_path):
    fp = tmp_path / "model.onnx"
    fp.write_bytes(b"\x00")
    resolved = safe_model_file(str(fp))
    assert resolved.endswith("model.onnx")


def test_image_file_ok(tmp_path):
    fp = tmp_path / "pic.jpg"
    fp.write_bytes(b"\x00")
    assert safe_image_file(str(fp)).endswith("pic.jpg")


def test_image_dir_ok(tmp_path):
    assert Path(safe_image_dir(str(tmp_path))).resolve() == tmp_path.resolve()


def test_label_dir_ok(tmp_path):
    assert Path(safe_label_dir(str(tmp_path))).resolve() == tmp_path.resolve()


def test_traversal_within_allowed_roots(tmp_path):
    # When `roots` is supplied, a sibling outside the root is rejected.
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    with pytest.raises(UnsafePathError) as exc:
        safe_path(str(outside), roots=[str(root)])
    assert exc.value.code == "OUT_OF_ROOT"


def test_within_root_ok(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    child = root / "x.txt"
    child.write_text("x")
    out = safe_path(str(child), roots=[str(root)])
    assert Path(out).resolve() == child.resolve()
