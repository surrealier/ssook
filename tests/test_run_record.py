"""Tests for core.run_record reproducible-run helper."""
import os

import yaml


def test_recorder_writes_yaml(tmp_path):
    from core.run_record import RunRecorder
    with RunRecorder(run_type="eval", output_dir=tmp_path,
                     inputs={"a": 1, "b": "x"}) as rec:
        rec.note("ran clean")

    p = tmp_path / "run.yaml"
    assert p.exists()
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert data["run_type"] == "eval"
    assert data["inputs"] == {"a": 1, "b": "x"}
    assert data["notes"] == ["ran clean"]
    assert data["duration_ms"] >= 0
    assert data["trace_id"]
    assert "env" in data


def test_recorder_captures_error(tmp_path):
    from core.run_record import RunRecorder
    try:
        with RunRecorder(run_type="bench", output_dir=tmp_path):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    data = yaml.safe_load((tmp_path / "run.yaml").read_text(encoding="utf-8"))
    assert "RuntimeError: boom" in data["error"]


def test_model_meta_hashes(tmp_path):
    from core.run_record import model_meta
    f = tmp_path / "fake.onnx"
    f.write_bytes(b"abc123")
    m = model_meta(str(f))
    assert m["size_mb"] >= 0
    assert m["sha256"]
    assert m["mtime"]


def test_env_snapshot_has_python_version():
    from core.run_record import env_snapshot
    e = env_snapshot()
    assert "python" in e and e["python"]
    assert e["ssook_version"]


def test_safe_handles_pydantic(tmp_path):
    from pydantic import BaseModel
    from core.run_record import _safe

    class M(BaseModel):
        a: int = 1
        b: str = "x"

    out = _safe(M())
    assert out == {"a": 1, "b": "x"}
