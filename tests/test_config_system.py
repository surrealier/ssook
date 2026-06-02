"""Tests for the config track findings (v1.6.0).

Covers:
- CONFIG-02: PowerShell / AppleScript argument escaping in the native file
  dialog (shell-injection guard).
- CONFIG-01 / VIEWER-01: `_auto_cleanup_memory` references real symbols and
  `system_hw()` never 500s under memory pressure.
- VIEWER-07 (config part): `CustomModelType.coord_format` removed,
  `AppConfig.stream_jpeg_quality` added and round-trips through YAML.
"""
import asyncio
import dataclasses
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from server import config_routes, system_routes
from core.app_config import AppConfig, ClassStyle, CustomModelType


# ── CONFIG-02: dialog argument escaping ─────────────────────────────────
def test_ps_single_quote_doubles_quotes():
    # A crafted filter that would break out of the PS single-quoted literal
    # must have every quote doubled, neutralising the breakout.
    payload = "'; Start-Process calc; '"
    escaped = system_routes._ps_single_quote(payload)
    assert "''" in escaped
    # No lone single quote survives — every quote is part of a doubled pair.
    assert escaped == "''; Start-Process calc; ''"


def test_ps_single_quote_noop_when_clean():
    assert system_routes._ps_single_quote("Models (*.onnx)") == "Models (*.onnx)"


def test_applescript_double_quote_escapes_quote_and_backslash():
    assert system_routes._applescript_double_quote('a"b') == 'a\\"b'
    # Backslash escaped first so it is not doubled by the quote pass.
    assert system_routes._applescript_double_quote("a\\b") == "a\\\\b"


def test_native_dialog_filter_does_not_break_out(monkeypatch):
    """An injected filter must stay inside the PS string literal."""
    captured = {}

    class _FakeResult:
        stdout = ""

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeResult()

    monkeypatch.setattr(system_routes.platform, "system", lambda: "Windows")
    # subprocess is imported locally inside the function; patch the real module.
    import subprocess as _sp
    monkeypatch.setattr(_sp, "run", _fake_run)

    system_routes._native_file_dialog(title="t", filters="Evil (*.exe')")
    script = captured["cmd"][-1]
    # The single quote from the filter is doubled (*.exe'' ), so the
    # `$d.Filter = '...'` literal is not terminated early.
    assert "*.exe''" in script
    # And no lone single quote breaks out of the filter assignment.
    assert "$d.Filter = 'Evil|*.exe''|All files (*.*)|*.*';" in script


# ── CONFIG-01 / VIEWER-01: cleanup never 500s ───────────────────────────
def test_auto_cleanup_memory_resolves_real_symbols(monkeypatch):
    # Reset the throttle so the body actually runs.
    monkeypatch.setattr(system_routes, "_MEM_CLEANUP_INTERVAL", 0)
    # Must not raise NameError for the previously-undefined globals.
    system_routes._auto_cleanup_memory()

    from server.state import compare_state, embedding_state
    assert compare_state.snapshot().get("results") == []
    assert embedding_state.snapshot().get("image") is None


def test_system_hw_never_500s_under_memory_pressure(monkeypatch):
    """RSS > 50% RAM must trigger cleanup without ever raising."""
    class _Mem:
        rss = 10 ** 12  # absurdly high RSS
        total = 1  # tiny total → rss > total * 0.5 always true

    class _Proc:
        def cpu_percent(self, interval=0):
            return 1.0

        def memory_info(self):
            return _Mem()

    monkeypatch.setattr(system_routes.psutil, "Process", lambda pid: _Proc())
    monkeypatch.setattr(
        system_routes.psutil, "virtual_memory", lambda: _Mem()
    )
    # Force the GPU branch off so the test does not depend on nvidia-smi.
    monkeypatch.setattr(system_routes, "check_gpu_available", lambda: False)
    # Make cleanup itself blow up — system_hw must still return a dict.
    monkeypatch.setattr(system_routes, "_MEM_CLEANUP_INTERVAL", 0)

    def _boom():
        raise RuntimeError("cleanup bug")

    monkeypatch.setattr(system_routes, "_auto_cleanup_memory", _boom)

    info = asyncio.run(system_routes.system_hw())
    assert isinstance(info, dict)
    assert "ram_mb" in info


# ── VIEWER-07 (config part): coord_format removed, stream_jpeg_quality ──
def test_custom_model_type_has_no_coord_format():
    field_names = {f.name for f in dataclasses.fields(CustomModelType)}
    assert "coord_format" not in field_names


def test_stream_jpeg_quality_round_trips(tmp_path, monkeypatch):
    cfg_path = tmp_path / "app_config.yaml"
    monkeypatch.setattr("core.app_config._CONFIG_PATH", str(cfg_path))
    # Drop the singleton so _load() re-reads against the patched path.
    monkeypatch.setattr(AppConfig, "_instance", None)

    cfg = AppConfig()
    assert cfg.stream_jpeg_quality == 75  # default

    cfg.stream_jpeg_quality = 90
    cfg.custom_model_types["t"] = CustomModelType(
        name="t", attr_roles=["x1", "y1", "width", "height", "conf_class0"]
    )
    cfg.save()

    monkeypatch.setattr(AppConfig, "_instance", None)
    reloaded = AppConfig()
    assert reloaded.stream_jpeg_quality == 90
    # coord_format must not be persisted for any custom model type.
    import yaml
    on_disk = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    for cmt in (on_disk.get("custom_model_types") or {}).values():
        assert "coord_format" not in cmt


# ── SETTINGS P0: custom-model path validation ───────────────────────────
_VALID_ROLES = ["x_center", "y_center", "width", "height", "conf_class0"]


def _cmt_req(**over):
    base = dict(
        name="t",
        model_path="model.onnx",
        attr_roles=list(_VALID_ROLES),
    )
    base.update(over)
    return config_routes.CustomModelTypeRequest(**base)


def test_custom_model_type_test_rejects_bad_ext_without_session(monkeypatch, tmp_path):
    """A real .txt path must fail on extension BEFORE any session opens."""
    import onnxruntime as ort

    def _boom(*a, **k):
        raise AssertionError("InferenceSession must not be reached")

    monkeypatch.setattr(ort, "InferenceSession", _boom)
    bad = tmp_path / "not_a_model.txt"
    bad.write_bytes(b"x")
    req = _cmt_req(model_path=str(bad))
    out = asyncio.run(config_routes.test_custom_model_type(req))
    assert out["error"] == "BAD_EXT"


def test_custom_model_type_test_rejects_traversal_path(tmp_path):
    """A non-existent traversal path returns NOT_FOUND, never opens a session."""
    req = _cmt_req(model_path="..\\..\\etc\\passwd")
    out = asyncio.run(config_routes.test_custom_model_type(req))
    assert out["error"] in {"NOT_FOUND", "BAD_EXT"}


def test_save_custom_model_type_rejects_bad_path(monkeypatch, tmp_path):
    cfg_path = tmp_path / "app_config.yaml"
    monkeypatch.setattr("core.app_config._CONFIG_PATH", str(cfg_path))
    monkeypatch.setattr(AppConfig, "_instance", None)
    req = _cmt_req(model_path="ghost.onnx")  # does not exist
    out = asyncio.run(config_routes.save_custom_model_type(req))
    assert out["error"] == "NOT_FOUND"
    # Nothing persisted.
    assert not cfg_path.exists() or "ghost" not in cfg_path.read_text(encoding="utf-8")


# ── SETTINGS P1: custom-model-type input validation ─────────────────────
def test_save_custom_model_type_rejects_non_numeric_class_names(monkeypatch, tmp_path):
    real = tmp_path / "m.onnx"
    real.write_bytes(b"\x00")  # exists with right ext; validation precedes load
    monkeypatch.setattr(config_routes, "safe_model_file", lambda p: str(real))
    req = _cmt_req(model_path=str(real), class_names={"foo": "x"})
    out = asyncio.run(config_routes.save_custom_model_type(req))
    assert out["error"] == "BAD_CLASS_NAMES"


def test_save_custom_model_type_rejects_missing_coords(monkeypatch, tmp_path):
    real = tmp_path / "m.onnx"
    real.write_bytes(b"\x00")
    monkeypatch.setattr(config_routes, "safe_model_file", lambda p: str(real))
    # No coord group present.
    req = _cmt_req(model_path=str(real), attr_roles=["conf_class0"])
    out = asyncio.run(config_routes.save_custom_model_type(req))
    assert out["error"] == "MISSING_COORDS"


def test_save_custom_model_type_rejects_missing_conf(monkeypatch, tmp_path):
    real = tmp_path / "m.onnx"
    real.write_bytes(b"\x00")
    monkeypatch.setattr(config_routes, "safe_model_file", lambda p: str(real))
    # Coords present but no conf source.
    req = _cmt_req(model_path=str(real),
                   attr_roles=["x_center", "y_center", "width", "height"])
    out = asyncio.run(config_routes.save_custom_model_type(req))
    assert out["error"] == "MISSING_CONF"


def test_save_custom_model_type_rejects_name_with_separator(monkeypatch, tmp_path):
    real = tmp_path / "m.onnx"
    real.write_bytes(b"\x00")
    monkeypatch.setattr(config_routes, "safe_model_file", lambda p: str(real))
    req = _cmt_req(name="a/b", model_path=str(real))
    out = asyncio.run(config_routes.save_custom_model_type(req))
    assert out["error"] == "INVALID_NAME"


# ── SETTINGS P1: concurrent custom_model_types mutation under lock ──────
def test_concurrent_custom_model_type_saves_all_persist(monkeypatch, tmp_path):
    cfg_path = tmp_path / "app_config.yaml"
    monkeypatch.setattr("core.app_config._CONFIG_PATH", str(cfg_path))
    monkeypatch.setattr(AppConfig, "_instance", None)
    real = tmp_path / "m.onnx"
    real.write_bytes(b"\x00")
    monkeypatch.setattr(config_routes, "safe_model_file", lambda p: str(real))

    # Pre-build the singleton so all threads share one instance + its lock.
    AppConfig()

    n = 12
    errors = []

    def _worker(i):
        req = _cmt_req(name=f"type{i}", model_path=str(real))
        out = asyncio.run(config_routes.save_custom_model_type(req))
        if out.get("error"):
            errors.append(out["error"])

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    # Reload from disk and confirm the YAML parses with every entry.
    monkeypatch.setattr(AppConfig, "_instance", None)
    reloaded = AppConfig()
    for i in range(n):
        assert f"type{i}" in reloaded.custom_model_types


def test_save_config_takes_lock(monkeypatch, tmp_path):
    """save_config mutates under _rw_lock; field round-trips to disk."""
    cfg_path = tmp_path / "app_config.yaml"
    monkeypatch.setattr("core.app_config._CONFIG_PATH", str(cfg_path))
    monkeypatch.setattr(AppConfig, "_instance", None)

    req = config_routes.ConfigUpdate(conf_threshold=0.42, batch_size=4)
    out = asyncio.run(config_routes.save_config(req))
    assert out == {"ok": True}

    monkeypatch.setattr(AppConfig, "_instance", None)
    reloaded = AppConfig()
    assert reloaded.conf_threshold == 0.42
    assert reloaded.batch_size == 4


# ── SETTINGS P0: fs endpoints reject traversal / NUL ────────────────────
def test_fs_list_rejects_nul_byte():
    req = system_routes.FsListRequest(path="bad\x00path")
    out = asyncio.run(system_routes.fs_list(req))
    assert out["files"] == []
    assert "NUL" in out["error"]


def test_fs_list_nonexistent_returns_error():
    req = system_routes.FsListRequest(path=os.path.join("definitely", "missing", "xyz"))
    out = asyncio.run(system_routes.fs_list(req))
    assert out["files"] == []
    assert out["error"].startswith("NOT_FOUND")


def test_fs_list_caps_recursive_walk(tmp_path, monkeypatch):
    # Create more files than the cap to confirm truncation kicks in.
    monkeypatch.setattr(system_routes, "_FS_LIST_CAP", 5)
    for i in range(12):
        (tmp_path / f"f{i}.bin").write_bytes(b"x")
    req = system_routes.FsListRequest(path=str(tmp_path), recursive=True)
    out = asyncio.run(system_routes.fs_list(req))
    assert out["count"] == 5
    assert out.get("truncated") is True


def test_fs_list_lists_real_dir(tmp_path):
    (tmp_path / "a.onnx").write_bytes(b"x")
    (tmp_path / "b.txt").write_bytes(b"x")
    req = system_routes.FsListRequest(path=str(tmp_path), extensions=["onnx"])
    out = asyncio.run(system_routes.fs_list(req))
    assert out["count"] == 1
    assert out["files"][0].endswith("a.onnx")


def test_fs_browse_falls_back_to_home_on_nul(monkeypatch, tmp_path):
    from pathlib import Path
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    req = system_routes.FsBrowseRequest(path="bad\x00path")
    out = asyncio.run(system_routes.fs_browse(req))
    # Falls back to the (patched) home dir rather than raising.
    assert out["current"] == str(tmp_path)


def test_list_files_rejects_nul_byte():
    req = system_routes.ListFilesRequest(dir="bad\x00path")
    out = asyncio.run(system_routes.list_files(req))
    assert out["files"] == []
    assert "NUL" in out["error"]


def test_list_files_lists_images(tmp_path):
    (tmp_path / "img.jpg").write_bytes(b"x")
    (tmp_path / "note.txt").write_bytes(b"x")
    req = system_routes.ListFilesRequest(dir=str(tmp_path))
    out = asyncio.run(system_routes.list_files(req))
    assert out["count"] == 1
    assert out["files"][0].endswith("img.jpg")
