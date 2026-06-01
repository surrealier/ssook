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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from server import system_routes
from core.app_config import AppConfig, CustomModelType


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
