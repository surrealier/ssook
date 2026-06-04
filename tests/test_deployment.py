"""Tests for the Deployment / export module (core.deployment + routes)."""
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


def _make_linear_onnx(path: str, in_features=8, out_features=4):
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, in_features])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, out_features])
    W = numpy_helper.from_array(np.random.randn(in_features, out_features).astype(np.float32), "W")
    B = numpy_helper.from_array(np.random.randn(out_features).astype(np.float32), "B")
    mm = helper.make_node("MatMul", ["X", "W"], ["mm"])
    add = helper.make_node("Add", ["mm", "B"], ["Y"])
    graph = helper.make_graph([mm, add], "linear", [X], [Y], [W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="ssook_deploy_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def linear_onnx(tmp_dir):
    return _make_linear_onnx(os.path.join(tmp_dir, "linear.onnx"))


# ── Registry ───────────────────────────────────────────
from core.deployment.registry import BaseExportTarget, ExportTargetRegistry, registry


class TestExportRegistry:
    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseExportTarget()

    def test_builtin_targets_present(self):
        names = {t.name for t in registry.list_all()}
        for expected in ("onnxruntime_cpu", "onnxruntime_cuda", "onnxruntime_directml",
                         "tensorrt", "openvino", "coreml", "torchscript"):
            assert expected in names

    def test_to_dict_shape(self):
        for t in registry.list_all():
            d = t.to_dict()
            for key in ("name", "label", "category", "available", "unavailable_reason", "output_ext"):
                assert key in d
            assert isinstance(d["available"], bool)

    def test_to_dict_accepts_and_uses_gpu(self):
        by_name = {t.name: t.to_dict() for t in registry.list_all()}
        # New fields present and correctly typed on every target.
        for d in by_name.values():
            assert isinstance(d["accepts"], list)
            assert isinstance(d["uses_gpu"], bool)
        # .pt-only targets.
        assert by_name["coreml"]["accepts"] == ["pt"]
        assert by_name["torchscript"]["accepts"] == ["pt"]
        # GPU-building targets.
        assert by_name["tensorrt"]["uses_gpu"] is True
        assert by_name["onnxruntime_cuda"]["uses_gpu"] is True
        # CPU ONNX takes both sources and is not a GPU build.
        assert by_name["onnxruntime_cpu"]["uses_gpu"] is False
        assert set(by_name["onnxruntime_cpu"]["accepts"]) == {"pt", "onnx"}

    def test_register_duplicate_raises(self):
        r = ExportTargetRegistry()

        class T(BaseExportTarget):
            name = "dup"
            def export(self, *a, **k): return {}
        r.register(T())
        with pytest.raises(ValueError):
            r.register(T())

    def test_onnx_cpu_always_available(self):
        cpu = registry.get("onnxruntime_cpu")
        assert cpu is not None
        assert cpu.is_available()[0] is True


# ── input-size parsing ─────────────────────────────────
from core.deployment.exporters import resolve_imgsz


class TestResolveImgsz:
    def test_int_string(self):
        assert resolve_imgsz({"input_size": "640"}) == 640

    def test_hxw_string(self):
        assert resolve_imgsz({"input_size": "640x480"}) == [640, 480]

    def test_square_list(self):
        assert resolve_imgsz({"input_size": [640, 640]}) == 640

    def test_zero_is_none(self):
        assert resolve_imgsz({"input_size": 0}) is None
        assert resolve_imgsz({}) is None


# ── path safety extensions ─────────────────────────────
from server.path_safety import safe_weights_file, safe_export_output, UnsafePathError


class TestPathSafety:
    def test_weights_accepts_pt_and_onnx(self, tmp_dir):
        for ext in (".pt", ".onnx"):
            p = os.path.join(tmp_dir, f"m{ext}")
            open(p, "wb").close()
            result = safe_weights_file(p)  # resolve() may expand 8.3 short paths
            assert result.endswith(ext) and os.path.isfile(result)

    def test_weights_rejects_exe(self, tmp_dir):
        p = os.path.join(tmp_dir, "m.exe")
        open(p, "wb").close()
        with pytest.raises(UnsafePathError):
            safe_weights_file(p)

    def test_weights_rejects_missing(self, tmp_dir):
        with pytest.raises(UnsafePathError):
            safe_weights_file(os.path.join(tmp_dir, "nope.pt"))

    def test_export_output_allows_runtime_exts(self, tmp_dir):
        for ext in (".onnx", ".engine", ".xml", ".mlpackage", ".torchscript"):
            out = os.path.join(tmp_dir, f"out{ext}")
            assert safe_export_output(out).endswith(ext)

    def test_export_output_rejects_bad_ext(self, tmp_dir):
        with pytest.raises(UnsafePathError):
            safe_export_output(os.path.join(tmp_dir, "out.bat"))


# ── ONNX export (works wherever onnx+onnxruntime are present) ──
class TestOnnxExportFromOnnx:
    def test_fp32_copy_and_validate(self, linear_onnx, tmp_dir):
        from core.deployment.runner import run_export
        out = os.path.join(tmp_dir, "exported.onnx")
        result = run_export({
            "model_path": linear_onnx, "target": "onnxruntime_cpu",
            "output_path": out, "opts": {"precision": "fp32"},
        })
        assert "error" not in result, result
        assert os.path.isfile(out)
        assert result["provider"] == "CPUExecutionProvider"
        assert result["validated"] is True
        assert result["output_size_mb"] >= 0
        assert os.path.isfile(result["manifest_path"])

    def test_unknown_target(self, linear_onnx, tmp_dir):
        from core.deployment.runner import run_export
        result = run_export({
            "model_path": linear_onnx, "target": "nope",
            "output_path": os.path.join(tmp_dir, "x.onnx"), "opts": {},
        })
        assert "error" in result


# ── artifact validation + opset readback ───────────────
class TestValidateArtifact:
    def test_valid_onnx_cpu_target_ok(self, linear_onnx):
        from core.deployment.runner import validate_artifact
        cpu = registry.get("onnxruntime_cpu")
        verdict = validate_artifact(linear_onnx, cpu)
        # CPUExecutionProvider is always available — no warning, ok True.
        assert verdict["ok"] is True
        assert "warning" not in verdict

    def test_missing_artifact(self, tmp_dir):
        from core.deployment.runner import validate_artifact
        cpu = registry.get("onnxruntime_cpu")
        verdict = validate_artifact(os.path.join(tmp_dir, "nope.onnx"), cpu)
        assert verdict["ok"] is False

    def test_onnx_actual_opset_reads_17(self, linear_onnx):
        from core.deployment.runner import onnx_actual_opset
        assert onnx_actual_opset(linear_onnx) == 17

    def test_onnx_actual_opset_unreadable(self, tmp_dir):
        from core.deployment.runner import onnx_actual_opset
        assert onnx_actual_opset(os.path.join(tmp_dir, "missing.onnx")) is None


# ── int8 calibration data.yaml auto-gen ────────────────
class TestCalibrationYaml:
    def test_generates_yaml_for_bare_dir(self, tmp_dir):
        from core.deployment.exporters import _calibration_data_yaml
        calib = os.path.join(tmp_dir, "calib_images")
        os.makedirs(calib, exist_ok=True)
        yaml_path = _calibration_data_yaml(calib)
        assert yaml_path != calib
        assert os.path.isfile(yaml_path)
        assert yaml_path.endswith((".yaml", ".yml"))
        with open(yaml_path, encoding="utf-8") as f:
            text = f.read()
        # The generated yaml points train/val at the calibration dir.
        assert os.path.abspath(calib).replace("\\", "/") in text.replace("\\", "/") \
            or os.path.basename(calib) in text

    def test_reuses_existing_yaml(self, tmp_dir):
        from core.deployment.exporters import _calibration_data_yaml
        calib = os.path.join(tmp_dir, "calib_with_yaml")
        os.makedirs(calib, exist_ok=True)
        existing = os.path.join(calib, "data.yaml")
        open(existing, "w").close()
        assert _calibration_data_yaml(calib) == existing


# ── manifest ───────────────────────────────────────────
from core.deployment.manifest import build_manifest, write_manifest, _session_recommendation


class TestManifest:
    def test_directml_session_note(self):
        rec = _session_recommendation("DmlExecutionProvider", {"device_id": 0})
        assert rec["session_options"]["enable_mem_pattern"] is False
        # CPU fallback is always appended.
        assert "CPUExecutionProvider" in rec["providers"]

    def test_build_and_write(self, linear_onnx, tmp_dir):
        target = registry.get("onnxruntime_cpu")
        export_result = {"output_path": linear_onnx, "output_size_mb": 0.01,
                         "provider": "CPUExecutionProvider", "provider_options": {}}
        opts = {"opset": 17, "batch": 1, "precision": "fp32"}
        manifest = build_manifest(linear_onnx, target, opts, export_result)
        assert manifest["schema"] == "ssook.deploy/1"
        assert manifest["target"]["provider"] == "CPUExecutionProvider"
        assert manifest["config"]["opset"] == 17
        path = write_manifest(manifest, linear_onnx)
        assert os.path.isfile(path)
        assert path.endswith(".deploy.json")


# ── route smoke ────────────────────────────────────────
class TestDeployRoutes:
    def _client(self):
        from fastapi.testclient import TestClient
        from server import app
        return TestClient(app)

    def test_targets_endpoint(self):
        r = self._client().get("/api/deploy/targets")
        assert r.status_code == 200
        names = {t["name"] for t in r.json()["targets"]}
        assert "tensorrt" in names and "onnxruntime_cpu" in names

    def test_run_rejects_bad_path(self):
        r = self._client().post("/api/deploy/run", json={
            "model_path": "/no/such/model.onnx", "target": "onnxruntime_cpu"})
        body = r.json()
        assert "error" in body and body.get("code", "").startswith("PATH_")

    def test_run_unknown_target(self, linear_onnx):
        r = self._client().post("/api/deploy/run", json={
            "model_path": linear_onnx, "target": "bogus"})
        assert "error" in r.json()

    def test_targets_endpoint_exposes_accepts_uses_gpu(self):
        targets = self._client().get("/api/deploy/targets").json()["targets"]
        by_name = {t["name"]: t for t in targets}
        assert by_name["coreml"]["accepts"] == ["pt"]
        assert by_name["tensorrt"]["uses_gpu"] is True

    def test_run_rejects_incompatible_source(self, linear_onnx):
        # CoreML accepts only .pt; an .onnx source must be rejected up front
        # with a clear source-requirement message (before any export starts).
        r = self._client().post("/api/deploy/run", json={
            "model_path": linear_onnx, "target": "coreml"})
        body = r.json()
        assert "error" in body
        assert "requires" in body["error"] and "source" in body["error"]

    def test_run_rejects_bad_opset(self, linear_onnx):
        r = self._client().post("/api/deploy/run", json={
            "model_path": linear_onnx, "target": "onnxruntime_cpu", "opset": 99})
        assert "error" in r.json() and "opset" in r.json()["error"]

    def test_run_rejects_zero_batch(self, linear_onnx):
        r = self._client().post("/api/deploy/run", json={
            "model_path": linear_onnx, "target": "onnxruntime_cpu", "batch": 0})
        assert "error" in r.json() and "batch" in r.json()["error"]

    def test_run_rejects_zero_workspace(self, linear_onnx):
        r = self._client().post("/api/deploy/run", json={
            "model_path": linear_onnx, "target": "onnxruntime_cpu", "workspace_gb": 0})
        assert "error" in r.json() and "workspace_gb" in r.json()["error"]


# ── EP-isolated subprocess export routing (pure logic, mocked manager) ──────
# These exercise the DECISION of when run_export dispatches to an EP venv vs
# running in-process. No real worker is spawned and no backend (openvino /
# coremltools) is imported — ep_manager is monkeypatched and the backend probe
# is forced. Safe to run in the base/torch env.
from core.deployment import runner as _runner


class TestEpRouteDecision:
    """_ep_route_for: which (target, env) combinations pick a subprocess EP."""

    def _route(self, monkeypatch, target_name, *, importable, venv_available):
        target = registry.get(target_name)
        monkeypatch.setattr(_runner, "_backend_importable", lambda mod: importable)
        # ep_manager.is_ep_available is consulted lazily inside _ep_route_for.
        from core import ep_manager
        monkeypatch.setattr(ep_manager, "is_ep_available", lambda key: venv_available)
        return _runner._ep_route_for(target)

    def test_openvino_routes_when_backend_missing_and_venv_present(self, monkeypatch):
        assert self._route(monkeypatch, "openvino",
                           importable=False, venv_available=True) == "openvino"

    def test_coreml_routes_when_backend_missing_and_venv_present(self, monkeypatch):
        assert self._route(monkeypatch, "coreml",
                           importable=False, venv_available=True) == "coreml"

    def test_no_route_when_backend_importable_in_process(self, monkeypatch):
        # Backend already here -> keep the faster in-process export.
        assert self._route(monkeypatch, "openvino",
                           importable=True, venv_available=True) is None

    def test_no_route_when_venv_absent(self, monkeypatch):
        # Backend missing AND no venv -> in-process (yields today's clear
        # "backend not installed" result).
        assert self._route(monkeypatch, "openvino",
                           importable=False, venv_available=False) is None

    def test_in_process_targets_never_route(self, monkeypatch):
        # ONNX-family / tensorrt / torchscript have no EP venv mapping; they
        # must run in-process regardless of probe/venv state.
        for name in ("onnxruntime_cpu", "onnxruntime_cuda", "onnxruntime_directml",
                     "tensorrt", "torchscript"):
            assert self._route(monkeypatch, name,
                               importable=False, venv_available=True) is None

    def test_route_survives_ep_manager_failure(self, monkeypatch):
        # An ep_manager probe blowing up must never block the in-process path.
        target = registry.get("openvino")
        monkeypatch.setattr(_runner, "_backend_importable", lambda mod: False)
        from core import ep_manager

        def _boom(key):
            raise RuntimeError("ep probe exploded")
        monkeypatch.setattr(ep_manager, "is_ep_available", _boom)
        assert _runner._ep_route_for(target) is None


class TestRunExportDispatch:
    """run_export wiring: dispatch to the mocked venv runner, fallback, and the
    _in_process recursion guard. No subprocess is ever launched."""

    def test_routes_to_venv_runner_and_returns_its_result(self, linear_onnx, monkeypatch):
        # Force the openvino route, then assert run_export hands off to the
        # subprocess helper and returns its payload verbatim (no in-process export).
        monkeypatch.setattr(_runner, "_ep_route_for", lambda target: "openvino")
        from core import ep_manager
        calls = {}

        def _fake_run(ep_key, config, on_progress=None):
            calls["ep_key"] = ep_key
            calls["config"] = config
            return {"ok": True, "via": "subprocess", "target": config.get("target")}
        monkeypatch.setattr(ep_manager, "run_export_in_venv", _fake_run)

        result = _runner.run_export({
            "model_path": linear_onnx, "target": "openvino",
            "output_path": "/tmp/out.xml", "opts": {"precision": "fp32"},
        })
        assert result == {"ok": True, "via": "subprocess", "target": "openvino"}
        assert calls["ep_key"] == "openvino"
        # The routing flag must NOT leak into the forwarded config.
        assert "_in_process" not in calls["config"]
        assert calls["config"]["target"] == "openvino"

    def test_in_process_flag_skips_routing(self, linear_onnx, monkeypatch):
        # When _in_process is set (as the worker sets it), routing is bypassed
        # entirely — _ep_route_for must not even be consulted.
        def _should_not_run(target):
            raise AssertionError("routing must be skipped when _in_process is set")
        monkeypatch.setattr(_runner, "_ep_route_for", _should_not_run)
        # onnxruntime_cpu runs a real (fast, in-process) export and succeeds.
        out = os.path.join(os.path.dirname(linear_onnx), "viacpu.onnx")
        result = _runner.run_export({
            "model_path": linear_onnx, "target": "onnxruntime_cpu",
            "output_path": out, "opts": {"precision": "fp32"}, "_in_process": True,
        })
        assert "error" not in result, result
        assert os.path.isfile(out)

    def test_dispatch_failure_falls_back_to_in_process(self, linear_onnx, monkeypatch):
        # If the subprocess helper raises, run_export must fall through to the
        # in-process attempt rather than hard-failing. Route a CPU export (which
        # would normally never route) and make the helper explode.
        monkeypatch.setattr(_runner, "_ep_route_for", lambda target: "openvino")
        from core import ep_manager

        def _boom(ep_key, config, on_progress=None):
            raise RuntimeError("subprocess launch failed")
        monkeypatch.setattr(ep_manager, "run_export_in_venv", _boom)

        out = os.path.join(os.path.dirname(linear_onnx), "fallback.onnx")
        result = _runner.run_export({
            "model_path": linear_onnx, "target": "onnxruntime_cpu",
            "output_path": out, "opts": {"precision": "fp32"},
        })
        # Fell back to the real in-process CPU export -> success.
        assert "error" not in result, result
        assert os.path.isfile(out)
