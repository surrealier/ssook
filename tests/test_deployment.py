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
