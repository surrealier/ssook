"""core/model_inspector.py — EP real-probe + parse-once unit tests.

These tests build a tiny ONNX model in-process (no fixtures, no cv2) so they run
in any env with onnx + onnxruntime. They verify observable behavior:
  - the EP compatibility now reflects a REAL session probe (not just the heuristic
    op table), reporting per-EP loads + a failure reason instead of a bare bool;
  - a provider that ORT cannot assign is reported unavailable WITH a non-empty
    reason, and never crashes the inspect;
  - the ONNX graph is parsed once per inspect (not 3x);
  - opset is read from the default ai.onnx domain, ignoring custom domains.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

import core.model_inspector as mi


def _make_tiny_onnx(tmp_path, *, extra_opset=None) -> str:
    """A 2-node Relu->Add ONNX model saved to disk. Returns the path."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    relu = helper.make_node("Relu", ["X"], ["h"])
    add = helper.make_node("Add", ["h", "h"], ["Y"])
    graph = helper.make_graph([relu, add], "tiny", [x], [y])
    opset_imports = [helper.make_opsetid("", 13)]
    if extra_opset is not None:
        opset_imports = extra_opset
    model = helper.make_model(graph, opset_imports=opset_imports)
    model.ir_version = 8
    path = os.path.join(str(tmp_path), "tiny.onnx")
    onnx.save(model, path)
    return path


class TestEpRealProbe:
    def test_cpu_probe_reports_loaded_true_with_no_error(self, tmp_path):
        path = _make_tiny_onnx(tmp_path)
        info = mi.inspect_model(path)  # default test_eps=True

        cpu = info.ep_compatibility["CPUExecutionProvider"]
        # CPU is the universal fallback — the real probe must confirm it loads.
        assert cpu["loads"] is True
        assert cpu["load_error"] is None
        assert "CPUExecutionProvider" in cpu["assigned_providers"]
        # compatible_eps reflects the verified probe, so CPU must be present.
        assert "CPUExecutionProvider" in info.compatible_eps

    def test_unavailable_provider_reports_false_with_reason(self, tmp_path):
        path = _make_tiny_onnx(tmp_path)
        # An EP that is not installed/assignable: ORT silently falls back to CPU
        # (it does NOT raise), so the bare-session signal would lie. _ep_loads
        # must catch the fallback and report loaded=False WITH a reason.
        loaded, load_error, assigned = mi._ep_loads(path, "TensorrtExecutionProvider")
        if loaded:
            pytest.skip("TensorRT is actually available in this env")
        assert loaded is False
        assert load_error  # non-empty reason — not swallowed to a bare False
        assert isinstance(load_error, str) and len(load_error) > 0

    def test_probe_never_raises_when_provider_errors(self, tmp_path, monkeypatch):
        """A provider whose session construction raises must be reported as
        unavailable-with-reason, never propagate out of inspect_model."""
        path = _make_tiny_onnx(tmp_path)
        real_session = ort.InferenceSession

        def _boom(model_path, providers=None, **kwargs):
            # CPU (used for I/O/metadata) must still work; only fail the probe of
            # a synthetic GPU EP so the inspect's own session is unaffected.
            if providers == ["CUDAExecutionProvider"]:
                raise RuntimeError("simulated CUDA init failure")
            return real_session(model_path, providers=providers, **kwargs)

        monkeypatch.setattr(ort, "InferenceSession", _boom)
        # Force CUDA into the candidate list regardless of the real env.
        monkeypatch.setattr(
            ort, "get_available_providers",
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        info = mi.inspect_model(path)  # must NOT raise

        cuda = info.ep_compatibility["CUDAExecutionProvider"]
        assert cuda["loads"] is False
        assert "simulated CUDA init failure" in cuda["load_error"]
        assert "CUDAExecutionProvider" not in info.compatible_eps
        # The inspect still produced real structure info despite the EP failure.
        assert info.num_nodes == 2
        assert "CPUExecutionProvider" in info.compatible_eps

    def test_ep_loads_returns_triple_contract(self, tmp_path):
        path = _make_tiny_onnx(tmp_path)
        result = mi._ep_loads(path, "CPUExecutionProvider")
        assert isinstance(result, tuple) and len(result) == 3
        loaded, load_error, assigned = result
        assert loaded is True
        assert load_error is None
        assert isinstance(assigned, list)


class TestParseOnce:
    def test_onnx_loaded_exactly_once(self, tmp_path, monkeypatch):
        """The graph helpers used to each onnx.load(path) — 3x per inspect plus
        the ORT session. After the refactor the proto is parsed exactly once."""
        path = _make_tiny_onnx(tmp_path)
        calls = {"n": 0}
        real_load = onnx.load

        def _counting_load(*args, **kwargs):
            calls["n"] += 1
            return real_load(*args, **kwargs)

        monkeypatch.setattr(onnx, "load", _counting_load)
        # Disable the EP probe so we only measure parse cost, not session loads.
        info = mi.inspect_model(path, test_eps=False)

        assert calls["n"] == 1
        # Structure fields are still populated correctly from the single parse.
        assert info.num_nodes == 2
        assert info.node_op_counts == {"Relu": 1, "Add": 1}
        assert info.num_parameters == 0  # no initializers in the tiny graph
        assert info.opset_version == 13

    def test_parse_failure_degrades_gracefully(self, tmp_path, monkeypatch):
        """A corrupt graph (onnx.load raises) must not abort the inspect — I/O
        info still comes from the ORT session, counts fall back to neutral."""
        path = _make_tiny_onnx(tmp_path)
        monkeypatch.setattr(
            onnx, "load",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("corrupt proto")),
        )
        info = mi.inspect_model(path, test_eps=False)
        assert info.num_nodes == 0
        assert info.node_op_counts == {}
        assert info.num_parameters is None
        assert info.opset_version == 0
        # ORT session still gave us the input/output tensor info.
        assert len(info.inputs) == 1 and info.inputs[0].name == "X"


class TestDefaultOpset:
    def test_default_domain_opset_selected_over_custom(self, tmp_path):
        # com.microsoft contrib opset listed FIRST; the standard ai.onnx (17)
        # must still be reported, not the order-dependent opset_import[0].
        model = onnx.load(
            _make_tiny_onnx(
                tmp_path,
                extra_opset=[
                    helper.make_opsetid("com.microsoft", 1),
                    helper.make_opsetid("", 17),
                ],
            )
        )
        opset, *_ = mi._get_opset_ir(model)
        assert opset == 17

    def test_default_opset_reversed_order(self, tmp_path):
        model = onnx.load(
            _make_tiny_onnx(
                tmp_path,
                extra_opset=[
                    helper.make_opsetid("", 17),
                    helper.make_opsetid("com.microsoft", 1),
                ],
            )
        )
        opset, *_ = mi._get_opset_ir(model)
        assert opset == 17
