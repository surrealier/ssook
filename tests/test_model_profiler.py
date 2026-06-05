"""core/model_profiler.py unit tests — fixed-batch throughput + EP fallback.

Two groups:
  - TestBatchScalingMath: PURE LOGIC over profile_to_dict / _resolve_providers /
    ProfileResult. No ORT session is built, so these are cheap and run anywhere.
  - TestProfileModelOnnx: build a tiny ONNX model and call profile_model end to
    end (imports onnxruntime + onnx → heavier; run centrally, not memory-safe).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from core.model_profiler import (
    ProfileResult, profile_to_dict, _resolve_providers,
)


def _make_result(**overrides) -> ProfileResult:
    """A minimal ProfileResult; only the latency/batch fields matter here."""
    base = dict(
        total_infer_ms=100.0, warmup_ms=1.0,
        avg_infer_ms=10.0, min_infer_ms=9.0, max_infer_ms=11.0,
        p50_ms=10.0, p95_ms=10.5, p99_ms=11.0,
        num_runs=10, layer_profiles=[], top_bottlenecks=[],
        num_parameters=None, estimated_flops=None, peak_memory_mb=None,
    )
    base.update(overrides)
    return ProfileResult(**base)


# ── Pure-logic: fixed-batch throughput math (no ORT) ─────

class TestBatchScalingMath:
    def test_batch1_throughput_is_inverse_latency(self):
        # batch=1: per-sample latency == per-batch latency, FPS == 1000/avg.
        d = profile_to_dict(_make_result(avg_infer_ms=10.0, batch_size=1))
        assert d["batch_size"] == 1
        assert d["avg_latency_per_sample_ms"] == pytest.approx(10.0)
        assert d["latency_per_batch_ms"] == pytest.approx(10.0)
        assert d["throughput_fps"] == pytest.approx(100.0)  # 1000/10

    def test_fixed_batch_throughput_is_per_sample(self):
        # batch=4, a single sess.run() of 20ms processes 4 samples →
        # per-sample latency = 5ms, throughput = 4 * (1000/20) = 200 FPS.
        d = profile_to_dict(_make_result(avg_infer_ms=20.0, batch_size=4))
        assert d["batch_size"] == 4
        assert d["latency_per_batch_ms"] == pytest.approx(20.0)
        assert d["avg_latency_per_sample_ms"] == pytest.approx(5.0)
        assert d["throughput_fps"] == pytest.approx(200.0)
        # The throughput must equal batch * naive-FPS, not the naive FPS itself.
        naive_fps = 1000.0 / d["latency_per_batch_ms"]
        assert d["throughput_fps"] == pytest.approx(4 * naive_fps)

    def test_zero_latency_does_not_divide_by_zero(self):
        d = profile_to_dict(_make_result(avg_infer_ms=0.0, batch_size=2))
        assert d["throughput_fps"] == 0.0

    def test_batch_size_floored_at_one(self):
        # A bogus 0 batch must not blow up the per-sample math.
        d = profile_to_dict(_make_result(avg_infer_ms=10.0, batch_size=0))
        assert d["batch_size"] == 1
        assert d["throughput_fps"] == pytest.approx(100.0)

    def test_avg_infer_ms_kept_for_backward_compat(self):
        # Existing key must remain (additive change only).
        d = profile_to_dict(_make_result(avg_infer_ms=12.5, batch_size=8))
        assert d["avg_infer_ms"] == pytest.approx(12.5)


# ── Pure-logic: EP fallback signalling (no ORT) ──────────

class TestEpFallbackFields:
    def test_fallback_fields_serialized(self):
        r = _make_result(
            provider="CPUExecutionProvider",
            requested_provider="CUDAExecutionProvider",
            provider_fallback=True,
            warnings=["Requested CUDAExecutionProvider is unavailable — ran on CPUExecutionProvider."],
        )
        d = profile_to_dict(r)
        assert d["provider_fallback"] is True
        assert d["requested_provider"] == "CUDAExecutionProvider"
        assert d["provider"] == "CPUExecutionProvider"
        assert d["warnings"] and "CUDA" in d["warnings"][0]

    def test_no_fallback_default(self):
        d = profile_to_dict(_make_result())
        assert d["provider_fallback"] is False
        assert d["requested_provider"] is None
        assert d["warnings"] == []


class TestResolveProviders:
    def test_specific_ep_reports_requested(self):
        providers, requested = _resolve_providers("CUDAExecutionProvider")
        assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert requested == "CUDAExecutionProvider"

    def test_cpu_request_has_no_requested_ep(self):
        # Asking for CPU explicitly is never a "fallback" — requested is None.
        providers, requested = _resolve_providers("CPUExecutionProvider")
        assert providers == ["CPUExecutionProvider"]
        assert requested is None

    def test_auto_has_no_requested_ep(self):
        providers, requested = _resolve_providers("auto")
        assert requested is None
        assert isinstance(providers, list) and providers

    def test_list_passthrough_has_no_requested_ep(self):
        providers, requested = _resolve_providers(["CPUExecutionProvider"])
        assert providers == ["CPUExecutionProvider"]
        assert requested is None


# ── End-to-end: requires onnxruntime + onnx (run centrally) ──

def _make_fixed_batch_conv(path: str, batch: int):
    """Tiny Conv model whose input has a FIXED leading batch dim."""
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    import numpy as np
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [batch, 3, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    W = numpy_helper.from_array(np.random.randn(4, 3, 3, 3).astype(np.float32), "W")
    conv = helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    g = helper.make_graph([conv], "fixed_batch", [X], [Y], [W])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)
    return path


@pytest.fixture
def fixed_batch_model(tmp_path):
    return _make_fixed_batch_conv(os.path.join(str(tmp_path), "b4.onnx"), batch=4)


class TestProfileModelOnnx:
    """Heavier: imports onnxruntime via profile_model. Run centrally."""

    def test_fixed_batch_throughput_per_sample(self, fixed_batch_model):
        from core.model_profiler import profile_model
        d = profile_to_dict(profile_model(fixed_batch_model, num_runs=3, warmup=1))
        assert d["batch_size"] == 4
        naive_fps = 1000.0 / d["latency_per_batch_ms"]
        # throughput_fps ≈ 4 * (1000/avg_infer_ms), NOT the naive single-sample FPS.
        # throughput uses the raw mean while naive_fps uses the 3-decimal-rounded
        # latency field, so allow 1% (rounding is ~0.2% at sub-ms; a naive/buggy
        # impl would be off by 4x, which 1% still catches).
        assert d["throughput_fps"] == pytest.approx(4 * naive_fps, rel=1e-2)
        # Both values are independently rounded to 3 decimals; at sub-millisecond
        # scale that rounding (±0.0005 each) swamps a tight relative tolerance, so
        # compare with an absolute tolerance that covers the combined rounding gap.
        assert d["avg_latency_per_sample_ms"] == pytest.approx(d["latency_per_batch_ms"] / 4, abs=1.5e-3)

    def test_requested_unavailable_ep_yields_warning(self, fixed_batch_model):
        # In a CPU-only test env, CUDA is absent → fallback + warning.
        import onnxruntime as ort
        from core.model_profiler import profile_model
        if "CUDAExecutionProvider" in ort.get_available_providers():
            pytest.skip("CUDA EP present — cannot exercise the fallback path here")
        result = profile_model(fixed_batch_model, num_runs=3, warmup=1,
                               provider="CUDAExecutionProvider")
        assert result.provider_fallback is True
        assert result.provider == "CPUExecutionProvider"
        assert result.requested_provider == "CUDAExecutionProvider"
        assert result.warnings and any("CUDA" in w for w in result.warnings)
        d = profile_to_dict(result)
        assert d["provider_fallback"] is True

    def test_auto_provider_no_fallback(self, fixed_batch_model):
        from core.model_profiler import profile_model
        result = profile_model(fixed_batch_model, num_runs=3, warmup=1)  # provider=None → auto
        assert result.provider_fallback is False
        assert result.requested_provider is None
