"""ONNX Model Profiler — layer-level timing, memory, FLOPs estimation."""
import os
import time
import tempfile
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort


@dataclass
class LayerProfile:
    name: str
    op_type: str
    duration_us: float  # microseconds


@dataclass
class ProfileResult:
    total_infer_ms: float
    warmup_ms: float
    avg_infer_ms: float
    min_infer_ms: float
    max_infer_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    num_runs: int
    layer_profiles: list[LayerProfile]
    top_bottlenecks: list[LayerProfile]  # top-5 slowest layers
    num_parameters: int | None
    estimated_flops: int | None
    peak_memory_mb: float | None


def _estimate_flops(path: str) -> int | None:
    """Rough FLOPs estimation from ONNX graph (Conv, MatMul, Gemm)."""
    try:
        import onnx
        model = onnx.load(path, load_external_data=False)
    except Exception:
        return None

    # Build shape map from initializers
    shape_map = {}
    for init in model.graph.initializer:
        shape_map[init.name] = list(init.dims)

    total = 0
    for node in model.graph.node:
        if node.op_type == "Conv":
            # FLOPs ≈ 2 * K*K*Cin*Cout*Hout*Wout (approximate)
            if len(node.input) >= 2 and node.input[1] in shape_map:
                w = shape_map[node.input[1]]  # [Cout, Cin/g, kH, kW]
                if len(w) == 4:
                    total += 2 * w[0] * w[1] * w[2] * w[3] * 56 * 56  # rough spatial
        elif node.op_type in ("MatMul", "Gemm"):
            if len(node.input) >= 2 and node.input[1] in shape_map:
                w = shape_map[node.input[1]]
                if len(w) == 2:
                    total += 2 * w[0] * w[1]
    return total if total > 0 else None


def _count_params(path: str) -> int | None:
    try:
        import onnx
        model = onnx.load(path, load_external_data=False)
        total = 0
        for init in model.graph.initializer:
            s = 1
            for d in init.dims:
                s *= d
            total += s
        return total
    except Exception:
        return None


def _get_layer_profiles(path: str, input_feed: dict) -> list[LayerProfile]:
    """Run with ORT profiling enabled and parse layer timings."""
    try:
        opts = ort.SessionOptions()
        opts.enable_profiling = True
        opts.profile_file_prefix = os.path.join(tempfile.gettempdir(), "ssook_prof")
        sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        sess.run(None, input_feed)
        prof_file = sess.end_profiling()

        import json
        with open(prof_file, "r") as f:
            events = json.load(f)

        layers = []
        for ev in events:
            if ev.get("cat") == "Node" and "dur" in ev:
                name = ev.get("name", "")
                args = ev.get("args", {})
                op = args.get("op_name", "")
                layers.append(LayerProfile(name=name, op_type=op, duration_us=ev["dur"]))

        try:
            os.remove(prof_file)
        except Exception:
            pass
        return layers
    except Exception:
        return []


def profile_model(path: str, num_runs: int = 20, warmup: int = 3) -> ProfileResult:
    """Profile an ONNX model: latency stats + layer-level timing."""
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    # Build dummy input
    input_feed = {}
    for inp in sess.get_inputs():
        shape = [s if isinstance(s, int) and s > 0 else 1 for s in inp.shape]
        dtype = np.float32
        if "int64" in (inp.type or ""):
            dtype = np.int64
        elif "int32" in (inp.type or ""):
            dtype = np.int32
        input_feed[inp.name] = np.random.randn(*shape).astype(dtype)

    # Warmup
    t0 = time.perf_counter()
    for _ in range(warmup):
        sess.run(None, input_feed)
    warmup_ms = (time.perf_counter() - t0) * 1000.0 / max(warmup, 1)

    # Benchmark runs
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        sess.run(None, input_feed)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies = np.array(latencies)

    # Layer profiling
    layers = _get_layer_profiles(path, input_feed)
    layers.sort(key=lambda x: x.duration_us, reverse=True)
    top5 = layers[:5]

    return ProfileResult(
        total_infer_ms=float(latencies.sum()),
        warmup_ms=warmup_ms,
        avg_infer_ms=float(latencies.mean()),
        min_infer_ms=float(latencies.min()),
        max_infer_ms=float(latencies.max()),
        p50_ms=float(np.percentile(latencies, 50)),
        p95_ms=float(np.percentile(latencies, 95)),
        p99_ms=float(np.percentile(latencies, 99)),
        num_runs=num_runs,
        layer_profiles=layers,
        top_bottlenecks=top5,
        num_parameters=_count_params(path),
        estimated_flops=_estimate_flops(path),
        peak_memory_mb=None,
    )


def profile_to_dict(result: ProfileResult) -> dict:
    """Convert ProfileResult to JSON-serializable dict."""
    return {
        "avg_infer_ms": round(result.avg_infer_ms, 3),
        "min_infer_ms": round(result.min_infer_ms, 3),
        "max_infer_ms": round(result.max_infer_ms, 3),
        "p50_ms": round(result.p50_ms, 3),
        "p95_ms": round(result.p95_ms, 3),
        "p99_ms": round(result.p99_ms, 3),
        "warmup_ms": round(result.warmup_ms, 3),
        "num_runs": result.num_runs,
        "num_parameters": result.num_parameters,
        "estimated_flops": result.estimated_flops,
        "peak_memory_mb": result.peak_memory_mb,
        "top_bottlenecks": [
            {"name": l.name, "op_type": l.op_type, "duration_us": round(l.duration_us, 1)}
            for l in result.top_bottlenecks
        ],
        "layer_profiles": [
            {"name": l.name, "op_type": l.op_type, "duration_us": round(l.duration_us, 1)}
            for l in result.layer_profiles[:50]  # limit to top 50
        ],
    }
