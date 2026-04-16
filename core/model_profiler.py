"""ONNX Model Profiler — layer-level timing, memory, FLOPs, diagnostics."""
import os
import time
import tempfile
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort


# ── Data classes ────────────────────────────────────────

@dataclass
class LayerProfile:
    name: str
    op_type: str
    duration_us: float


@dataclass
class ProfileResult:
    # Latency
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
    top_bottlenecks: list[LayerProfile]
    num_parameters: int | None
    estimated_flops: int | None
    peak_memory_mb: float | None
    # Extended (v1.5)
    op_type_summary: list[dict] = field(default_factory=list)
    graph_depth: int = 0
    weight_memory_mb: float = 0.0
    peak_activation_mb: float = 0.0
    total_memory_mb: float = 0.0
    total_macs: int = 0
    flops_per_layer: list[dict] = field(default_factory=list)
    bottleneck_diagnosis: list[dict] = field(default_factory=list)
    optimization_suggestions: list[str] = field(default_factory=list)
    quantizable_ratio: float = 0.0
    non_quantizable_ops: list[str] = field(default_factory=list)
    estimated_int8_speedup: float = 1.0
    input_info: list[dict] = field(default_factory=list)
    output_info: list[dict] = field(default_factory=list)


# ── Helpers ─────────────────────────────────────────────

def _resolve_dynamic_dim(dim, axis: int, ndim: int) -> int:
    if isinstance(dim, int) and dim > 0:
        return dim
    name = str(dim).lower() if dim is not None else ""
    if "batch" in name: return 1
    if "height" in name or "width" in name: return 640
    if "sequence" in name or "seq" in name or "length" in name: return 64
    if "channel" in name: return 3
    if ndim == 4: return [1, 3, 640, 640][axis] if axis < 4 else 64
    if ndim == 3: return [1, 64, 256][axis] if axis < 3 else 64
    if ndim == 2: return 1 if axis == 0 else 256
    return 1 if axis == 0 else 64


_DTYPE_MAP = {
    "tensor(float)": np.float32, "tensor(float16)": np.float16,
    "tensor(double)": np.float64, "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8, "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
}


def _build_dummy_feed(sess) -> dict:
    feed = {}
    for inp in sess.get_inputs():
        ndim = len(inp.shape)
        shape = [_resolve_dynamic_dim(s, i, ndim) for i, s in enumerate(inp.shape)]
        dtype = _DTYPE_MAP.get(inp.type, np.float32)
        if np.issubdtype(dtype, np.floating):
            feed[inp.name] = np.random.randn(*shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            feed[inp.name] = np.random.randint(0, 255, shape, dtype=dtype)
        else:
            feed[inp.name] = np.random.randn(*shape).astype(np.float32)
    return feed


def _get_layer_profiles(path: str, input_feed: dict) -> list[LayerProfile]:
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
        try: os.remove(prof_file)
        except Exception: pass
        return layers
    except Exception:
        return []


# ── Graph analysis ──────────────────────────────────────

_DTYPE_BYTES = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2}

_QUANTIZABLE_OPS = {
    "Conv", "MatMul", "Gemm", "ConvTranspose", "Linear",
    "Add", "Mul", "Relu", "Clip", "Sigmoid", "MaxPool", "AveragePool",
    "GlobalAveragePool", "Concat", "Reshape", "Transpose", "Flatten",
}

_MEMORY_BOUND_OPS = {"Reshape", "Transpose", "Concat", "Flatten", "Squeeze", "Unsqueeze", "Gather", "Slice", "Split", "Pad"}


def _analyze_graph(path: str, layer_profiles: list[LayerProfile]):
    """Deep graph analysis: op summary, memory, FLOPs, diagnostics."""
    try:
        import onnx
        from onnx import numpy_helper, TensorProto
        model = onnx.load(path, load_external_data=False)
    except Exception:
        return {}, [], 0, 0.0, 0.0, 0, [], [], [], 0.0, [], 1.0

    graph = model.graph

    # Shape map from initializers
    shape_map = {}
    param_map = {}  # name -> param count
    for init in graph.initializer:
        dims = list(init.dims)
        shape_map[init.name] = dims
        p = 1
        for d in dims: p *= d
        param_map[init.name] = p

    # Shape map from value_info (shape inference results)
    vi_shapes = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.type.HasField("tensor_type"):
            tt = vi.type.tensor_type
            if tt.HasField("shape"):
                dims = []
                for d in tt.shape.dim:
                    dims.append(d.dim_value if d.dim_value > 0 else 640)
                vi_shapes[vi.name] = dims

    # Try shape inference
    try:
        import onnx
        inferred = onnx.shape_inference.infer_shapes(model, data_prop=True)
        for vi in inferred.graph.value_info:
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
                dims = []
                for d in vi.type.tensor_type.shape.dim:
                    dims.append(d.dim_value if d.dim_value > 0 else 640)
                vi_shapes[vi.name] = dims
    except Exception:
        pass

    # Build timing map from layer profiles
    timing_map = {}
    for lp in layer_profiles:
        timing_map[lp.name] = lp.duration_us

    # Op type summary + FLOPs + diagnostics
    op_stats = {}  # op_type -> {count, total_time, params}
    total_flops = 0
    total_macs = 0
    flops_per_layer = []
    weight_bytes = 0
    peak_act_bytes = 0
    act_sizes = []
    non_quant_ops = set()
    quant_count = 0
    total_op_count = 0
    diagnosis = []
    suggestions = set()

    for node in graph.node:
        op = node.op_type
        total_op_count += 1

        # Op stats
        if op not in op_stats:
            op_stats[op] = {"count": 0, "total_time_us": 0.0, "param_count": 0}
        op_stats[op]["count"] += 1
        t_us = timing_map.get(node.name, 0.0)
        op_stats[op]["total_time_us"] += t_us

        # Params for this node
        node_params = 0
        for inp_name in node.input:
            if inp_name in param_map:
                node_params += param_map[inp_name]
                weight_bytes += param_map[inp_name] * 4  # assume fp32
        op_stats[op]["param_count"] += node_params

        # Quantizable check
        if op in _QUANTIZABLE_OPS:
            quant_count += 1
        elif op not in {"BatchNormalization", "LayerNormalization", "InstanceNormalization",
                        "Softmax", "Dropout", "Identity", "Shape", "Cast", "ConstantOfShape", "Constant"}:
            non_quant_ops.add(op)

        # FLOPs estimation
        node_flops = 0
        if op == "Conv" and len(node.input) >= 2 and node.input[1] in shape_map:
            w = shape_map[node.input[1]]
            if len(w) == 4:
                out_shape = vi_shapes.get(node.output[0], [1, w[0], 56, 56])
                h_out = out_shape[2] if len(out_shape) > 2 else 56
                w_out = out_shape[3] if len(out_shape) > 3 else 56
                # groups
                groups = 1
                for attr in node.attribute:
                    if attr.name == "group": groups = attr.i
                cin_per_g = w[1]
                node_flops = 2 * w[0] * cin_per_g * w[2] * w[3] * h_out * w_out
                # Bias
                if len(node.input) >= 3 and node.input[2]:
                    node_flops += w[0] * h_out * w_out
        elif op == "ConvTranspose" and len(node.input) >= 2 and node.input[1] in shape_map:
            w = shape_map[node.input[1]]
            if len(w) == 4:
                out_shape = vi_shapes.get(node.output[0], [1, w[1], 112, 112])
                h_out = out_shape[2] if len(out_shape) > 2 else 112
                w_out = out_shape[3] if len(out_shape) > 3 else 112
                node_flops = 2 * w[0] * w[1] * w[2] * w[3] * h_out * w_out
        elif op in ("MatMul", "Gemm") and len(node.input) >= 2 and node.input[1] in shape_map:
            w = shape_map[node.input[1]]
            if len(w) == 2:
                # For MatMul: (M,K)x(K,N) -> 2*M*K*N; use batch=1 if no info
                inp_shape = vi_shapes.get(node.input[0], [1, w[0]])
                m = inp_shape[-2] if len(inp_shape) >= 2 else 1
                node_flops = 2 * m * w[0] * w[1]
        elif op == "BatchNormalization":
            out_shape = vi_shapes.get(node.output[0], [])
            if out_shape:
                elems = 1
                for d in out_shape: elems *= d
                node_flops = 4 * elems  # mean, var, normalize, scale+shift

        node_macs = node_flops // 2
        total_flops += node_flops
        total_macs += node_macs

        if node_flops > 0:
            flops_per_layer.append({
                "name": node.name, "op_type": op,
                "flops": node_flops, "macs": node_macs,
            })

        # Activation memory estimation
        for out_name in node.output:
            if out_name in vi_shapes:
                s = vi_shapes[out_name]
                elems = 1
                for d in s: elems *= d
                act_sizes.append(elems * 4)  # fp32

        # Bottleneck diagnosis
        if t_us > 100:  # only diagnose layers > 100μs
            category = "compute" if op in ("Conv", "MatMul", "Gemm", "ConvTranspose") else (
                "memory" if op in _MEMORY_BOUND_OPS else "other")
            severity = "high" if t_us > 1000 else ("medium" if t_us > 500 else "low")
            suggestion = ""

            if op == "Conv" and len(node.input) >= 2 and node.input[1] in shape_map:
                w = shape_map[node.input[1]]
                if len(w) == 4:
                    groups = 1
                    for attr in node.attribute:
                        if attr.name == "group": groups = attr.i
                    if groups == 1 and w[1] > 16 and w[2] >= 3:
                        suggestion = f"Conv({w[2]}×{w[3]}, {w[1]}→{w[0]}): consider Depthwise Separable Conv for ~{w[2]*w[3]}x FLOPs reduction"
                        suggestions.add("Replace standard Conv with Depthwise Separable Conv where possible")
                    if w[0] >= 512 and w[1] >= 512:
                        suggestions.add("Large Conv channels detected — consider channel pruning")
            elif op in ("Transpose", "Reshape"):
                suggestion = f"{op} is memory-bound — check if it can be fused with adjacent ops"
                suggestions.add("Fuse consecutive Reshape/Transpose operations")
            elif op == "Cast":
                suggestion = "Unnecessary Cast may indicate mixed-precision issues"
                suggestions.add("Remove unnecessary Cast operations")

            if t_us > 200:
                diagnosis.append({
                    "layer": node.name, "op_type": op, "time_us": round(t_us, 1),
                    "category": category, "severity": severity, "suggestion": suggestion,
                })

    # Peak activation memory (rough: max of running sum over sliding window)
    if act_sizes:
        peak_act_bytes = max(act_sizes) * 3  # rough: ~3 concurrent tensors

    # Quantization suggestions
    q_ratio = quant_count / max(total_op_count, 1)
    if q_ratio > 0.7:
        suggestions.add(f"Model is {q_ratio*100:.0f}% quantizable — INT8 quantization recommended for ~2-3x speedup")
    est_speedup = 1.0 + (q_ratio * 2.0)  # rough: up to 3x for fully quantizable

    # BN fold suggestion
    has_bn = any(n.op_type == "BatchNormalization" for n in graph.node)
    has_conv = any(n.op_type == "Conv" for n in graph.node)
    if has_bn and has_conv:
        suggestions.add("BatchNormalization can be folded into Conv for inference (use onnxsim)")

    # Graph depth (longest path)
    depth = 0
    output_to_node = {}
    for node in graph.node:
        for o in node.output:
            output_to_node[o] = node
    # Simple BFS depth from outputs
    visited = set()
    def _depth(name, d=0):
        if name in visited or name not in output_to_node: return d
        visited.add(name)
        node = output_to_node[name]
        return max((_depth(i, d+1) for i in node.input), default=d)
    for out in graph.output:
        depth = max(depth, _depth(out.name))

    # Op summary sorted by time
    op_summary = []
    total_time = sum(s["total_time_us"] for s in op_stats.values())
    for op, s in sorted(op_stats.items(), key=lambda x: -x[1]["total_time_us"]):
        op_summary.append({
            "op_type": op, "count": s["count"],
            "total_time_us": round(s["total_time_us"], 1),
            "time_pct": round(s["total_time_us"] / max(total_time, 1) * 100, 1),
            "param_count": s["param_count"],
        })

    # Sort diagnosis by time
    diagnosis.sort(key=lambda x: -x["time_us"])

    return (
        op_summary, flops_per_layer, depth,
        weight_bytes / 1024 / 1024, peak_act_bytes / 1024 / 1024,
        total_flops, total_macs, diagnosis, sorted(suggestions),
        q_ratio, sorted(non_quant_ops), est_speedup,
    )


# ── Main profiler ───────────────────────────────────────

def profile_model(path: str, num_runs: int = 20, warmup: int = 3) -> ProfileResult:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_feed = _build_dummy_feed(sess)

    # Input/output info
    input_info = [{"name": i.name, "shape": [str(d) for d in i.shape], "type": i.type} for i in sess.get_inputs()]
    output_info = [{"name": o.name, "shape": [str(d) for d in o.shape], "type": o.type} for o in sess.get_outputs()]

    # Warmup
    t0 = time.perf_counter()
    for _ in range(warmup):
        sess.run(None, input_feed)
    warmup_ms = (time.perf_counter() - t0) * 1000.0 / max(warmup, 1)

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        sess.run(None, input_feed)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies = np.array(latencies)

    # Layer profiling
    layers = _get_layer_profiles(path, input_feed)
    layers.sort(key=lambda x: x.duration_us, reverse=True)

    # Count params
    num_params = 0
    try:
        import onnx
        model = onnx.load(path, load_external_data=False)
        for init in model.graph.initializer:
            s = 1
            for d in init.dims: s *= d
            num_params += s
    except Exception:
        pass

    # Extended analysis
    (op_summary, flops_per_layer, depth,
     weight_mb, peak_act_mb, total_flops, total_macs,
     diagnosis, suggestions, q_ratio, non_q_ops, est_speedup,
    ) = _analyze_graph(path, layers)

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
        top_bottlenecks=layers[:5],
        num_parameters=num_params or None,
        estimated_flops=total_flops or None,
        peak_memory_mb=round(weight_mb + peak_act_mb, 2) or None,
        op_type_summary=op_summary,
        graph_depth=depth,
        weight_memory_mb=round(weight_mb, 2),
        peak_activation_mb=round(peak_act_mb, 2),
        total_memory_mb=round(weight_mb + peak_act_mb, 2),
        total_macs=total_macs,
        flops_per_layer=flops_per_layer[:30],
        bottleneck_diagnosis=diagnosis[:20],
        optimization_suggestions=suggestions,
        quantizable_ratio=round(q_ratio, 3),
        non_quantizable_ops=non_q_ops,
        estimated_int8_speedup=round(est_speedup, 1),
        input_info=input_info,
        output_info=output_info,
    )


def profile_to_dict(result: ProfileResult) -> dict:
    d = {
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
            for l in result.layer_profiles[:50]
        ],
        # Extended v1.5
        "op_type_summary": result.op_type_summary,
        "graph_depth": result.graph_depth,
        "weight_memory_mb": result.weight_memory_mb,
        "peak_activation_mb": result.peak_activation_mb,
        "total_memory_mb": result.total_memory_mb,
        "total_macs": result.total_macs,
        "flops_per_layer": result.flops_per_layer,
        "bottleneck_diagnosis": result.bottleneck_diagnosis,
        "optimization_suggestions": result.optimization_suggestions,
        "quantizable_ratio": result.quantizable_ratio,
        "non_quantizable_ops": result.non_quantizable_ops,
        "estimated_int8_speedup": result.estimated_int8_speedup,
        "input_info": result.input_info,
        "output_info": result.output_info,
    }
    return d
