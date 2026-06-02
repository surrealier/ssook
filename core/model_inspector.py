"""ONNX Model Inspector — graph info, I/O shapes, metadata, EP compatibility."""
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import onnxruntime as ort

log = logging.getLogger("ssook.model_inspector")


@dataclass
class TensorInfo:
    name: str
    shape: list
    dtype: str


@dataclass
class ModelInspection:
    file_name: str
    file_size_mb: float
    opset_version: int
    ir_version: int
    producer: str
    domain: str
    description: str
    inputs: list[TensorInfo]
    outputs: list[TensorInfo]
    metadata: dict
    num_nodes: int
    node_op_counts: dict       # {op_type: count}
    available_eps: list[str]
    compatible_eps: list[str]  # EPs that can actually run this model
    num_parameters: Optional[int]
    ep_compatibility: dict = None  # {ep: {supported_ops, fallback_ops, supported_ratio}}


def _load_proto(path: str):
    """Load the ONNX ModelProto once (no external data), or None on failure.

    The graph is parsed a single time per inspect and reused by the count/opset
    helpers below — a multi-GB model was previously parsed 3x (once per helper)
    on top of the ORT session. Failure degrades gracefully: helpers fall back to
    their neutral defaults so a corrupt graph still yields I/O info from ORT.
    """
    try:
        import onnx
        return onnx.load(path, load_external_data=False)
    except Exception as exc:
        log.debug("onnx.load failed for %s: %s", path, exc)
        return None


def _count_parameters(model) -> Optional[int]:
    """Count total parameters from a loaded ONNX model's initializers."""
    if model is None:
        return None
    total = 0
    for init in model.graph.initializer:
        size = 1
        for d in init.dims:
            size *= d
        total += size
    return total


def _count_nodes(model) -> tuple[int, dict]:
    """Count nodes and op types from a loaded ONNX model's graph."""
    if model is None:
        return 0, {}
    op_counts: dict[str, int] = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    return len(model.graph.node), op_counts


def _get_opset_ir(model) -> tuple[int, int, str, str, str]:
    """Get opset version, IR version, producer, domain, description.

    Opset is read from the default ai.onnx domain (empty or 'ai.onnx' domain
    string), not opset_import[0]: a model can carry a com.microsoft contrib
    opset first, and users care about the standard ONNX opset number.
    """
    if model is None:
        return 0, 0, "", "", ""
    opset = _default_opset(model.opset_import)
    return (opset, model.ir_version, model.producer_name or "",
            model.domain or "", model.doc_string or "")


def _default_opset(opset_import) -> int:
    """Return the standard ai.onnx opset version, ignoring custom domains."""
    default_versions = [op.version for op in opset_import
                        if op.domain in ("", "ai.onnx")]
    if default_versions:
        return max(default_versions)
    # No default-domain opset (unusual) — fall back to whatever is first.
    return opset_import[0].version if opset_import else 0


def inspect_model(path: str, *, test_eps: bool = True, gpu_lock=None) -> ModelInspection:
    """Inspect an ONNX model file and return detailed information.

    `test_eps`: when True (default), actually instantiate a session per available
    EP to confirm ORT can load the model — the only ground-truth EP-compatibility
    signal ORT exposes. Each provider's result is reported as `loads` (bool) plus
    `load_error` (reason string or None) so the UI can distinguish "EP missing /
    driver mismatch" from "model incompatible". When False, only the heuristic
    op-support table is computed (no session probe) — used by callers that want a
    fast estimate without the memory-spiky GPU load test.

    `gpu_lock`: optional context-manager lock the caller passes to serialise
    GPU-backed EP load tests (e.g. server's task_locks['gpu_infer']). Kept as a
    parameter so core/ stays free of server imports.
    """
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    file_name = os.path.basename(path)

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    # Inputs
    inputs = []
    for inp in session.get_inputs():
        inputs.append(TensorInfo(
            name=inp.name,
            shape=[str(d) for d in inp.shape],
            dtype=inp.type or "unknown",
        ))

    # Outputs
    outputs = []
    for out in session.get_outputs():
        outputs.append(TensorInfo(
            name=out.name,
            shape=[str(d) for d in out.shape],
            dtype=out.type or "unknown",
        ))

    # Metadata
    meta = session.get_modelmeta()
    metadata = dict(meta.custom_metadata_map) if meta.custom_metadata_map else {}

    # Parse the ONNX graph once and reuse it across all three helpers (was 3x).
    model_proto = _load_proto(path)
    opset, ir_ver, producer, domain, description = _get_opset_ir(model_proto)
    num_nodes, op_counts = _count_nodes(model_proto)
    num_params = _count_parameters(model_proto)

    # EP compatibility — heuristic op-level estimate plus an optional REAL probe.
    available_eps = ort.get_available_providers()
    compatible_eps = []
    ep_compat = {}
    all_ops = set(op_counts.keys())
    total_ops = num_nodes

    for ep in available_eps:
        # CPU is the universal fallback — it runs every op. Report it as fully
        # supported rather than "unknown" (it has no curated op table).
        if ep == "CPUExecutionProvider":
            ep_compat[ep] = {
                "supported_ops": sorted(all_ops),
                "fallback_ops": [],
                "supported_ratio": 1.0,
                "known": True,
            }
        else:
            ep_supported = _get_ep_supported_ops(ep)
            if ep_supported is not None:
                supported = all_ops & ep_supported
                fallback = all_ops - ep_supported
                supported_count = sum(op_counts.get(op, 0) for op in supported)
                ep_compat[ep] = {
                    "supported_ops": sorted(supported),
                    "fallback_ops": sorted(fallback),
                    # The op table is an estimate, not a real load result — the
                    # UI labels it as such (see inspector.ep_estimate).
                    "supported_ratio": round(supported_count / max(total_ops, 1), 3),
                    "known": True,
                }
            else:
                # Unknown EP: no op table, so we can't claim any ratio. Report
                # ratio=None + known=false so the UI renders "unknown" instead
                # of a misleading 100%.
                ep_compat[ep] = {
                    "supported_ops": [],
                    "fallback_ops": [],
                    "supported_ratio": None,
                    "known": False,
                }

        # Real probe: actually try to instantiate a session on this EP. This is
        # the ground truth — the heuristic above can be wrong in both directions.
        # `compatible_eps` reflects what ORT verified it can load, not the table.
        if test_eps:
            loaded, load_error, assigned = _ep_loads(path, ep, gpu_lock=gpu_lock)
            ep_compat[ep]["loads"] = loaded
            ep_compat[ep]["load_error"] = load_error
            # The EP ORT actually fell back to (e.g. asking for CUDA may yield
            # ['CUDAExecutionProvider', 'CPUExecutionProvider']) — surface it so
            # a "loads" that silently ran on CPU is visible.
            ep_compat[ep]["assigned_providers"] = assigned
            if loaded:
                compatible_eps.append(ep)
        elif ep == "CPUExecutionProvider":
            # No real probe requested — fall back to the heuristic: CPU always
            # runs, so report it compatible to preserve prior behaviour.
            compatible_eps.append(ep)

    return ModelInspection(
        file_name=file_name,
        file_size_mb=round(file_size_mb, 2),
        opset_version=opset,
        ir_version=ir_ver,
        producer=producer,
        domain=domain,
        description=description,
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
        num_nodes=num_nodes,
        node_op_counts=op_counts,
        available_eps=available_eps,
        compatible_eps=compatible_eps,
        num_parameters=num_params,
        ep_compatibility=ep_compat,
    )


# Known EP op support (heuristic — not exhaustive)
_EP_OPS = {
    "CUDAExecutionProvider": {
        "Conv", "ConvTranspose", "MatMul", "Gemm", "Relu", "Sigmoid", "Tanh",
        "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Exp", "Log",
        "BatchNormalization", "InstanceNormalization", "LayerNormalization",
        "MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool",
        "Softmax", "LogSoftmax", "Concat", "Reshape", "Transpose", "Flatten",
        "Squeeze", "Unsqueeze", "Gather", "Slice", "Split", "Pad", "Resize",
        "Upsample", "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin",
        "Clip", "Cast", "Where", "Tile", "Expand", "Shape", "NonZero",
        "TopK", "ScatterND", "GatherND", "Attention", "SkipLayerNormalization",
    },
    "TensorrtExecutionProvider": {
        "Conv", "ConvTranspose", "MatMul", "Gemm", "Relu", "Sigmoid", "Tanh",
        "Add", "Sub", "Mul", "Div", "BatchNormalization", "MaxPool", "AveragePool",
        "GlobalAveragePool", "Softmax", "Concat", "Reshape", "Transpose", "Flatten",
        "Squeeze", "Unsqueeze", "Gather", "Slice", "Pad", "Resize", "Clip", "Cast",
        "ReduceMean", "ReduceSum", "TopK",
    },
    "OpenVINOExecutionProvider": {
        "Conv", "ConvTranspose", "MatMul", "Gemm", "Relu", "Sigmoid", "Tanh",
        "Add", "Sub", "Mul", "Div", "BatchNormalization", "MaxPool", "AveragePool",
        "GlobalAveragePool", "Softmax", "Concat", "Reshape", "Transpose", "Flatten",
        "Squeeze", "Unsqueeze", "Gather", "Slice", "Pad", "Resize", "Clip", "Cast",
        "ReduceMean", "Interpolate", "Split",
    },
    "DmlExecutionProvider": {
        "Conv", "ConvTranspose", "MatMul", "Gemm", "Relu", "Sigmoid", "Tanh",
        "Add", "Sub", "Mul", "Div", "BatchNormalization", "MaxPool", "AveragePool",
        "GlobalAveragePool", "Softmax", "Concat", "Reshape", "Transpose", "Flatten",
        "Squeeze", "Unsqueeze", "Gather", "Slice", "Pad", "Resize", "Clip", "Cast",
    },
}


def _get_ep_supported_ops(ep_name: str):
    """Return set of supported ops for an EP, or None if unknown."""
    return _EP_OPS.get(ep_name)


# GPU-backed EPs whose real-load test contends for device memory — serialise
# those behind the shared gpu_infer lock so we don't spike VRAM concurrently.
_GPU_EPS = {"CUDAExecutionProvider", "TensorrtExecutionProvider",
            "ROCMExecutionProvider", "MIGraphXExecutionProvider", "DmlExecutionProvider"}


def _ep_loads(path: str, ep: str, *, gpu_lock=None) -> tuple[bool, Optional[str], list[str]]:
    """Probe whether ORT can instantiate a session on `ep`.

    Returns (loaded, load_error, assigned_providers):
      - loaded: True iff ORT actually assigned the requested `ep`. This is NOT
        the same as "a session was created": ORT silently falls back to CPU when
        an EP is unavailable (e.g. TensorRT DLL missing) instead of raising, so a
        created session running on CPU must be reported as loaded=False for `ep`.
      - load_error: the reason string ("<ExcType>: <message>") on a raise, or a
        "fell back to [...]" note when ORT dropped the EP to CPU, else None.
        Previously this was swallowed to a bare False, so the user could not tell
        "EP not installed / driver mismatch" from "model incompatible".
      - assigned_providers: what ORT actually selected; empty list on a raise.

    Never raises — a provider that errors is reported as unavailable-with-reason
    so one bad EP cannot crash the whole inspect.
    """
    def _try() -> tuple[bool, Optional[str], list[str]]:
        try:
            session = ort.InferenceSession(path, providers=[ep])
            assigned = list(session.get_providers())
            if ep in assigned:
                return True, None, assigned
            # Session built but ORT dropped the requested EP (silent CPU
            # fallback) — report it as unavailable WITH the reason.
            reason = f"EP unavailable: ORT fell back to {assigned}"
            log.debug("EP %s not assigned for %s; %s", ep, path, reason)
            return False, reason, assigned
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            log.debug("EP %s failed to load %s: %s", ep, path, reason)
            return False, reason, []

    # GPU-backed EP load tests contend for device memory; the caller may pass a
    # lock to serialise them. Without one we still run, just unguarded.
    if ep in _GPU_EPS and gpu_lock is not None:
        with gpu_lock:
            return _try()
    return _try()


def inspection_to_dict(info: ModelInspection) -> dict:
    """Convert ModelInspection to JSON-serializable dict."""
    return {
        "file_name": info.file_name,
        "file_size_mb": info.file_size_mb,
        "opset_version": info.opset_version,
        "ir_version": info.ir_version,
        "producer": info.producer,
        "domain": info.domain,
        "description": info.description,
        "inputs": [{"name": t.name, "shape": t.shape, "dtype": t.dtype} for t in info.inputs],
        "outputs": [{"name": t.name, "shape": t.shape, "dtype": t.dtype} for t in info.outputs],
        "metadata": info.metadata,
        "num_nodes": info.num_nodes,
        "node_op_counts": info.node_op_counts,
        "available_eps": info.available_eps,
        "compatible_eps": info.compatible_eps,
        "num_parameters": info.num_parameters,
        "ep_compatibility": info.ep_compatibility or {},
    }
