"""ONNX Model Inspector — graph info, I/O shapes, metadata, EP compatibility."""
import os
from dataclasses import dataclass
from typing import Optional

import onnxruntime as ort


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


def _count_parameters(path: str) -> Optional[int]:
    """Count total parameters from ONNX initializers."""
    try:
        import onnx
        model = onnx.load(path, load_external_data=False)
        total = 0
        for init in model.graph.initializer:
            size = 1
            for d in init.dims:
                size *= d
            total += size
        return total
    except Exception:
        return None


def _count_nodes(path: str) -> tuple[int, dict]:
    """Count nodes and op types from ONNX graph."""
    try:
        import onnx
        model = onnx.load(path, load_external_data=False)
        op_counts = {}
        for node in model.graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        return len(model.graph.node), op_counts
    except Exception:
        return 0, {}


def _get_opset_ir(path: str) -> tuple[int, int, str, str, str]:
    """Get opset version, IR version, producer, domain, description."""
    try:
        import onnx
        model = onnx.load(path, load_external_data=False)
        opset = model.opset_import[0].version if model.opset_import else 0
        return opset, model.ir_version, model.producer_name or "", model.domain or "", model.doc_string or ""
    except Exception:
        return 0, 0, "", "", ""


def inspect_model(path: str, *, test_eps: bool = False, gpu_lock=None) -> ModelInspection:
    """Inspect an ONNX model file and return detailed information.

    `test_eps`: when True, actually instantiate a session per available EP to
    confirm it can load the model (memory-spiky and slow). Default False uses
    the heuristic op-support table only — accelerator EPs are reported as
    "compatible if the op set is covered" without a real session.

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

    # Opset, IR, producer
    opset, ir_ver, producer, domain, description = _get_opset_ir(path)

    # Node counts
    num_nodes, op_counts = _count_nodes(path)

    # Parameters
    num_params = _count_parameters(path)

    # EP compatibility — op-level support analysis (+ optional real load test).
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
            compatible_eps.append(ep)
            continue

        ep_supported = _get_ep_supported_ops(ep)
        if ep_supported is not None:
            supported = all_ops & ep_supported
            fallback = all_ops - ep_supported
            supported_count = sum(op_counts.get(op, 0) for op in supported)
            ep_compat[ep] = {
                "supported_ops": sorted(supported),
                "fallback_ops": sorted(fallback),
                "supported_ratio": round(supported_count / max(total_ops, 1), 3),
                "known": True,
            }
        else:
            # Unknown EP: we have no op table, so we can't claim any ratio.
            # Report ratio=None + known=false so the UI renders "unknown"
            # instead of a misleading 100%.
            ep_compat[ep] = {
                "supported_ops": [],
                "fallback_ops": [],
                "supported_ratio": None,
                "known": False,
            }

        # Accelerator EPs: only do the expensive real-load test when requested.
        if test_eps:
            loaded = _ep_loads(path, ep, gpu_lock=gpu_lock)
            ep_compat[ep]["loads"] = loaded
            if loaded:
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


def _ep_loads(path: str, ep: str, *, gpu_lock=None) -> bool:
    """Return True if the model instantiates a session on `ep` (real test)."""
    def _try() -> bool:
        try:
            ort.InferenceSession(path, providers=[ep])
            return True
        except Exception:
            return False

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
