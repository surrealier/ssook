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


def inspect_model(path: str) -> ModelInspection:
    """Inspect an ONNX model file and return detailed information."""
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

    # EP compatibility
    available_eps = ort.get_available_providers()
    compatible_eps = []
    for ep in available_eps:
        try:
            test_sess = ort.InferenceSession(path, providers=[ep])
            compatible_eps.append(ep)
            del test_sess
        except Exception:
            pass

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
    )


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
    }
