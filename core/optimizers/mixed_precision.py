"""Mixed Precision Quantization — exclude sensitive layers from INT8."""
import os
import tempfile
import numpy as np
from core.optimizer_registry import BaseOptimizer


def compute_sensitivity_scores(model_path: str) -> dict[str, float]:
    """Compute per-node quantization sensitivity based on weight distribution.

    Keys on node.name when present, else node.output[0] — many real ONNX
    graphs leave node.name empty, and an empty key never matches ORT's
    `nodes_to_exclude`, silently turning mixed precision into full INT8.
    """
    import onnx
    from onnx import numpy_helper
    model = onnx.load(model_path, load_external_data=False)
    init_map = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    scores: dict[str, float] = {}
    for node in model.graph.node:
        if node.op_type not in ("Conv", "MatMul", "Gemm"):
            continue
        key = node.name or (node.output[0] if node.output else "")
        if not key:
            continue
        for inp in node.input:
            if inp in init_map:
                w = init_map[inp].flatten().astype(np.float64)
                if w.size == 0:
                    continue
                rng = float(np.max(np.abs(w)))
                std = float(np.std(w))
                outlier_ratio = float(np.mean(np.abs(w) > 3 * std)) if std > 0 else 0
                scores[key] = rng * (1 + outlier_ratio * 10)
                break
    return scores


def _ensure_named(model_path: str) -> str:
    """Return a model path whose quantizable nodes all have non-empty names.

    ORT's `nodes_to_exclude` matches by node.name, so unnamed nodes can never
    be excluded. If any quantizable node is unnamed we write a temp copy with
    deterministic names (`{op_type}_{i}`) keyed identically to the sensitivity
    scores. Returns the original path when no renaming is needed.
    """
    import onnx
    model = onnx.load(model_path, load_external_data=False)
    changed = False
    for i, node in enumerate(model.graph.node):
        if not node.name:
            node.name = f"{node.op_type}_{i}"
            changed = True
    if not changed:
        return model_path
    fd, tmp = tempfile.mkstemp(suffix=".onnx", prefix="ssook_named_")
    os.close(fd)
    onnx.save(model, tmp)
    return tmp


class MixedPrecisionOptimizer(BaseOptimizer):
    name = "mixed_precision"
    category = "quantization"
    description = "Mixed precision INT8 — exclude sensitive layers automatically"

    def can_apply(self, model_path):
        return model_path.endswith(".onnx")

    def apply(self, model_path, output_path, **kw):
        from onnxruntime.quantization import quantize_static as _qs, QuantType, QuantFormat
        from core.quantizer import _AutoCalibrationReader

        exclude_pct = kw.get("exclude_pct", 20)
        calibration_dir = kw.get("calibration_dir", "")
        max_images = kw.get("max_images", 100)

        # Name unnamed nodes first so scores and exclusions share the same keys.
        named_path = _ensure_named(model_path)
        tmp_named = named_path if named_path != model_path else None
        try:
            scores = compute_sensitivity_scores(named_path)
            if not scores:
                return {"error": "No quantizable layers found"}

            sorted_nodes = sorted(scores.items(), key=lambda x: -x[1])
            n_exclude = max(1, int(len(sorted_nodes) * exclude_pct / 100))
            # Keys are guaranteed non-empty (compute_sensitivity_scores drops blanks).
            excluded = [name for name, _ in sorted_nodes[:n_exclude]]

            if calibration_dir and os.path.isdir(calibration_dir):
                reader = _AutoCalibrationReader(named_path, calibration_dir, max_images)
                if not reader.images:
                    return {"error": "No calibration images found"}
                _qs(named_path, output_path,
                    calibration_data_reader=reader,
                    quant_format=QuantFormat.QDQ,
                    per_channel=True,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8,
                    nodes_to_exclude=excluded)
            else:
                # No calibration data — fall back to dynamic quantization, which
                # also honors nodes_to_exclude.
                from onnxruntime.quantization import quantize_dynamic as _qd
                _qd(named_path, output_path, weight_type=QuantType.QUInt8,
                    nodes_to_exclude=excluded)
        finally:
            if tmp_named and os.path.isfile(tmp_named):
                try:
                    os.remove(tmp_named)
                except OSError:
                    pass

        orig = os.path.getsize(model_path)
        new = os.path.getsize(output_path)
        return {
            "method": "mixed_precision",
            "excluded_nodes": excluded,
            "num_excluded": len(excluded),
            "exclude_pct": exclude_pct,
            "sensitivity_scores": {k: round(v, 4) for k, v in sorted_nodes[:10]},
            "original_size_mb": round(orig / 1024 / 1024, 2),
            "quantized_size_mb": round(new / 1024 / 1024, 2),
            "compression_ratio": round(orig / max(new, 1), 2),
            "output_path": output_path,
        }
