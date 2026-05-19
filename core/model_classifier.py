"""Heuristic ONNX → ssook task_type classifier.

ssook's "No-Code First" philosophy means the user should drop a model
and have the right tab open with the right defaults — without manually
picking `yolo` / `classification` / `segmentation` / `embedding` / `vlm`
from a dropdown.

The classifier inspects input/output shapes (without running inference)
and returns a (task_type, confidence, reason) triple. Heuristics here
are intentionally conservative: when in doubt we say `unknown` rather
than guess.

Public API:
    classify(path) -> dict[task_type, confidence, reason, suggested_tab,
                          input_shape, output_shapes, input_names, output_names]
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("ssook.classifier")


_TAB_BY_TASK = {
    "detection": "viewer",
    "classification": "viewer",
    "segmentation": "viewer",
    "instance_segmentation": "viewer",
    "pose": "viewer",
    "embedding": "embedding-viewer",
    "vlm_image_encoder": "vlm",
    "vlm_text_encoder": "vlm",
    "clip_pair": "clip",
}


def classify(path: str) -> dict:
    """Inspect an ONNX file's I/O signature and guess the task."""
    if not path or not os.path.isfile(path):
        return _result("unknown", 0.0, "File not found")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as e:
        return _result("unknown", 0.0, f"Cannot open: {e}")

    inputs = [(i.name, _shape_list(i.shape)) for i in sess.get_inputs()]
    outputs = [(o.name, _shape_list(o.shape)) for o in sess.get_outputs()]

    info = _classify_from_io(inputs, outputs, os.path.basename(path).lower())
    info["input_names"] = [n for n, _ in inputs]
    info["output_names"] = [n for n, _ in outputs]
    info["input_shape"] = inputs[0][1] if inputs else []
    info["output_shapes"] = [s for _, s in outputs]
    info["suggested_tab"] = _TAB_BY_TASK.get(info["task_type"], "viewer")
    return info


def _classify_from_io(inputs: list, outputs: list, name_hint: str) -> dict:
    """Pure heuristic core, no ORT session needed — easy to unit-test."""
    if not inputs or not outputs:
        return _result("unknown", 0.0, "No I/O tensors")

    # Text encoder: int64 input named input_ids → embedding output.
    # We approximate by input shape containing dynamic 'token' axis.
    in_name_str = " ".join(n for n, _ in inputs).lower()
    if "input_ids" in in_name_str or "tokens" in in_name_str:
        return _result("vlm_text_encoder", 0.85,
                       "Token-id input — likely a CLIP/BERT text encoder")

    first_in_shape = inputs[0][1]
    first_out_shape = outputs[0][1]

    # Image input (N,C,H,W) — anything below is image-based.
    if not _looks_image_input(first_in_shape):
        return _result("unknown", 0.0,
                       f"Input shape {first_in_shape} doesn't look like an image tensor")

    nlabels_hint = _name_hint_task(name_hint)

    # Single 2D output (N, K) — classification.
    if len(outputs) == 1 and len(first_out_shape) == 2 and _resolve_int(first_out_shape[1]):
        k = _resolve_int(first_out_shape[1])
        # CLIP image encoder typically 512 / 768 features — call it embedding.
        if k in {256, 384, 512, 768, 1024, 1280, 1536, 2048}:
            return _result("vlm_image_encoder" if "clip" in name_hint else "embedding",
                           0.75, f"2D output dim={k} looks like an embedding")
        return _result("classification", 0.8,
                       f"Single 2D output (logits, K={k})")

    # Single 4D output (N, C, H, W) — segmentation.
    if len(outputs) == 1 and len(first_out_shape) == 4:
        return _result("segmentation", 0.8,
                       f"Single 4D output {first_out_shape} = mask-per-class")

    # Two outputs with one 4D mask-like — instance segmentation (YOLO-seg).
    has_4d = any(len(s) == 4 for _, s in outputs)
    if has_4d and len(outputs) >= 2:
        return _result("instance_segmentation", 0.7,
                       "4D mask output + box outputs — instance segmentation")

    # Common YOLO detection: single output (1, A, N) where A∈{84, 85, 116} etc.
    if len(outputs) == 1 and len(first_out_shape) == 3:
        a, b, c = first_out_shape
        a_i, b_i, c_i = _resolve_int(a), _resolve_int(b), _resolve_int(c)
        for cand in (b_i, c_i):
            if cand and 4 < cand < 1000:
                # Pose heads carry 56 attrs (4 box + 1 cls + 51 kpts) on COCO.
                if cand == 56 or "pose" in name_hint:
                    return _result("pose", 0.75,
                                   f"Output attrs={cand} matches pose (4+1+17*3)")
                return _result("detection", 0.8,
                               f"3D output {first_out_shape} matches YOLO head")
        return _result("detection", 0.5,
                       f"3D output {first_out_shape} — likely detection")

    # Multi-output with a (N, K) head — best-guess detection.
    if len(outputs) >= 2:
        return _result("detection", 0.4,
                       f"Multi-output ({len(outputs)}) — assuming detection")

    if nlabels_hint:
        return _result(nlabels_hint, 0.3, "Filename hint only")

    return _result("unknown", 0.0,
                   f"Outputs={[s for _, s in outputs]} — no rule matched")


def _looks_image_input(shape: list) -> bool:
    if len(shape) != 4:
        return False
    # Either NCHW with C∈{1,3,4} or NHWC with C in last dim.
    c1 = _resolve_int(shape[1])
    c3 = _resolve_int(shape[3])
    if c1 in {1, 3, 4}:
        return True
    if c3 in {1, 3, 4}:
        return True
    return False


def _shape_list(shape) -> list:
    """Normalise ORT shape (mix of int / str / None) to a JSON-safe list."""
    out = []
    for s in shape or []:
        if isinstance(s, int) and s > 0:
            out.append(s)
        elif s is None:
            out.append("?")
        else:
            out.append(str(s))
    return out


def _resolve_int(v) -> Optional[int]:
    if isinstance(v, int) and v > 0:
        return v
    return None


def _name_hint_task(name: str) -> Optional[str]:
    n = name.lower()
    if "yolo" in n and "seg" in n: return "instance_segmentation"
    if "yolo" in n and "pose" in n: return "pose"
    if "yolo" in n: return "detection"
    if "rtmdet" in n or "detr" in n: return "detection"
    if "deeplab" in n or "unet" in n: return "segmentation"
    if "resnet" in n or "efficientnet" in n or "mobilenet" in n or "vit" in n: return "classification"
    if "clip" in n and "text" in n: return "vlm_text_encoder"
    if "clip" in n and ("image" in n or "vision" in n): return "vlm_image_encoder"
    if "clip" in n: return "vlm_image_encoder"
    return None


def _result(task_type: str, confidence: float, reason: str) -> dict:
    return {"task_type": task_type, "confidence": round(confidence, 2), "reason": reason}
