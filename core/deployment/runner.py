"""Pure orchestration for one export job — no FastAPI/HTTP here.

The router owns the TaskState and threads ``on_progress``/``is_stopped`` in,
exactly like ``core.benchmark_runner`` and the optimization pipeline.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional

from core.deployment.registry import registry
from core.deployment.manifest import build_manifest, write_manifest

log = logging.getLogger("ssook.deployment")


def validate_artifact(artifact_path: str, target) -> dict:
    """Light load-verification so a broken artifact is never reported as success.

    ONNX is opened under the CPU provider (always present). Engines/IR/CoreML
    only get an existence check — loading them needs their runtime/hardware.
    """
    if not artifact_path or not os.path.exists(artifact_path):
        return {"ok": False, "error": "artifact not found after export"}
    ext = os.path.splitext(artifact_path)[1].lower()
    if ext == ".onnx":
        try:
            import onnxruntime as ort
            ort.InferenceSession(artifact_path, providers=["CPUExecutionProvider"])
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": f"ONNX failed to load: {e}"}
    return {"ok": True}


def run_export(
    config: dict,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    is_stopped: Optional[Callable[[], bool]] = None,
) -> dict:
    """Run a single export. ``config`` keys:

    model_path, target, output_path, opts{opset,batch,input_size,dynamic,
    precision,calibration_dir,device,workspace_gb,simplify,class_names},
    validate (default True), write_manifest (default True), model_meta.
    """
    target_name = config.get("target", "")
    target = registry.get(target_name)
    if target is None:
        return {"error": f"Unknown export target '{target_name}'", "target": target_name}

    available, reason = target.is_available()
    if not available:
        return {"error": f"Target '{target_name}' unavailable: {reason}", "target": target_name}

    model_path = config["model_path"]
    output_path = config["output_path"]
    opts = dict(config.get("opts", {}))

    log.info("export start: target=%s source=%s -> %s precision=%s",
             target_name, os.path.basename(model_path), output_path,
             opts.get("precision", "fp32"))

    result = target.export(model_path, output_path, opts,
                           on_progress=on_progress, is_stopped=is_stopped)
    if "error" in result:
        log.warning("export failed: target=%s err=%s", target_name, result["error"])
        return result

    # Build is single-shot; if cancellation was requested while it ran, skip the
    # post-export validation/manifest steps.
    if is_stopped and is_stopped():
        result["cancelled"] = True
        return result

    final_path = result.get("output_path", output_path)
    result["original_size_mb"] = _size_mb(model_path)

    if config.get("validate", True):
        if on_progress:
            on_progress(1, 1, "Validating artifact...")
        verdict = validate_artifact(final_path, target)
        result["validated"] = verdict.get("ok")
        if verdict.get("error"):
            result["validation_error"] = verdict["error"]

    if config.get("write_manifest", True):
        try:
            manifest = build_manifest(model_path, target, opts, result, config.get("model_meta"))
            result["manifest_path"] = write_manifest(manifest, final_path)
        except Exception as e:  # manifest is a convenience, not the deliverable
            log.debug("manifest write failed: %s", e)
            result["manifest_error"] = str(e)

    log.info("export done: target=%s -> %s (%s MB)", target_name, final_path,
             result.get("output_size_mb"))
    return result


def _size_mb(path: str) -> float:
    try:
        if os.path.isdir(path):
            from core.deployment.exporters import dir_size_mb
            return dir_size_mb(path)
        return round(os.path.getsize(path) / 1024 / 1024, 2)
    except OSError:
        return 0.0
