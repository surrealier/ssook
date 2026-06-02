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


def onnx_actual_opset(artifact_path: str) -> Optional[int]:
    """Read the produced ONNX's default-domain opset, or None if unreadable.

    The manifest must record the opset the artifact ACTUALLY carries (opset
    re-spec on an existing graph can silently no-op), not the requested value.
    """
    try:
        import onnx
        model = onnx.load(artifact_path)
        for imp in model.opset_import:
            # The default ("") domain is the ai.onnx version most tools mean by "opset".
            if imp.domain in ("", "ai.onnx"):
                return int(imp.version)
        return int(model.opset_import[0].version) if model.opset_import else None
    except Exception:
        return None


def validate_artifact(artifact_path: str, target) -> dict:
    """Light load-verification so a broken artifact is never reported as success.

    ONNX is opened under the TARGET's execution provider when that provider is
    present in this process (with CPU as fallback), so a CUDA/TensorRT export
    is actually exercised on its provider. If the target provider isn't
    available in-process, or the provider-specific session fails to build, we
    fall back to a CPU-only session and surface a ``warning`` — a CPU-validated
    artifact is still a valid artifact. Engines/IR/CoreML only get an existence
    check — loading them needs their runtime/hardware.
    """
    if not artifact_path or not os.path.exists(artifact_path):
        return {"ok": False, "error": "artifact not found after export"}
    ext = os.path.splitext(artifact_path)[1].lower()
    if ext == ".onnx":
        try:
            import onnxruntime as ort
        except Exception as e:
            return {"ok": False, "error": f"onnxruntime unavailable: {e}"}

        provider = getattr(target, "provider", None)
        available = set(ort.get_available_providers())
        # Skip CPU here — it's the universal fallback handled below; trying it as
        # the "target provider" would only duplicate the CPU session.
        if provider and provider != "CPUExecutionProvider" and provider in available:
            try:
                ort.InferenceSession(artifact_path,
                                     providers=[provider, "CPUExecutionProvider"])
                return {"ok": True}
            except Exception as e:
                log.debug("provider %s session failed, falling back to CPU: %s", provider, e)
                # Fall through to CPU-only below with a warning.
        try:
            ort.InferenceSession(artifact_path, providers=["CPUExecutionProvider"])
        except Exception as e:
            return {"ok": False, "error": f"ONNX failed to load: {e}"}
        if provider and provider != "CPUExecutionProvider":
            return {"ok": True,
                    "warning": f"validated on CPU; target provider {provider} "
                               f"unavailable in this process"}
        return {"ok": True}
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
        if verdict.get("warning"):
            result["validation_warning"] = verdict["warning"]

    if config.get("write_manifest", True):
        try:
            # Record the opset the artifact ACTUALLY carries (best-effort read of
            # the produced .onnx); falls back to the requested opset otherwise.
            actual_opset = None
            if os.path.splitext(final_path)[1].lower() == ".onnx":
                actual_opset = onnx_actual_opset(final_path)
            manifest = build_manifest(model_path, target, opts, result,
                                      config.get("model_meta"), actual_opset=actual_opset)
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
