"""/api/deploy/* — model deployment / export API routes.

Exports a trained model (PyTorch .pt or existing .onnx) into a runtime
artifact for a chosen execution provider — ONNX (onnxruntime CPU/CUDA/DirectML),
TensorRT engine, OpenVINO IR, CoreML, or TorchScript — with configurable
opset / batch / input-size / precision, and writes a deploy manifest.

Long-running-job pattern mirrors server/optimization_routes.py.
"""
import contextlib
import logging
import os
from typing import Optional, Union

from fastapi import APIRouter
from pydantic import BaseModel

from server.path_safety import (
    safe_weights_file, safe_export_output, safe_image_dir, UnsafePathError,
)
from server.state import TaskState, all_states, executor, task_locks

log = logging.getLogger("ssook.deployment")
router = APIRouter()

deploy_state = TaskState()
deploy_state["results"] = {}
all_states["deploy"] = deploy_state


class DeployRequest(BaseModel):
    model_path: str
    target: str                                  # registry id, e.g. "tensorrt"
    output_path: str = ""
    opset: Optional[int] = None
    batch: Optional[int] = None
    input_size: Union[str, int, list, None] = None
    dynamic: bool = False
    precision: str = "fp32"                       # fp32 | fp16 | int8
    calibration_dir: str = ""
    device: str = ""
    workspace_gb: float = 4.0
    simplify: bool = False
    write_manifest: bool = True
    do_validate: bool = True           # load-verify the artifact after export


@router.get("/api/deploy/targets")
async def list_targets():
    """List export targets with availability + platform so the UI can grey out
    targets whose backend/runtime isn't installed on this host."""
    from core.deployment import registry
    return {"targets": [t.to_dict() for t in registry.list_all()]}


def _derive_output(model_path: str, target_name: str, ext: str) -> str:
    base, _ = os.path.splitext(model_path)
    return f"{base}_{target_name}{ext}"


def _source_kind(model_path: str) -> str:
    """Map a source extension to a target.accepts kind ("pt" | "onnx" | ext)."""
    ext = os.path.splitext(model_path)[1].lower()
    if ext in (".pt", ".pth"):
        return "pt"
    if ext == ".onnx":
        return "onnx"
    return ext.lstrip(".") or "?"


@router.post("/api/deploy/run")
async def run_deploy(req: DeployRequest):
    from core.deployment import registry

    target = registry.get(req.target)
    if target is None:
        return {"error": f"Unknown export target '{req.target}'"}

    try:
        model_path = safe_weights_file(req.model_path)
        out = req.output_path or _derive_output(model_path, req.target, target.output_ext)
        out = safe_export_output(out)
        calibration_dir = safe_image_dir(req.calibration_dir) if req.calibration_dir else ""
    except UnsafePathError as e:
        return {"error": str(e), "code": f"PATH_{e.code}"}

    # Reject incompatible source kinds early so the user gets a clear message
    # instead of a backend-specific failure mid-export (CoreML/TorchScript need .pt).
    source_kind = _source_kind(req.model_path)
    if source_kind not in target.accepts:
        accepts = "/".join(target.accepts)
        return {"error": f"Target '{req.target}' requires a {accepts} source "
                         f"(got {source_kind})"}

    # Validate the configurable levers up front (opset range follows the ONNX
    # opset versions onnx.version_converter / ultralytics realistically accept).
    if req.opset is not None and not (9 <= req.opset <= 22):
        return {"error": f"opset must be between 9 and 22 (got {req.opset})"}
    if req.batch is not None and req.batch < 1:
        return {"error": f"batch must be >= 1 (got {req.batch})"}
    if req.workspace_gb <= 0:
        return {"error": f"workspace_gb must be > 0 (got {req.workspace_gb})"}

    if not deploy_state.try_start(progress=0, total=1, msg="Starting export...",
                                  results={}, cancel_requested=False):
        return {"error": "Export already running"}

    config = {
        "model_path": model_path, "target": req.target, "output_path": out,
        "validate": req.do_validate, "write_manifest": req.write_manifest,
        "opts": {
            "opset": req.opset, "batch": req.batch, "input_size": req.input_size,
            "dynamic": req.dynamic, "precision": req.precision,
            "calibration_dir": calibration_dir, "device": req.device,
            "workspace_gb": req.workspace_gb, "simplify": req.simplify,
        },
    }

    def _run():
        # Only GPU-building targets (TensorRT, ONNX-CUDA) serialize with
        # inference via the gpu_infer lock. CPU-only exports (CPU ONNX,
        # OpenVINO, TorchScript, CoreML) must NOT hold it — otherwise a long
        # CPU export would block the viewer/benchmark from running inference.
        gpu_guard = task_locks["gpu_infer"] if target.uses_gpu else contextlib.nullcontext()
        with gpu_guard:
            try:
                from core.deployment.runner import run_export

                def _on_progress(done, total, msg):
                    deploy_state.update(progress=done, total=max(total, 1), msg=msg)

                # Cancellation via the per-tab stop (cancel_requested) OR the
                # global force-stop / shutdown (running=False). Export is
                # single-shot so this only short-circuits the post-export steps.
                def _is_stopped():
                    return deploy_state.get("cancel_requested", False) or \
                        not deploy_state.get("running", False)

                result = run_export(config, on_progress=_on_progress, is_stopped=_is_stopped)
                if "error" in result:
                    deploy_state.update(msg=f"Error: {result['error']}", results=result)
                elif result.get("cancelled"):
                    deploy_state.update(msg="Stopped", results=result)
                else:
                    deploy_state.update(progress=1, msg="Complete", results=result)
            except Exception as e:
                log.exception("export failed")
                deploy_state.update(msg=f"Error: {e}")
            finally:
                # The worker owns `running` — cleared only here, so try_start
                # stays closed for the whole job and a concurrent POST can't
                # start a second export that clobbers this one's results.
                deploy_state.update(running=False, cancel_requested=False)

    executor.submit(_run)
    return {"ok": True, "output_path": out}


@router.get("/api/deploy/status")
async def deploy_status():
    return deploy_state.snapshot()


@router.get("/api/deploy/stop")
async def deploy_stop():
    # Export is single-shot/non-interruptible, so request cancellation WITHOUT
    # clearing `running` (the worker owns it). This keeps the start guard closed
    # so a second POST can't race; cancellation only skips post-export steps.
    deploy_state["cancel_requested"] = True
    return {"ok": True}
