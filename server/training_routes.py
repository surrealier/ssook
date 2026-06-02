"""/api/training/* — model training API routes.

Long-running-job pattern copied from server/optimization_routes.py: a
RLock-guarded TaskState registered in all_states, an atomic try_start() guard,
work dispatched to the shared executor, cooperative stop by polling the
running flag, and a status endpoint returning a lock-isolated snapshot.

Heavy training deps (torch/ultralytics/timm) are imported lazily inside the
worker (via core.training), never at module import, so this router loads in the
ONNX-runtime-only base environment too.
"""
import logging
import os
from typing import Optional, Union

from fastapi import APIRouter
from pydantic import BaseModel

from server.path_safety import (
    safe_image_dir, safe_label_dir, safe_weights_file, UnsafePathError,
)
from server.state import TaskState, all_states, executor, task_locks

log = logging.getLogger("ssook.training")
router = APIRouter()

# results is a dict (TaskState defaults it to a list) — set explicitly.
training_state = TaskState()
training_state["results"] = {}
all_states["training"] = training_state


class TrainRequest(BaseModel):
    trainer: str                      # registry id, e.g. "yolo_detect"
    model: str = ""                   # base checkpoint/backbone (name or path)
    images_dir: str = ""
    labels_dir: str = ""
    names: Union[str, list, dict, None] = None
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    lr0: Optional[float] = None
    device: str = ""
    val_ratio: float = 0.2
    seed: int = 0
    patience: Optional[int] = None
    project: str = ""
    name: str = ""
    keep: Optional[dict] = None       # fork-only keep/multilabel passthrough


def _is_path_like(model: str) -> bool:
    """A model field is a path (validate it) vs a bare name ultralytics fetches."""
    return bool(model) and (os.sep in model or "/" in model or os.path.exists(model))


@router.get("/api/training/trainers")
async def list_trainers():
    """List registered trainers + their availability, plus a GPU hint."""
    from core.training import registry
    try:
        from server.utils import check_gpu_available
        gpu = bool(check_gpu_available())
    except Exception:
        gpu = False
    return {"trainers": [t.to_dict() for t in registry.list_all()], "cuda_available": gpu}


@router.post("/api/training/run")
async def run_training_route(req: TrainRequest):
    # Validate paths at the boundary; return {error} with 200 like other jobs.
    try:
        images_dir = safe_image_dir(req.images_dir) if req.images_dir else ""
        labels_dir = safe_label_dir(req.labels_dir) if req.labels_dir else ""
        model = safe_weights_file(req.model) if _is_path_like(req.model) else req.model
    except UnsafePathError as e:
        return {"error": str(e), "code": f"PATH_{e.code}"}

    if not images_dir:
        return {"error": "images_dir is required"}

    if not training_state.try_start(progress=0, total=int(req.epochs), msg="Starting training...",
                                    results={"epochs": [], "summary": {}}, cancel_requested=False):
        return {"error": "Training already running"}

    config = {
        "trainer": req.trainer, "model": model,
        "images_dir": images_dir, "labels_dir": labels_dir, "names": req.names,
        "epochs": req.epochs, "imgsz": req.imgsz, "batch": req.batch,
        "lr0": req.lr0, "device": req.device, "val_ratio": req.val_ratio,
        "seed": req.seed, "patience": req.patience,
        "project": req.project, "name": req.name, "keep": req.keep,
    }

    def _run():
        # Serialize on the GPU lock so a long training run doesn't contend with
        # live inference/benchmark on the same device.
        with task_locks["gpu_infer"]:
            # Accumulate epochs in a worker-local list and publish the whole
            # results object via the locked update(). snapshot() is a SHALLOW
            # copy, so mutating a nested list in place would race the status
            # poller's serialization — we hand it a fresh list each time instead.
            epoch_rows: list[dict] = []
            try:
                from core.training.runner import run_training

                def _on_progress(done, total, msg):
                    training_state.update(progress=done, total=total, msg=msg)

                def _on_metric(epoch, metrics):
                    epoch_rows.append({"epoch": epoch, **metrics})
                    training_state.update(results={"epochs": list(epoch_rows), "summary": {}})

                # Cooperative stop via the per-tab stop (cancel_requested) OR the
                # global force-stop / shutdown (running=False).
                def _is_stopped():
                    return training_state.get("cancel_requested", False) or \
                        not training_state.get("running", False)

                result = run_training(config, on_progress=_on_progress,
                                      on_metric=_on_metric, is_stopped=_is_stopped)
                if "error" in result:
                    training_state.update(msg=f"Error: {result['error']}")
                    return
                stopped = training_state.get("cancel_requested", False)
                training_state.update(
                    msg="Stopped" if stopped else "Complete",
                    results={"epochs": list(epoch_rows), "summary": dict(result)})
            except Exception as e:
                log.exception("training failed")
                training_state.update(msg=f"Error: {e}")
            finally:
                # Worker owns `running` — cleared only here so the start guard
                # stays closed for the whole run (no concurrent re-entry).
                training_state.update(running=False, cancel_requested=False)

    executor.submit(_run)
    return {"ok": True}


@router.get("/api/training/status")
async def training_status():
    return training_state.snapshot()


@router.get("/api/training/stop")
async def training_stop():
    # Cooperative: request cancellation WITHOUT clearing `running` (the worker
    # owns it). The trainer polls this at the next epoch boundary and exits;
    # keeping `running` set means the start guard can't be re-entered mid-run.
    training_state["cancel_requested"] = True
    return {"ok": True}


@router.get("/api/training/runs")
async def list_runs_route():
    """List finished training runs (newest first) for the runs browser."""
    from core.training.runs import list_runs  # lazy: keep router import-light
    return {"runs": list_runs()}


@router.delete("/api/training/runs/{name}")
async def delete_run_route(name: str):
    """Delete one run dir; traversal/missing names return {error} with 200."""
    from core.training.runs import delete_run
    try:
        delete_run(name)
    except ValueError as e:
        return {"error": str(e)}
    return {"ok": True}
