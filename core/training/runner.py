"""Pure orchestration for one training job — no FastAPI/HTTP here.

The router owns the TaskState and threads the callbacks in, exactly like
``core.benchmark_runner``.
"""
from __future__ import annotations

import logging
from typing import Optional

from core.training.registry import registry, ProgressFn, MetricFn, StopFn

log = logging.getLogger("ssook.training")


def run_training(
    config: dict,
    on_progress: Optional[ProgressFn] = None,
    on_metric: Optional[MetricFn] = None,
    is_stopped: Optional[StopFn] = None,
) -> dict:
    """Resolve a trainer, check availability, and run it.

    ``config`` keys: trainer (id), model, images_dir, labels_dir, names,
    epochs, imgsz, batch, lr0, device, val_ratio, seed, patience, project,
    name, keep.
    """
    trainer_name = config.get("trainer", "")
    trainer = registry.get(trainer_name)
    if trainer is None:
        return {"error": f"Unknown trainer '{trainer_name}'", "trainer": trainer_name}

    available, reason = trainer.is_available()
    if not available:
        return {"error": f"Trainer '{trainer_name}' unavailable: {reason}", "trainer": trainer_name}

    log.info("training start: trainer=%s model=%s", trainer_name, config.get("model") or trainer.default_model)
    result = trainer.train(config, on_progress=on_progress, on_metric=on_metric, is_stopped=is_stopped)
    log.info("training done: trainer=%s best=%s", trainer_name, result.get("best_model"))
    return result
