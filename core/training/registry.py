"""Trainer registry — the plugin contract for the Training module.

Mirrors ``core.optimizer_registry`` / ``core.deployment.registry``: a
``BaseTrainer`` ABC, a registry with register/get/list_*, and a module-level
singleton populated at import time. The router lists trainers via
``to_dict()`` and runs one via ``train()``.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

log = logging.getLogger("ssook.training")

# Progress callback: (epoch_done, epochs_total, message)
ProgressFn = Callable[[int, int, str], None]
# Metric callback: (epoch_index_1based, metrics_dict) — one row per epoch.
MetricFn = Callable[[int, dict], None]
# Cooperative-stop probe: return True to ask the trainer to stop at the next
# epoch boundary.
StopFn = Callable[[], bool]


class BaseTrainer(ABC):
    """One trainable model family.

    Keep torch / ultralytics / timm imports *inside* ``train``/``is_available``
    (lazy) so the registry lists in an env without a training framework.
    """

    name: str = ""          # stable id, e.g. "yolo_detect", "timm_classify"
    label: str = ""         # human label
    framework: str = ""     # "ultralytics" | "torchvision"
    task: str = ""          # "detect" | "segment" | "pose" | "classify"
    description: str = ""
    # "yolo"  -> images + YOLO .txt labels (+ generated data.yaml)
    # "imagefolder" -> dir of class-named subfolders of images
    dataset_kind: str = "yolo"
    default_model: str = ""  # default base checkpoint / backbone

    def is_available(self) -> tuple[bool, str]:
        return True, ""

    @abstractmethod
    def train(
        self,
        config: dict,
        on_progress: Optional[ProgressFn] = None,
        on_metric: Optional[MetricFn] = None,
        is_stopped: Optional[StopFn] = None,
    ) -> dict:
        """Train and return a result dict containing at least ``best_model``
        (path to the best checkpoint), ``save_dir``, ``epochs_completed`` and
        ``metrics``."""
        ...

    def to_dict(self) -> dict:
        available, reason = self.is_available()
        return {
            "name": self.name,
            "label": self.label or self.name,
            "framework": self.framework,
            "task": self.task,
            "description": self.description,
            "dataset_kind": self.dataset_kind,
            "default_model": self.default_model,
            "available": available,
            "unavailable_reason": reason,
        }


class TrainerRegistry:
    def __init__(self) -> None:
        self._trainers: dict[str, BaseTrainer] = {}

    def register(self, trainer: BaseTrainer) -> None:
        if trainer.name in self._trainers:
            raise ValueError(f"Trainer '{trainer.name}' already registered")
        self._trainers[trainer.name] = trainer

    def get(self, name: str) -> Optional[BaseTrainer]:
        return self._trainers.get(name)

    def list_by_task(self, task: str) -> list[BaseTrainer]:
        return [t for t in self._trainers.values() if t.task == task]

    def list_all(self) -> list[BaseTrainer]:
        return list(self._trainers.values())


# Module-level singleton — populated by core.training.trainers at import time.
registry = TrainerRegistry()
