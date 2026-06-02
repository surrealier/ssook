"""Export-target registry — the plugin contract for the Deployment module.

Deliberately parallels ``core.optimizer_registry`` so the two subsystems feel
the same to callers and to the frontend: a ``BaseExportTarget`` ABC, a
registry with register/get/list_*, and a module-level singleton populated at
import time. The router lists targets via ``to_dict()`` and runs one via
``export()``.
"""
from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from typing import Callable, Optional

log = logging.getLogger("ssook.deployment")


class BaseExportTarget(ABC):
    """One deployable runtime/EP target.

    Concrete targets set the class attrs and implement ``export``. Keep heavy
    backend imports *inside* ``export``/``is_available`` (lazy) so the registry
    can be listed in an environment that lacks the backend.
    """

    name: str = ""            # stable id, e.g. "onnx", "tensorrt", "openvino"
    label: str = ""           # human label for the UI
    category: str = ""        # runtime family: onnxruntime|tensorrt|openvino|coreml|torchscript
    description: str = ""
    provider: Optional[str] = None   # ORT provider this artifact runs on, if any
    output_ext: str = ".onnx"        # primary artifact extension
    # OS where the artifact can be BUILT (not necessarily run). CoreML build
    # is macOS-only; the rest build cross-platform.
    platforms: tuple[str, ...] = ("win32", "linux", "darwin")

    def is_available(self) -> tuple[bool, str]:
        """Return (available, reason). Default: only a platform check.

        Targets with extra backends override this to also probe imports and
        return a friendly "pip install ..." reason when missing — surfaced in
        the UI so an unbuildable target is greyed out with an explanation.
        """
        if sys.platform not in self.platforms:
            return False, f"not supported on {sys.platform} (needs {'/'.join(self.platforms)})"
        return True, ""

    @abstractmethod
    def export(
        self,
        model_path: str,
        output_path: str,
        opts: dict,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        is_stopped: Optional[Callable[[], bool]] = None,
    ) -> dict:
        """Produce the artifact at ``output_path``.

        ``opts`` carries the configurable levers: opset, imgsz, batch,
        dynamic, half/precision, calibration_dir, device, workspace_gb, ...
        Returns a flat result dict that always contains ``output_path`` plus
        ``target`` and size metadata (mirrors the optimizer result shape).
        """
        ...

    def to_dict(self) -> dict:
        available, reason = self.is_available()
        return {
            "name": self.name,
            "label": self.label or self.name,
            "category": self.category,
            "description": self.description,
            "provider": self.provider,
            "output_ext": self.output_ext,
            "platforms": list(self.platforms),
            "available": available,
            "unavailable_reason": reason,
        }


class ExportTargetRegistry:
    def __init__(self) -> None:
        self._targets: dict[str, BaseExportTarget] = {}

    def register(self, target: BaseExportTarget) -> None:
        if target.name in self._targets:
            raise ValueError(f"Export target '{target.name}' already registered")
        self._targets[target.name] = target

    def get(self, name: str) -> Optional[BaseExportTarget]:
        return self._targets.get(name)

    def list_by_category(self, category: str) -> list[BaseExportTarget]:
        return [t for t in self._targets.values() if t.category == category]

    def list_all(self) -> list[BaseExportTarget]:
        return list(self._targets.values())


# Module-level singleton — populated by core.deployment.targets at import time.
registry = ExportTargetRegistry()
