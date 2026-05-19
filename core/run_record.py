"""Reproducible Run records.

Every long-running task (eval, bench, VLM batch, dataset op) should drop
a `run.yaml` next to its output. Future-you (or a colleague) can then
answer "how did this result come about?" without grep-archaeology.

Schema:
    ssook_version: str
    run_type:      str            # eval | bench | vlm_batch | data_split | ...
    started_at:    ISO8601
    ended_at:      ISO8601
    duration_ms:   int
    trace_id:      str (12-char hex)
    inputs:        dict           # caller-supplied; usually params + paths
    model:         dict|None      # {path, sha256, size_mb, mtime}
    env:           dict           # python, ort, ssook, platform
    notes:         str|None

The writer is fail-safe — if anything blows up while computing the
record, we log and move on. The result is the priority, not the metadata.
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

log = logging.getLogger("ssook.runrecord")

_SSOOK_VERSION = "1.5.3"  # kept in sync with server/__init__.py FastAPI title


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _file_sha256(path: str, chunk: int = 1 << 20) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def model_meta(path: Optional[str]) -> Optional[dict]:
    if not path or not os.path.isfile(path):
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return {
        "path": os.path.abspath(path),
        "sha256": _file_sha256(path),
        "size_mb": round(st.st_size / (1024 * 1024), 3),
        "mtime": _utc_from_ts(st.st_mtime),
    }


def _utc_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def env_snapshot() -> dict:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ssook_version": _SSOOK_VERSION,
    }
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
        info["onnxruntime_providers"] = list(ort.get_available_providers())
    except Exception:
        pass
    try:
        import numpy as np
        info["numpy"] = np.__version__
    except Exception:
        pass
    try:
        import cv2
        info["opencv"] = cv2.__version__
    except Exception:
        pass
    try:
        import torch  # noqa
        # torch.__version__ is a TorchVersion object — coerce to plain str
        # or PyYAML emits a !!python/object tag that won't safe_load.
        info["torch"] = str(torch.__version__)
        cuda_ver = getattr(torch.version, "cuda", None)
        info["cuda"] = str(cuda_ver) if cuda_ver is not None else None
    except Exception:
        pass
    return info


class RunRecorder:
    """Lightweight start/stop context manager.

    Usage:
        with RunRecorder(run_type="eval", output_dir=d, inputs=req.dict(),
                         model_path=req.model_path) as rec:
            ... do work ...
            rec.note("Stopped early") # optional
        # On exit, writes <output_dir>/run.yaml — no exception leaks out.
    """

    def __init__(self, run_type: str, output_dir: str | os.PathLike,
                 inputs: Optional[dict] = None,
                 model_path: Optional[str] = None):
        self.run_type = run_type
        self.output_dir = Path(output_dir)
        self.inputs = inputs or {}
        self.model_path = model_path
        self.trace_id = uuid.uuid4().hex[:12]
        self._start = time.perf_counter()
        self.started_at = _utc()
        self._notes: list[str] = []
        self.error: Optional[str] = None

    def note(self, msg: str) -> None:
        self._notes.append(msg)

    def __enter__(self) -> "RunRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc:
            self.error = f"{exc_type.__name__}: {exc}"
        self.write()
        return False  # don't swallow

    def write(self) -> Optional[Path]:
        """Drop the record. Returns the path or None on failure."""
        ended_at = _utc()
        duration_ms = int((time.perf_counter() - self._start) * 1000)
        record = {
            "ssook_version": _SSOOK_VERSION,
            "run_type": self.run_type,
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "ended_at": ended_at,
            "duration_ms": duration_ms,
            "inputs": _safe(self.inputs),
            "model": model_meta(self.model_path),
            "env": env_snapshot(),
            "notes": self._notes or None,
        }
        if self.error:
            record["error"] = self.error
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out = self.output_dir / "run.yaml"
            with out.open("w", encoding="utf-8") as f:
                yaml.dump(record, f, allow_unicode=True, sort_keys=False)
            return out
        except OSError as e:
            log.warning("run.yaml write failed for %s: %s", self.output_dir, e)
            return None


def _safe(obj: Any) -> Any:
    """Coerce arbitrary input into YAML-serialisable values."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if hasattr(obj, "model_dump"):  # pydantic v2
        try:
            return _safe(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "dict"):  # pydantic v1
        try:
            return _safe(obj.dict())
        except Exception:
            pass
    return str(obj)
