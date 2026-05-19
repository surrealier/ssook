"""Model metadata cache.

`core.model_inspector.inspect_model` re-parses the ONNX graph on every
call, which adds ~50-200ms per click in the inspector/profiler tabs.
This cache keys on (path, mtime, size) and stores the JSON-serializable
metadata dict on disk under `settings/cache/model_meta/`.

Invalidation is automatic: any change to the file's mtime or size
forces a refresh. Manual eviction is rarely needed; `clear()` wipes
the whole directory.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

from core import paths

log = logging.getLogger("ssook.model_cache")


def _fingerprint(path: str) -> Optional[tuple[int, int]]:
    try:
        st = os.stat(path)
        return int(st.st_mtime), int(st.st_size)
    except OSError:
        return None


def _key(path: str) -> str:
    fp = _fingerprint(path)
    if fp is None:
        return ""
    h = hashlib.sha1(f"{os.path.abspath(path)}|{fp[0]}|{fp[1]}".encode()).hexdigest()
    return h[:16]


def _cache_path(path: str) -> Path:
    return paths.cache_dir("model_meta") / f"{_key(path)}.json"


def get_or_compute(path: str, compute: Callable[[str], dict]) -> dict:
    """Return cached metadata for `path`, else run `compute(path)` and cache."""
    if not path or not os.path.isfile(path):
        return compute(path)  # let downstream raise the right error
    cp = _cache_path(path)
    if cp.exists():
        try:
            with cp.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.debug("cache read failed for %s: %s — recomputing", path, e)
    data = compute(path)
    try:
        with cp.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(data), f, ensure_ascii=False, indent=2)
    except (OSError, TypeError) as e:
        log.debug("cache write failed for %s: %s", path, e)
    return data


def _json_safe(obj: Any) -> Any:
    """Coerce numpy types / dataclasses to plain JSON-serializable values."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if hasattr(obj, "tolist"):  # numpy array
        try:
            return obj.tolist()
        except Exception:
            return str(obj)
    return str(obj)


def clear() -> int:
    """Wipe every cached entry. Returns count removed."""
    root = paths.cache_dir("model_meta")
    n = 0
    for p in root.glob("*.json"):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n
