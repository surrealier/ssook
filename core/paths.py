"""Single source of truth for ssook's temp / cache / log directories.

Routes used to scatter outputs across `tempfile.gettempdir()`, the project
root, and hard-coded paths like `_cmp_dir`. That makes cleanup unreliable
and `%TEMP%` clears can wipe partial work mid-run. This module centralises
the layout:

    settings/
      logs/      — rotating logs (managed by core.logging_setup)
      cache/<category>/  — model meta, embeddings, etc. (long-lived)
      tmp/<category>/    — short-lived artefacts (cleanup_stale eligible)

All paths are project-relative by default but overridable via env:
    SSOOK_CACHE_DIR  → cache root
    SSOOK_TMP_DIR    → tmp root
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Iterable

from core import env

log = logging.getLogger("ssook.paths")

_ROOT = Path(__file__).resolve().parent.parent


def _resolve(env_key: str, fallback_rel: str) -> Path:
    custom = env.get_str(env_key, "").strip()
    if custom:
        p = Path(custom).expanduser()
    else:
        p = _ROOT / fallback_rel
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_root() -> Path:
    return _resolve("SSOOK_CACHE_DIR", "settings/cache")


def tmp_root() -> Path:
    return _resolve("SSOOK_TMP_DIR", "settings/tmp")


def cache_dir(category: str) -> Path:
    """Long-lived cache for a category (model_meta, embeddings, dhash, ...)."""
    p = cache_root() / category
    p.mkdir(parents=True, exist_ok=True)
    return p


def tmp_dir(category: str) -> Path:
    """Short-lived working dir for a category (compare, bench, vlm, ...)."""
    p = tmp_root() / category
    p.mkdir(parents=True, exist_ok=True)
    return p


def cleanup_stale(category: str, older_than_days: int = 7) -> int:
    """Delete tmp files older than N days. Returns count removed."""
    root = tmp_dir(category)
    cutoff = time.time() - older_than_days * 86400
    removed = 0
    for child in root.iterdir():
        try:
            if child.stat().st_mtime < cutoff:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
                removed += 1
        except OSError as e:
            log.debug("cleanup_stale skip %s: %s", child, e)
    if removed:
        log.info("cleanup_stale(%s): removed %d entries", category, removed)
    return removed


def cleanup_all(categories: Iterable[str] = ("compare", "bench", "vlm", "embedding")) -> int:
    total = 0
    for c in categories:
        total += cleanup_stale(c)
    return total
