"""Minimal .env loader for ssook.

We deliberately avoid `python-dotenv` for a 20-line implementation. The
parser supports KEY=VALUE, # comments, blank lines, and basic quoting
(`KEY="value with spaces"`). Existing OS environment variables always win
so a developer can override via shell without editing .env.

Call `load_env()` once at boot (e.g. from server.__init__). Read via
`get_int("SSOOK_PORT", 8765)` / `get_str(...)` / `get_bool(...)`.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional


log = logging.getLogger("ssook.env")

_LOADED = False


def load_env(paths: Optional[Iterable[str | Path]] = None, override: bool = False) -> dict:
    """Read .env files (project root + cwd by default) into os.environ.

    Returns the dict of keys actually applied (for diagnostics).
    """
    global _LOADED
    applied: dict[str, str] = {}
    if paths is None:
        root = Path(__file__).resolve().parent.parent
        paths = [root / ".env", Path.cwd() / ".env"]
    seen: set[Path] = set()
    for p in paths:
        p = Path(p)
        if not p.exists() or p in seen:
            continue
        seen.add(p)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as e:
            log.warning("Cannot read %s: %s", p, e)
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            # Strip optional surrounding quotes.
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            if not key:
                continue
            if not override and key in os.environ:
                continue
            os.environ[key] = val
            applied[key] = val
    _LOADED = True
    if applied:
        log.info("Loaded %d env keys from .env (%s)", len(applied), ", ".join(sorted(applied)))
    return applied


def get_str(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def get_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("env %s is not an int: %r — using default %d", key, raw, default)
        return default


def get_bool(key: str, default: bool = False) -> bool:
    raw = (os.environ.get(key) or "").strip().lower()
    if raw == "":
        return default
    return raw in {"1", "true", "yes", "on"}


def get_path_list(key: str) -> list[str]:
    raw = os.environ.get(key, "")
    return [p.strip() for p in raw.split(",") if p.strip()]
