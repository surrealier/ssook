"""Rotating-file logging for ssook.

Single entry point: call `configure()` once at startup. Subsequent calls
are no-ops so it's safe to invoke from both `run_web.py` and
`server.__init__` without double-handlers.

Log file: `settings/logs/ssook.log` (10 MiB × 5 rotation).
Level: SSOOK_LOG_LEVEL env (INFO default).
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False


def configure(log_dir: str | None = None, level: str | None = None) -> Path:
    global _CONFIGURED
    if _CONFIGURED:
        # Return current log path for callers that want to surface it.
        return _current_log_path()

    log_dir_p = Path(log_dir or os.environ.get("SSOOK_LOG_DIR") or "settings/logs")
    log_dir_p.mkdir(parents=True, exist_ok=True)
    log_path = log_dir_p / "ssook.log"

    lvl_name = (level or os.environ.get("SSOOK_LOG_LEVEL") or "INFO").upper()
    lvl = getattr(logging, lvl_name, logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(lvl)
    # Avoid duplicate handlers if something configured root earlier.
    for h in list(root.handlers):
        if isinstance(h, (RotatingFileHandler, logging.StreamHandler)):
            root.removeHandler(h)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    # Tame chatty libraries.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)

    _CONFIGURED = True
    _set_log_path(log_path)
    logging.getLogger("ssook.boot").info("logging configured (level=%s, file=%s)", lvl_name, log_path)
    return log_path


_LOG_PATH: Path | None = None


def _set_log_path(p: Path) -> None:
    global _LOG_PATH
    _LOG_PATH = p


def _current_log_path() -> Path:
    return _LOG_PATH or Path("settings/logs/ssook.log")
