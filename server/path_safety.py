"""Path safety helpers for FastAPI handlers.

ssook is a single-user local desktop app, so the threat model is mostly
*accidental* misuse: a malformed path that escapes into the filesystem, a
malicious browser tab POSTing during a vulnerable moment, a footgun in
future LAN-exposure scenarios. We don't whitelist roots by default because
users legitimately point at arbitrary dataset folders; instead we enforce:

- absolute, normalized paths (no `..` traversal after resolve)
- no NUL bytes
- optional must_exist / must_be_file / must_be_dir
- optional extension allowlist
- optional `roots` allowlist (use when the route should only touch a
  bounded surface, like the ssook project's bundled samples)

Routes opt in by calling `safe_path(...)` (or one of the wrappers) at the
boundary, or by attaching it as a Pydantic `@field_validator`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


class UnsafePathError(ValueError):
    """Raised when a user-supplied path fails safety validation."""

    def __init__(self, code: str, msg: str):
        super().__init__(msg)
        self.code = code


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_MODEL_EXTS = {".onnx"}
_LABEL_EXTS = {".txt", ".json", ".xml"}


def safe_path(
    user_path: str,
    *,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    allowed_exts: Optional[Iterable[str]] = None,
    roots: Optional[Iterable[str]] = None,
) -> str:
    """Normalize and validate a user-supplied path.

    Returns the resolved absolute path. Raises UnsafePathError on any
    boundary violation.
    """
    if user_path is None or user_path == "":
        raise UnsafePathError("EMPTY", "Path is empty")
    if not isinstance(user_path, str):
        raise UnsafePathError("TYPE", f"Path must be a string, got {type(user_path).__name__}")
    if "\x00" in user_path:
        raise UnsafePathError("NUL", "Path contains a NUL byte")

    try:
        # `resolve(strict=False)` follows symlinks but does not require existence.
        resolved = Path(user_path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise UnsafePathError("RESOLVE", f"Cannot resolve path: {e}") from e

    abs_path = str(resolved)

    if must_exist and not resolved.exists():
        raise UnsafePathError("NOT_FOUND", f"Path does not exist: {abs_path}")
    if must_be_file and resolved.exists() and not resolved.is_file():
        raise UnsafePathError("NOT_FILE", f"Path is not a file: {abs_path}")
    if must_be_dir and resolved.exists() and not resolved.is_dir():
        raise UnsafePathError("NOT_DIR", f"Path is not a directory: {abs_path}")

    if allowed_exts:
        ext = resolved.suffix.lower()
        allowed = {e.lower() for e in allowed_exts}
        if ext not in allowed:
            raise UnsafePathError(
                "BAD_EXT",
                f"Extension '{ext}' not allowed (expected one of {sorted(allowed)})",
            )

    if roots:
        root_paths = [Path(r).expanduser().resolve(strict=False) for r in roots]
        try:
            ok = any(_is_within(resolved, rp) for rp in root_paths)
        except OSError:
            ok = False
        if not ok:
            raise UnsafePathError(
                "OUT_OF_ROOT",
                f"Path is outside the allowed roots: {abs_path}",
            )

    return abs_path


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


# ── Convenience wrappers ─────────────────────────────────────────────────
def safe_image_file(p: str) -> str:
    return safe_path(p, must_exist=True, must_be_file=True, allowed_exts=_IMAGE_EXTS)


def safe_video_file(p: str) -> str:
    return safe_path(p, must_exist=True, must_be_file=True, allowed_exts=_VIDEO_EXTS)


def safe_model_file(p: str) -> str:
    return safe_path(p, must_exist=True, must_be_file=True, allowed_exts=_MODEL_EXTS)


def safe_image_dir(p: str) -> str:
    return safe_path(p, must_exist=True, must_be_dir=True)


def safe_label_dir(p: str) -> str:
    return safe_path(p, must_exist=True, must_be_dir=True)
