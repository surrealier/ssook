"""Sync PyInstaller hidden_imports blocks in ssook.spec.

Walks `core/` and `server/`, builds the canonical hidden_imports lines,
and rewrites the marked blocks in the spec file. Run after adding a new
module so the frozen build doesn't miss it.

    python tools/sync_pyinstaller_spec.py
    python tools/sync_pyinstaller_spec.py --check   # CI mode: exit 1 if out-of-date
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _walk(pkg: str) -> list[str]:
    """Return every importable dotted module path under <pkg>/."""
    out: list[str] = []
    base = ROOT / pkg
    if not base.is_dir():
        return out
    for path in sorted(base.rglob("*.py")):
        if path.name == "__init__.py":
            mod = ".".join(path.parent.relative_to(ROOT).parts)
            if mod and mod != pkg:
                out.append(mod)
            elif mod == pkg:
                out.append(pkg)
            continue
        rel = path.relative_to(ROOT)
        if "__pycache__" in rel.parts:
            continue
        mod = ".".join(rel.with_suffix("").parts)
        # Skip private helpers and tests inside the package (none today).
        out.append(mod)
    return sorted(set(out))


def _format_block(modules: list[str], indent: str = "        ") -> str:
    if not modules:
        return ""
    lines = []
    current = indent
    for m in modules:
        chunk = f"'{m}', "
        if len(current) + len(chunk) > 96 and current.strip():
            lines.append(current.rstrip())
            current = indent
        current += chunk
    if current.strip():
        lines.append(current.rstrip())
    return "\n".join(lines)


def _rewrite_block(text: str, marker_start: str, marker_end: str, replacement: str) -> str:
    pat = re.compile(
        rf"(# {re.escape(marker_start)}[^\n]*\n)(.*?)(\n[ \t]*# {re.escape(marker_end)})",
        re.DOTALL,
    )
    if not pat.search(text):
        raise SystemExit(f"Marker {marker_start!r} or {marker_end!r} missing in spec")
    return pat.sub(lambda m: m.group(1) + replacement + m.group(3), text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if spec is stale")
    args = ap.parse_args(argv)

    spec_path = ROOT / "ssook.spec"
    if not spec_path.exists():
        print("ssook.spec not found", file=sys.stderr)
        return 1

    text = spec_path.read_text(encoding="utf-8")
    core_modules = [m for m in _walk("core") if m != "core"]
    server_modules = _walk("server")

    new = text
    new = _rewrite_block(
        new, "core.* (auto-sync block)", "server.* (auto-sync block)",
        _format_block(core_modules) + "\n",
    )
    new = _rewrite_block(
        new, "server.* (auto-sync block)", "End auto-sync.",
        _format_block(server_modules) + "\n",
    )

    if new == text:
        print("ssook.spec already up-to-date")
        return 0

    if args.check:
        print("ssook.spec is OUT OF DATE — run tools/sync_pyinstaller_spec.py", file=sys.stderr)
        return 1

    spec_path.write_text(new, encoding="utf-8")
    print(f"ssook.spec updated (core={len(core_modules)}, server={len(server_modules)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
