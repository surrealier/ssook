"""Discover, summarize, and delete finished training runs — pure, no FastAPI.

The trainers write checkpoints under ``<repo>/runs/train/<run_name>/``:

* Ultralytics (detect/segment/pose/classify) -> ``weights/best.pt`` plus an
  ``args.yaml`` (holds ``task:``) and a ``results.csv`` (per-epoch metrics).
* timm/torchvision -> ``best.pt`` at the run root (a torch state dict).

``list_runs`` reports a flat summary per run for the UI; ``delete_run`` removes
one after validating the name stays strictly inside the runs root (no traversal
or absolute escape). This module stays import-light on purpose — it never loads
torch/yaml-heavy deps so the router can call it in the ONNX-only base env. For
timm checkpoints we therefore report ``has_weights`` only and leave task/metrics
best-effort empty rather than unpickling the torch dict.
"""
from __future__ import annotations

import csv
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

log = logging.getLogger("ssook.training")


def default_runs_dir() -> Path:
    # Mirror trainers._default_runs_dir() without importing trainers (keeps this
    # module free of the heavy lazy-import chain). Kept in sync by location.
    return Path(__file__).resolve().parents[2] / "runs" / "train"


def _parse_task(run_dir: Path) -> Optional[str]:
    """Read the ``task:`` key from a YOLO run's args.yaml, if present."""
    args_yaml = run_dir / "args.yaml"
    if not args_yaml.is_file():
        return None
    # A tiny hand-roll avoids importing PyYAML just to read one scalar key.
    try:
        with open(args_yaml, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("task:"):
                    value = stripped.split(":", 1)[1].strip().strip("'\"")
                    return value or None
    except OSError:
        return None
    return None


def _parse_last_metrics(run_dir: Path) -> dict:
    """Return the last row of a YOLO results.csv as {column: float}.

    Headers are whitespace-stripped (Ultralytics pads them); only numeric cells
    are kept so the epoch index and any string columns are dropped.
    """
    results_csv = run_dir / "results.csv"
    if not results_csv.is_file():
        return {}
    try:
        with open(results_csv, "r", encoding="utf-8", errors="ignore", newline="") as f:
            rows = list(csv.reader(f))
    except OSError:
        return {}
    if len(rows) < 2:
        return {}
    headers = [h.strip() for h in rows[0]]
    last = rows[-1]
    metrics: dict = {}
    for header, cell in zip(headers, last):
        cell = cell.strip()
        try:
            metrics[header] = float(cell)
        except ValueError:
            continue  # skip non-numeric (e.g. timestamps)
    return metrics


def _detect_checkpoint(run_dir: Path) -> str:
    """Return the abs path of the run's best checkpoint, or '' if none.

    YOLO writes weights/best.pt; timm writes best.pt at the run root.
    """
    yolo_best = run_dir / "weights" / "best.pt"
    if yolo_best.is_file():
        return str(yolo_best)
    timm_best = run_dir / "best.pt"
    if timm_best.is_file():
        return str(timm_best)
    return ""


def _dir_size_mb(run_dir: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(run_dir):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return round(total / (1024 * 1024), 2)


def _framework_for(run_dir: Path, best_model: str) -> Optional[str]:
    """Best-effort framework guess from the checkpoint layout."""
    if best_model.endswith(os.path.join("weights", "best.pt")) or (run_dir / "weights" / "best.pt").is_file():
        return "ultralytics"
    if (run_dir / "best.pt").is_file():
        return "torchvision"
    return None


def list_runs(runs_dir: Optional[str] = None) -> list[dict]:
    """Summarize every run subdir under ``runs_dir`` (default <repo>/runs/train).

    One bad/corrupt run dir is skipped rather than failing the whole listing.
    Sorted by mtime descending (newest first).
    """
    root = Path(runs_dir) if runs_dir else default_runs_dir()
    if not root.is_dir():
        return []

    runs: list[dict] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            best_model = _detect_checkpoint(entry)
            run = {
                "name": entry.name,
                "task": _parse_task(entry),
                "framework": _framework_for(entry, best_model),
                "best_model": best_model,
                "last_metrics": _parse_last_metrics(entry),
                "mtime": entry.stat().st_mtime,
                "size_mb": _dir_size_mb(entry),
                "has_weights": bool(best_model),
            }
            runs.append(run)
        except OSError as e:
            log.warning("list_runs: skipping %s: %s", entry, e)
            continue

    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def delete_run(name: str, runs_dir: Optional[str] = None) -> bool:
    """Delete a single run dir. Raises ValueError on traversal/missing run.

    ``name`` is resolved against the runs root and must stay strictly inside it
    (blocks ``../`` escapes and absolute paths). Returns True once removed.
    """
    root = (Path(runs_dir) if runs_dir else default_runs_dir())
    root_real = os.path.realpath(str(root))
    target_real = os.path.realpath(os.path.join(root_real, name))

    # commonpath raises on mixed drives (Windows); treat that as an escape too.
    try:
        inside = os.path.commonpath([root_real, target_real]) == root_real
    except ValueError:
        inside = False
    if not inside or target_real == root_real:
        raise ValueError(f"Invalid run name '{name}' (escapes the runs directory)")

    if not os.path.isdir(target_real):
        raise ValueError(f"Run '{name}' not found")

    shutil.rmtree(target_real)
    log.info("deleted training run: %s", target_real)
    return True
