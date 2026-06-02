"""YOLO dataset preparation for training.

Turns a loose image+label directory (like the repo's flat ``labeling/`` dir:
``cane_01.png`` + ``cane_01.txt``) into a standard Ultralytics layout with a
generated ``data.yaml``:

    <out_root>/
      train/images/*, train/labels/*
      val/images/*,   val/labels/*
      data.yaml   (path/train/val/nc/names)

Files are hard-linked (copy fallback) so even large datasets prepare quickly
without duplicating bytes. If the source dir already contains a ``data.yaml``
it is used as-is. Class ids are discovered from the labels; names come from the
caller, else a class catalog match, else ``class_{i}``.
"""
from __future__ import annotations

import logging
import os
import random
import shutil
from pathlib import Path
from typing import Optional

log = logging.getLogger("ssook.training")

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _glob_images(d: str) -> list[str]:
    out: list[str] = []
    for root, _dirs, files in os.walk(d):
        for f in files:
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTS:
                out.append(os.path.join(root, f))
    return sorted(out)


def _label_for(image_path: str, labels_dir: Optional[str]) -> Optional[str]:
    """Resolve the YOLO .txt label for an image (sibling, or in labels_dir)."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    if labels_dir:
        cand = os.path.join(labels_dir, stem + ".txt")
    else:
        cand = os.path.join(os.path.dirname(image_path), stem + ".txt")
    return cand if os.path.isfile(cand) else None


def discover_classes(label_files: list[str]) -> int:
    """Return nc = max class id + 1 across all label files (0 if none)."""
    max_id = -1
    for lf in label_files:
        try:
            with open(lf, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cid = int(float(line.split()[0]))
                    except (ValueError, IndexError):
                        continue
                    if cid > max_id:
                        max_id = cid
        except OSError:
            continue
    return max_id + 1


def resolve_names(nc: int, names_arg) -> dict[int, str]:
    """Build a {id: name} dict from caller input, a catalog match, or fallback."""
    if isinstance(names_arg, dict) and names_arg:
        return {int(k): str(v) for k, v in names_arg.items()}
    if isinstance(names_arg, (list, tuple)) and names_arg:
        return {i: str(n) for i, n in enumerate(names_arg)}
    if isinstance(names_arg, str) and names_arg.strip():
        parts = [p.strip() for p in names_arg.split(",") if p.strip()]
        if parts:
            return {i: n for i, n in enumerate(parts)}
    try:
        from core.class_catalog import suggest, as_class_names
        cat = suggest(nc)
        if cat:
            return as_class_names(cat)
    except Exception:
        pass
    return {i: f"class_{i}" for i in range(max(nc, 0))}


def _link_or_copy(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)  # hard link — no byte duplication on same volume
    except OSError:
        shutil.copy2(src, dst)


def find_existing_data_yaml(images_dir: str) -> Optional[str]:
    for cand in ("data.yaml", "data.yml", "dataset.yaml"):
        p = os.path.join(images_dir, cand)
        if os.path.isfile(p):
            return p
    return None


def prepare_yolo_dataset(
    images_dir: str,
    labels_dir: Optional[str] = None,
    names=None,
    val_ratio: float = 0.2,
    seed: int = 0,
    out_root: Optional[str] = None,
    keep: Optional[dict] = None,
) -> dict:
    """Prepare a training-ready YOLO dataset; return a summary dict.

    If ``images_dir`` already holds a data.yaml it is returned untouched.
    ``keep``/multilabel-style fork options are written into data.yaml verbatim
    when provided (a vendored Ultralytics fork consumes them; stock ultralytics
    ignores unknown keys).
    """
    existing = find_existing_data_yaml(images_dir)
    if existing:
        log.info("dataset: using existing data.yaml at %s", existing)
        return {"data_yaml": existing, "prepared": False, "source": images_dir}

    images = _glob_images(images_dir)
    if not images:
        raise ValueError(f"No images found under {images_dir}")

    pairs: list[tuple[str, Optional[str]]] = [(img, _label_for(img, labels_dir)) for img in images]
    label_files = [lbl for _img, lbl in pairs if lbl]
    if not label_files:
        raise ValueError(
            f"No YOLO .txt labels found for images in {images_dir}"
            + (f" / {labels_dir}" if labels_dir else "")
        )

    nc = discover_classes(label_files)
    name_map = resolve_names(nc, names)
    # Ultralytics requires len(names) == nc. Reconcile so the data.yaml is
    # internally consistent even when caller names are shorter/longer than the
    # discovered class count: cover the highest of (discovered nc, max named id),
    # then fill any gaps with class_{i}.
    nc = max(nc, (max(name_map) + 1) if name_map else 0) or 1
    name_map = {i: name_map.get(i, f"class_{i}") for i in range(nc)}

    # Deterministic split.
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
    val_pairs = shuffled[:n_val]
    train_pairs = shuffled[n_val:] or shuffled  # never leave train empty

    if out_root is None:
        from core.paths import tmp_dir
        safe = "".join(c if c.isalnum() else "_" for c in os.path.basename(os.path.normpath(images_dir)))
        out_root = str(tmp_dir("training") / f"{safe or 'dataset'}_split")

    # Rebuild cleanly so a re-prep never mixes with a stale split.
    if os.path.isdir(out_root):
        shutil.rmtree(out_root, ignore_errors=True)
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_root, split, "labels"), exist_ok=True)

    def _place(split: str, items: list[tuple[str, Optional[str]]]) -> int:
        n = 0
        for img, lbl in items:
            base = os.path.basename(img)
            _link_or_copy(img, os.path.join(out_root, split, "images", base))
            if lbl:
                stem = os.path.splitext(base)[0]
                _link_or_copy(lbl, os.path.join(out_root, split, "labels", stem + ".txt"))
            n += 1
        return n

    n_train = _place("train", train_pairs)
    n_val_written = _place("val", val_pairs)

    data_yaml = os.path.join(out_root, "data.yaml")
    payload = {
        "path": out_root,
        "train": "train/images",
        # A single-image dataset has no val split; point val at train so
        # Ultralytics validation runs on something instead of an empty dir.
        "val": "val/images" if n_val_written > 0 else "train/images",
        "nc": nc,
        "names": {int(k): v for k, v in name_map.items()},
    }
    if keep:
        payload["keep"] = keep
    _write_yaml(data_yaml, payload)

    log.info("dataset prepared: %d train / %d val, nc=%d -> %s",
             n_train, n_val_written, nc, data_yaml)
    return {
        "data_yaml": data_yaml,
        "prepared": True,
        "out_root": out_root,
        "nc": nc,
        "names": payload["names"],
        "n_train": n_train,
        "n_val": n_val_written,
        "n_total": len(pairs),
        "n_unlabeled": len(pairs) - len(label_files),
    }


def _write_yaml(path: str, data: dict) -> None:
    """Write data.yaml — use PyYAML if present, else a minimal hand-roll."""
    try:
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        return
    except ImportError:
        pass
    lines = []
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {vv}")
        else:
            lines.append(f"{k}: {v}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
