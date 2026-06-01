"""/api/data/* 라우터."""
import csv
import glob as glob_module
import io
import os
import random
import shutil
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel

from server.state import explorer_state, splitter_state, converter_state, remapper_state, merger_state, sampler_state, executor
from server.path_safety import (
    UnsafePathError,
    safe_image_dir,
    safe_image_file,
    safe_label_dir,
    safe_path,
    _IMAGE_EXTS,
)
from server.utils import imread, glob_images, encode_jpeg, generate_palette, draw_label

router = APIRouter()


def _find_image_for_stem(input_dir: str, stem: str) -> Optional[str]:
    """Locate the image matching a label stem, probing the full image-extension
    set (DATA-06 — the old 4-ext probe silently skipped .webp/.tif)."""
    for ext in sorted(_IMAGE_EXTS):
        candidate = os.path.join(input_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def _image_dims(img_path: Optional[str]) -> Optional[tuple[int, int]]:
    """Return (width, height) for an image, or None when it cannot be read.
    Callers must NOT fall back to a fixed size — wrong dims corrupt absolute
    coordinates (DATA-06)."""
    if not img_path:
        return None
    img = imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def _parse_yolo_label_line(parts: list[str]) -> Optional[dict]:
    """Parse one YOLO label line into a normalized record, or None if it is not
    a valid annotation. Distinguishes plain bbox (exactly 5 tokens) from
    polygon segmentation (DATA-05); polygons yield their enclosing bbox plus the
    raw vertices so callers can preserve segmentation where the target supports it.

    Returns {cid, cx, cy, w, h, polygon: Optional[list[float]]}.
    Raises ValueError on non-numeric tokens so callers can count bad lines.
    """
    if len(parts) < 5:
        return None
    cid = int(parts[0])
    coords = parts[1:]
    # Polygon: odd token count overall (1 class + even vertex coords).
    if len(parts) > 5 and len(coords) % 2 == 0:
        xs = [float(coords[i]) for i in range(0, len(coords), 2)]
        ys = [float(coords[i]) for i in range(1, len(coords), 2)]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return {
            "cid": cid,
            "cx": (x_min + x_max) / 2,
            "cy": (y_min + y_max) / 2,
            "w": x_max - x_min,
            "h": y_max - y_min,
            "polygon": [c for xy in zip(xs, ys) for c in xy],
        }
    # Plain bbox — exactly 5 tokens.
    cx, cy, bw, bh = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
    return {"cid": cid, "cx": cx, "cy": cy, "w": bw, "h": bh, "polygon": None}


# ── Data: Explorer API ──────────────────────────────────
# NOTE: explorer_state imported from server.state — do NOT re-declare.
@router.post("/api/data/explorer")
async def data_explorer(req: dict):
    try:
        img_dir = safe_image_dir(req.get("img_dir", ""))
        raw_lbl = req.get("label_dir", "")
        lbl_dir = safe_label_dir(raw_lbl) if raw_lbl else ""
    except UnsafePathError as e:
        return {"error": e.code}
    limit = min(int(req.get("limit", 5000)), 50000)
    if explorer_state["running"]:
        return {"error": "Already loading"}
    explorer_state.update(running=True, progress=0, total=0, msg="Scanning...", data=None)

    def _run():
        try:
            imgs = glob_images(img_dir)
            n = len(imgs)
            explorer_state["total"] = n
            class_counts = {}
            img_class_counts = {}  # class -> set of image indices (for image-unit counting)
            file_info = []
            box_sizes = []  # (w, h) normalized
            aspect_ratios = []
            box_aspect_ratios = []
            bad_lines = 0  # un-parseable label lines (DATA-13)
            corrupt_files = 0  # label files that raised while reading
            for i, fp in enumerate(imgs[:limit]):
                # Cooperative cancel — Stop button flips running=False (DATA-11).
                if not explorer_state["running"]:
                    explorer_state.update(msg="Cancelled")
                    return
                explorer_state["progress"] = i + 1
                stem = os.path.splitext(os.path.basename(fp))[0]
                txt = os.path.join(lbl_dir, stem + ".txt") if lbl_dir else ""
                boxes = []
                if txt and os.path.isfile(txt):
                    try:
                        with open(txt) as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) < 5:
                                    continue
                                # Guard per-line so one bad line never aborts the scan (DATA-13).
                                try:
                                    cid = int(parts[0])
                                    bw, bh = float(parts[3]), float(parts[4])
                                except ValueError:
                                    bad_lines += 1
                                    continue
                                boxes.append(cid)
                                class_counts[cid] = class_counts.get(cid, 0) + 1
                                if cid not in img_class_counts:
                                    img_class_counts[cid] = set()
                                img_class_counts[cid].add(i)
                                box_sizes.append({"w": bw, "h": bh})
                                box_aspect_ratios.append(round(bw / max(bh, 1e-6), 2))
                    except OSError:
                        corrupt_files += 1
                # image aspect ratio
                try:
                    im = imread(fp)
                    if im is not None:
                        h_, w_ = im.shape[:2]
                        aspect_ratios.append(round(w_ / max(h_, 1), 2))
                except Exception:
                    pass
                # box_details intentionally omitted — fetched lazily in /preview (DATA-12).
                file_info.append({"name": os.path.basename(fp), "path": fp, "boxes": len(boxes),
                                  "classes": list(set(boxes))})
            img_class_count_dict = {k: len(v) for k, v in img_class_counts.items()}
            explorer_state["data"] = {
                "total": n, "shown": len(file_info), "files": file_info,
                "class_counts": class_counts, "img_class_counts": img_class_count_dict,
                "box_sizes": box_sizes, "aspect_ratios": aspect_ratios,
                "box_aspect_ratios": box_aspect_ratios,
                "bad_lines": bad_lines, "corrupt_files": corrupt_files,
            }
            msg = "Complete"
            if bad_lines or corrupt_files:
                msg = f"Complete ({bad_lines} bad lines, {corrupt_files} unreadable label files)"
            explorer_state.update(running=False, msg=msg)
        except Exception as e:
            explorer_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/data/explorer/status")
async def explorer_status():
    s = dict(explorer_state)
    if not s["running"] and s["data"]:
        data = s["data"]
        s.pop("data")
        s.update(data)
    elif s["running"]:
        s.pop("data", None)
    return s

@router.get("/api/data/explorer/export-stats")
async def explorer_export_stats():
    data = explorer_state.get("data")
    if not data:
        return Response(content="No data loaded", media_type="text/plain", status_code=400)
    files = data.get("files", [])
    class_counts = data.get("class_counts", {})
    img_class_counts = data.get("img_class_counts", {})
    total = data.get("total", len(files))
    images_with_labels = sum(1 for f in files if f.get("boxes", 0) > 0)
    images_without_labels = len(files) - images_with_labels
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["metric", "value"])
    writer.writerow(["total_images", total])
    writer.writerow(["images_scanned", len(files)])
    writer.writerow(["images_with_labels", images_with_labels])
    writer.writerow(["images_without_labels", images_without_labels])
    writer.writerow([])
    writer.writerow(["class_id", "box_count", "image_count"])
    all_classes = sorted(set(list(class_counts.keys()) + list(img_class_counts.keys())), key=lambda x: int(x))
    for cls in all_classes:
        writer.writerow([cls, class_counts.get(cls, 0), img_class_counts.get(cls, 0)])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=explorer_stats.csv"}
    )

@router.post("/api/data/explorer/preview")
async def explorer_preview(req: dict):
    """Return base64 JPEG of image with bounding boxes overlaid."""
    try:
        img_path = safe_image_file(req.get("img_path", ""))
        raw_lbl = req.get("label_dir", "")
        lbl_dir = safe_label_dir(raw_lbl) if raw_lbl else ""
    except UnsafePathError as e:
        return {"error": e.code}
    img = imread(img_path)
    if img is None:
        return {"error": "Cannot read image"}
    h, w = img.shape[:2]
    stem = os.path.splitext(os.path.basename(img_path))[0]
    txt = os.path.join(lbl_dir, stem + ".txt") if lbl_dir else ""
    boxes = []
    if txt and os.path.isfile(txt):
        with open(txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # Tolerate malformed lines so preview never 500s on bad data (DATA-13).
                try:
                    cid = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    continue
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                boxes.append({"cid": cid, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    total_cls = max((b["cid"] for b in boxes), default=0) + 1
    palette = generate_palette(max(total_cls, 1))
    for b in boxes:
        color = palette[b["cid"] % len(palette)]
        cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, 2)
        label = str(b["cid"])
        draw_label(img, label, b["x1"], b["y1"], color, 0.6, 1, True)
    return {"image": encode_jpeg(img, 90), "width": w, "height": h, "box_count": len(boxes)}


# ── Data: Splitter API ─────────────────────────────────
class SplitterRequest(BaseModel):
    img_dir: str
    label_dir: str = ""
    output_dir: str
    train: float = 0.7
    val: float = 0.2
    test: float = 0.1
    strategy: str = "random"  # random | stratified


@router.post("/api/data/splitter")
async def data_splitter(req: SplitterRequest):
    try:
        img_dir = safe_image_dir(req.img_dir)
        label_dir = safe_label_dir(req.label_dir) if req.label_dir else ""
        output_dir = safe_path(req.output_dir, must_be_dir=False)
    except UnsafePathError as e:
        return {"error": e.code}
    if splitter_state["running"]:
        return {"error": "Already running"}
    splitter_state.update(running=True, msg="Splitting...", progress=0, total=0, results={})

    def _run():
        try:
            import random, shutil
            imgs = glob_images(img_dir)
            if not imgs:
                splitter_state.update(running=False, msg="No images found")
                return
            n = len(imgs)
            splitter_state["total"] = n

            # Normalize ratios — treat 0 as empty split.
            ratios = {"train": max(req.train, 0), "val": max(req.val, 0), "test": max(req.test, 0)}
            total_ratio = sum(ratios.values())
            if total_ratio <= 0:
                splitter_state.update(running=False, msg="Error: All ratios are 0")
                return
            norm = {k: v / total_ratio for k, v in ratios.items()}

            if req.strategy == "stratified" and label_dir:
                # Group images by class set
                class_groups = {}
                for fp in imgs:
                    stem = os.path.splitext(os.path.basename(fp))[0]
                    txt = os.path.join(label_dir, stem + ".txt")
                    classes = set()
                    if os.path.isfile(txt):
                        with open(txt) as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) < 5:
                                    continue
                                try:
                                    classes.add(int(parts[0]))
                                except ValueError:
                                    continue  # tolerate malformed lines (DATA-13)
                    key = tuple(sorted(classes)) if classes else (-1,)
                    class_groups.setdefault(key, []).append(fp)
                splits = {"train": [], "val": [], "test": []}
                # Accumulate fractional counts across groups to preserve global ratio.
                frac_train = 0.0
                frac_val = 0.0
                # Splits that should receive coverage (ratio > 0), ordered for tie-breaking.
                active = [s for s in ("train", "val", "test") if ratios[s] > 0]
                for group_imgs in class_groups.values():
                    random.shuffle(group_imgs)
                    gn = len(group_imgs)
                    frac_train += gn * norm["train"]
                    frac_val += gn * norm["val"]
                    n_train = round(frac_train)
                    n_val = round(frac_val)
                    # Clamp so we don't exceed group size.
                    n_train = min(n_train, gn)
                    n_val = min(n_val, gn - n_train)
                    g_train = group_imgs[:n_train]
                    g_val = group_imgs[n_train:n_train + n_val]
                    g_test = group_imgs[n_train + n_val:]
                    # Guarantee rare-class coverage: a group with >=2 images and >=2
                    # active splits must appear in at least 2 splits, otherwise a class
                    # seen only in train would never be validated/tested (DATA-10).
                    if gn >= 2 and len(active) >= 2:
                        present = [s for s, g in (("train", g_train), ("val", g_val), ("test", g_test)) if g]
                        if len(present) < 2:
                            assigned = {"train": g_train, "val": g_val, "test": g_test}
                            donor = present[0] if present else active[0]
                            recipient = next(s for s in active if s != donor)
                            moved = assigned[donor].pop()
                            assigned[recipient].append(moved)
                            g_train, g_val, g_test = assigned["train"], assigned["val"], assigned["test"]
                    splits["train"].extend(g_train)
                    splits["val"].extend(g_val)
                    splits["test"].extend(g_test)
                    frac_train -= len(g_train)
                    frac_val -= len(g_val)
            else:
                # Random split
                random.shuffle(imgs)
                n_train = round(n * norm["train"])
                n_val = round(n * norm["val"])
                # Clamp to avoid exceeding total.
                n_val = min(n_val, n - n_train)
                splits = {"train": imgs[:n_train], "val": imgs[n_train:n_train + n_val], "test": imgs[n_train + n_val:]}

            # Copy files with progress
            total_files = sum(len(v) for v in splits.values())
            splitter_state["total"] = total_files
            done = 0
            for split_name, split_files in splits.items():
                if not split_files:
                    continue
                img_out = os.path.join(output_dir, split_name, "images")
                lbl_out = os.path.join(output_dir, split_name, "labels")
                os.makedirs(img_out, exist_ok=True)
                os.makedirs(lbl_out, exist_ok=True)
                for fp in split_files:
                    if not splitter_state["running"]:
                        splitter_state.update(msg="Cancelled")  # cooperative cancel (DATA-11)
                        return
                    shutil.copy2(fp, img_out)
                    stem = os.path.splitext(os.path.basename(fp))[0]
                    txt = os.path.join(label_dir, stem + ".txt") if label_dir else ""
                    if txt and os.path.isfile(txt):
                        shutil.copy2(txt, lbl_out)
                    done += 1
                    splitter_state["progress"] = done
            splitter_state["results"] = {
                **{k: len(v) for k, v in splits.items()},
                "ratios": {k: round(v, 4) for k, v in norm.items()},  # echo normalized ratios (DATA-10)
            }
            splitter_state.update(running=False, msg="Complete")
        except Exception as e:
            splitter_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/data/splitter/status")
async def splitter_status():
    return splitter_state.snapshot() if hasattr(splitter_state, 'snapshot') else dict(splitter_state)


# ── Data: Converter API ────────────────────────────────
class ConverterRequest(BaseModel):
    input_dir: str
    output_dir: str
    from_fmt: str = "YOLO"
    to_fmt: str = "COCO JSON"

@router.post("/api/data/converter")
async def data_converter(req: ConverterRequest):
    try:
        input_dir = safe_path(req.input_dir, must_exist=True, must_be_dir=True)
        output_dir = safe_path(req.output_dir, must_be_dir=False)
    except UnsafePathError as e:
        return {"error": e.code}
    if converter_state["running"]:
        return {"error": "Already running"}
    converter_state.update(running=True, progress=0, total=0, msg="Converting...", results={})

    def _run():
        try:
            import json as _json
            os.makedirs(output_dir, exist_ok=True)
            label_files = sorted(glob_module.glob(os.path.join(input_dir, "*.txt")))
            img_files = glob_images(input_dir)
            converter_state["total"] = len(label_files) or len(img_files)

            if req.from_fmt == "YOLO" and "COCO" in req.to_fmt:
                coco = {"images": [], "annotations": [], "categories": []}
                ann_id = 1
                cats_seen = set()
                missing_image = 0  # label with no matching image (DATA-06)
                skipped_no_size = 0  # image present but dims undeterminable
                bad_lines = 0
                for i, txt in enumerate(label_files):
                    if not converter_state["running"]:
                        converter_state.update(msg="Cancelled")  # DATA-11
                        return
                    stem = os.path.splitext(os.path.basename(txt))[0]
                    img_path = _find_image_for_stem(input_dir, stem)
                    dims = _image_dims(img_path)
                    converter_state["progress"] = i + 1
                    # Never default to 640×640 — wrong dims corrupt absolute coords (DATA-06).
                    if dims is None:
                        if img_path is None:
                            missing_image += 1
                        else:
                            skipped_no_size += 1
                        continue
                    w, h = dims
                    coco["images"].append({"id": i + 1, "file_name": os.path.basename(img_path), "width": w, "height": h})
                    with open(txt) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            try:
                                rec = _parse_yolo_label_line(parts)
                            except ValueError:
                                bad_lines += 1
                                continue
                            if rec is None:
                                continue
                            cx, cy, bw, bh = rec["cx"], rec["cy"], rec["w"], rec["h"]
                            x = (cx - bw / 2) * w
                            y = (cy - bh / 2) * h
                            bw_px = bw * w
                            bh_px = bh * h
                            ann = {"id": ann_id, "image_id": i + 1, "category_id": rec["cid"],
                                   "bbox": [round(x, 1), round(y, 1), round(bw_px, 1), round(bh_px, 1)],
                                   "area": round(bw_px * bh_px, 1), "iscrowd": 0}
                            # Preserve polygon as COCO segmentation (absolute px) (DATA-05).
                            if rec["polygon"] is not None:
                                seg = []
                                for j in range(0, len(rec["polygon"]), 2):
                                    seg.append(round(rec["polygon"][j] * w, 1))
                                    seg.append(round(rec["polygon"][j + 1] * h, 1))
                                ann["segmentation"] = [seg]
                            coco["annotations"].append(ann)
                            ann_id += 1
                            cats_seen.add(rec["cid"])
                for c in sorted(cats_seen):
                    coco["categories"].append({"id": c, "name": str(c)})
                with open(os.path.join(output_dir, "annotations.json"), "w") as f:
                    _json.dump(coco, f, indent=2)
                converter_state["results"] = {"images": len(coco["images"]), "annotations": ann_id - 1,
                                              "missing_image": missing_image, "skipped_no_size": skipped_no_size,
                                              "bad_lines": bad_lines}

            elif "COCO" in req.from_fmt and req.to_fmt == "YOLO":
                json_files = glob_module.glob(os.path.join(input_dir, "*.json"))
                if not json_files:
                    converter_state.update(running=False, msg="No JSON files found")
                    return
                with open(json_files[0]) as f:
                    coco = _json.load(f)
                img_map = {img["id"]: img for img in coco.get("images", [])}
                converter_state["total"] = len(coco.get("annotations", []))
                per_image = {}
                skipped_no_size = 0
                for idx, ann in enumerate(coco.get("annotations", [])):
                    converter_state["progress"] = idx + 1
                    iid = ann["image_id"]
                    img_info = img_map.get(iid, {})
                    w, h = img_info.get("width"), img_info.get("height")
                    # COCO images carry explicit dims; skip rather than assume 640 (DATA-06).
                    if not w or not h:
                        skipped_no_size += 1
                        continue
                    bx, by, bw, bh = ann["bbox"]
                    cx = (bx + bw / 2) / w
                    cy = (by + bh / 2) / h
                    nw = bw / w
                    nh = bh / h
                    per_image.setdefault(iid, []).append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                for iid, lines in per_image.items():
                    img_info = img_map.get(iid, {})
                    stem = os.path.splitext(img_info.get("file_name", str(iid)))[0]
                    with open(os.path.join(output_dir, stem + ".txt"), "w") as f:
                        f.write("\n".join(lines) + "\n")
                converter_state["results"] = {"images": len(per_image),
                                              "labels": sum(len(v) for v in per_image.values()),
                                              "skipped_no_size": skipped_no_size}

            elif req.from_fmt == "YOLO" and "VOC" in req.to_fmt:
                from xml.etree.ElementTree import Element, SubElement, tostring
                from xml.dom.minidom import parseString
                count = 0
                missing_image = 0
                skipped_no_size = 0
                bad_lines = 0
                files_written = 0
                for i, txt in enumerate(label_files):
                    if not converter_state["running"]:
                        converter_state.update(msg="Cancelled")  # DATA-11
                        return
                    stem = os.path.splitext(os.path.basename(txt))[0]
                    img_path = _find_image_for_stem(input_dir, stem)
                    dims = _image_dims(img_path)
                    converter_state["progress"] = i + 1
                    # VOC bndbox is absolute px — undeterminable dims would corrupt it (DATA-06).
                    if dims is None:
                        if img_path is None:
                            missing_image += 1
                        else:
                            skipped_no_size += 1
                        continue
                    w, h = dims
                    root = Element("annotation")
                    SubElement(root, "filename").text = os.path.basename(img_path)
                    sz = SubElement(root, "size")
                    SubElement(sz, "width").text = str(w)
                    SubElement(sz, "height").text = str(h)
                    SubElement(sz, "depth").text = "3"
                    with open(txt) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            try:
                                rec = _parse_yolo_label_line(parts)
                            except ValueError:
                                bad_lines += 1
                                continue
                            if rec is None:
                                continue
                            cx, cy, bw, bh = rec["cx"], rec["cy"], rec["w"], rec["h"]
                            obj = SubElement(root, "object")
                            SubElement(obj, "name").text = str(rec["cid"])
                            bnd = SubElement(obj, "bndbox")
                            SubElement(bnd, "xmin").text = str(int((cx - bw / 2) * w))
                            SubElement(bnd, "ymin").text = str(int((cy - bh / 2) * h))
                            SubElement(bnd, "xmax").text = str(int((cx + bw / 2) * w))
                            SubElement(bnd, "ymax").text = str(int((cy + bh / 2) * h))
                            count += 1
                    xml_str = parseString(tostring(root)).toprettyxml(indent="  ")
                    with open(os.path.join(output_dir, stem + ".xml"), "w") as f:
                        f.write(xml_str)
                    files_written += 1
                converter_state["results"] = {"files": files_written, "objects": count,
                                              "missing_image": missing_image, "skipped_no_size": skipped_no_size,
                                              "bad_lines": bad_lines}
            else:
                converter_state["results"] = {"error": f"Unsupported: {req.from_fmt} → {req.to_fmt}"}

            converter_state.update(running=False, msg="Complete")
        except Exception as e:
            converter_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/data/converter/status")
async def converter_status():
    return converter_state.snapshot() if hasattr(converter_state, 'snapshot') else dict(converter_state)


# ── Data: Remapper API ─────────────────────────────────
class RemapperRequest(BaseModel):
    label_dir: str
    output_dir: str
    mapping: dict = {}  # {"old_id": "new_id"}
    auto_reindex: bool = True
    recursive: bool = False

@router.post("/api/data/remapper")
async def data_remapper(req: RemapperRequest):
    try:
        label_dir = safe_label_dir(req.label_dir)
        output_dir = safe_path(req.output_dir, must_be_dir=False)
    except UnsafePathError as e:
        return {"error": e.code}
    if remapper_state["running"]:
        return {"error": "Already running"}
    remapper_state.update(running=True, progress=0, total=0, msg="Remapping...", results={})

    def _run():
        try:
            os.makedirs(output_dir, exist_ok=True)
            label_files = sorted(glob_module.glob(os.path.join(label_dir, "*.txt")))
            if req.recursive:
                label_files += sorted(glob_module.glob(os.path.join(label_dir, "**", "*.txt"), recursive=True))
                label_files = list(dict.fromkeys(label_files))
            if not label_files:
                remapper_state.update(running=False, msg="No label files found")
                return
            mapping = {int(k): int(v) for k, v in req.mapping.items()} if req.mapping else {}
            remapper_state["total"] = len(label_files)
            bad_lines = 0

            def _apply_explicit(cid: int) -> Optional[int]:
                """Apply the explicit mapping. None drops the label (unmapped)."""
                if not mapping:
                    return cid
                return mapping.get(cid)  # unmapped class -> None -> dropped

            # DATA-04: when auto_reindex is on, scan the post-mapping class ids first so
            # the output is compacted to a contiguous 0..N-1 range (downstream nc=).
            reindex: dict[int, int] = {}
            if req.auto_reindex:
                present: set[int] = set()
                for txt in label_files:
                    with open(txt) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            try:
                                cid = int(parts[0])
                            except ValueError:
                                continue
                            mapped = _apply_explicit(cid)
                            if mapped is not None:
                                present.add(mapped)
                reindex = {old: new for new, old in enumerate(sorted(present))}

            count = 0
            for idx, txt in enumerate(label_files):
                if not remapper_state["running"]:
                    remapper_state.update(msg="Cancelled")  # DATA-11
                    return
                lines = []
                with open(txt) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        try:
                            cid = int(parts[0])
                        except ValueError:
                            bad_lines += 1  # tolerate malformed lines (DATA-13)
                            continue
                        cid = _apply_explicit(cid)
                        if cid is None:
                            continue
                        if reindex:
                            cid = reindex[cid]
                        lines.append(f"{cid} {' '.join(parts[1:])}")
                        count += 1
                with open(os.path.join(output_dir, os.path.basename(txt)), "w") as f:
                    f.write("\n".join(lines) + "\n" if lines else "")
                remapper_state["progress"] = idx + 1
            results = {"files": len(label_files), "labels": count, "bad_lines": bad_lines}
            if reindex:
                # Expose the reindex map (old->new) so the UI can show the new id space (DATA-04).
                results["reindex_map"] = {str(k): v for k, v in reindex.items()}
            remapper_state["results"] = results
            remapper_state.update(running=False, msg="Complete")
        except Exception as e:
            remapper_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/data/remapper/status")
async def remapper_status():
    return remapper_state.snapshot() if hasattr(remapper_state, 'snapshot') else dict(remapper_state)


# ── Data: Merger API ───────────────────────────────────
class MergerRequest(BaseModel):
    datasets: list[str]
    output_dir: str
    dhash_threshold: int = 10
    recursive: bool = False

@router.post("/api/data/merger")
async def data_merger(req: MergerRequest):
    try:
        datasets = [safe_image_dir(d) for d in req.datasets]
        output_dir = safe_path(req.output_dir, must_be_dir=False)
    except UnsafePathError as e:
        return {"error": e.code}
    if not datasets:
        return {"error": "EMPTY"}
    if merger_state["running"]:
        return {"error": "Already running"}
    merger_state.update(running=True, progress=0, total=0, msg="Merging...", results={})

    def _run():
        try:
            import shutil
            from core.hashing import compute_dhash, cluster_near_duplicates

            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
            all_imgs = []
            for d in datasets:
                all_imgs.extend(glob_images(d, recursive=req.recursive))
            merger_state["total"] = len(all_imgs)

            # Phase 1: hash every readable image (DATA-08 — BK-tree clustering
            # replaces the old O(N²) incremental Hamming scan).
            merger_state["msg"] = "Hashing..."
            valid_imgs: list[str] = []
            hashes: list[int] = []
            for i, fp in enumerate(all_imgs):
                if not merger_state["running"]:
                    merger_state.update(msg="Cancelled")  # DATA-11
                    return
                h = compute_dhash(fp)
                if h is not None:
                    valid_imgs.append(fp)
                    hashes.append(h)
                merger_state["progress"] = i + 1

            # Group ids per index; keep the first member of each cluster.
            group_ids = cluster_near_duplicates(hashes, req.dhash_threshold)
            merger_state.update(msg="Copying...", progress=0, total=len(valid_imgs))
            seen_groups: set[int] = set()
            copied = 0
            dupes = 0
            labels_copied = 0
            labels_missing = 0
            for i, fp in enumerate(valid_imgs):
                if not merger_state["running"]:
                    merger_state.update(msg="Cancelled")  # DATA-11
                    return
                merger_state["progress"] = i + 1
                gid = group_ids[i]
                if gid in seen_groups:
                    dupes += 1
                    continue
                seen_groups.add(gid)
                dst = os.path.join(output_dir, "images", os.path.basename(fp))
                if os.path.exists(dst):
                    name, ext = os.path.splitext(os.path.basename(fp))
                    dst = os.path.join(output_dir, "images", f"{name}_{i}{ext}")
                shutil.copy2(fp, dst)
                # Copy label if found; track presence so the user sees coverage (DATA-08).
                stem = os.path.splitext(os.path.basename(fp))[0]
                parent = os.path.dirname(fp)
                found_label = False
                for lbl_dir_name in ["labels", "../labels", "."]:
                    txt = os.path.join(parent, lbl_dir_name, stem + ".txt")
                    if os.path.isfile(txt):
                        shutil.copy2(txt, os.path.join(output_dir, "labels", os.path.basename(dst).rsplit(".", 1)[0] + ".txt"))
                        found_label = True
                        break
                if found_label:
                    labels_copied += 1
                else:
                    labels_missing += 1
                copied += 1
            merger_state["results"] = {"total": len(all_imgs), "copied": copied, "duplicates": dupes,
                                       "labels_copied": labels_copied, "labels_missing": labels_missing}
            merger_state.update(running=False, msg="Complete")
        except Exception as e:
            merger_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/data/merger/status")
async def merger_status():
    return merger_state.snapshot() if hasattr(merger_state, 'snapshot') else dict(merger_state)


# ── Data: Sampler API ──────────────────────────────────
class SamplerRequest(BaseModel):
    img_dir: str
    label_dir: str = ""
    output_dir: str
    strategy: str = "Random"
    target_count: int = 500
    seed: int = 42
    include_labels: bool = True
    recursive: bool = False

def _farthest_point_sample(candidates, features, n):
    """Select n items from candidates maximizing diversity via farthest-point sampling."""
    import numpy as np, random as _rnd
    if len(candidates) <= n:
        return list(candidates)
    feat = np.array([features[c] for c in candidates])
    selected = [_rnd.randrange(len(candidates))]
    dists = np.full(len(candidates), np.inf)
    for _ in range(n - 1):
        last = feat[selected[-1]]
        d = np.sum((feat - last) ** 2, axis=1)
        dists = np.minimum(dists, d)
        dists[selected] = -1
        selected.append(int(np.argmax(dists)))
    return [candidates[i] for i in selected]

@router.post("/api/data/sampler")
async def data_sampler(req: SamplerRequest):
    try:
        img_dir = safe_image_dir(req.img_dir)
        label_dir = safe_label_dir(req.label_dir) if req.label_dir else ""
        output_dir = safe_path(req.output_dir, must_be_dir=False)
    except UnsafePathError as e:
        return {"error": e.code}
    if sampler_state["running"]:
        return {"error": "Already running"}
    sampler_state.update(running=True, progress=0, total=0, msg="Scanning...", results={})

    def _run():
        try:
            import random, shutil
            import numpy as np
            random.seed(req.seed)
            np.random.seed(req.seed)
            imgs = glob_images(img_dir, recursive=req.recursive)
            if not imgs:
                sampler_state.update(running=False, msg="No images found")
                return

            lbl_dir = label_dir or img_dir
            # Parse labels: class→images, image→bbox features.
            class_images = {}  # {cid: set of img paths}
            img_features = {}  # {img_path: mean bbox center [cx, cy]}
            for fp in imgs:
                stem = os.path.splitext(os.path.basename(fp))[0]
                txt = os.path.join(lbl_dir, stem + ".txt")
                centers = []
                classes = set()
                if os.path.isfile(txt):
                    with open(txt) as f:
                        for line in f:
                            p = line.strip().split()
                            if len(p) < 5:
                                continue
                            try:
                                classes.add(int(p[0]))
                                centers.append([float(p[1]), float(p[2])])
                            except ValueError:
                                continue  # tolerate malformed lines (DATA-13)
                for c in classes:
                    class_images.setdefault(c, set()).add(fp)
                # CAVEAT: the only per-image feature is the mean bbox *center*; box-less
                # images all collapse to [0.5, 0.5], so the 'balanced' diversity sampling
                # is degenerate on unlabeled / classification-style data.
                img_features[fp] = np.mean(centers, axis=0).tolist() if centers else [0.5, 0.5]

            # Clamp target to the pool — we can never select more images than exist.
            target = min(req.target_count, len(imgs))
            selected = set()
            strategy = req.strategy.lower()
            if strategy == "random":
                selected = set(random.sample(imgs, target))
            elif strategy == "stratified":
                total_class_associations = sum(len(v) for v in class_images.values())
                for cid, cimgs in class_images.items():
                    n = max(1, int(target * len(cimgs) / max(total_class_associations, 1)))
                    selected.update(random.sample(list(cimgs), min(n, len(cimgs))))
            elif strategy == "balanced":
                if not class_images:
                    selected = set(random.sample(imgs, target))
                else:
                    per_class = max(1, target // len(class_images))
                    for cid, cimgs in class_images.items():
                        pool = list(cimgs)
                        if len(pool) <= per_class:
                            selected.update(pool)
                        else:
                            selected.update(_farthest_point_sample(pool, img_features, per_class))

            # DATA-09: trim/top-up to EXACTLY target (when target <= pool). Per-class
            # rounding routinely over- or under-shoots; correct it deterministically.
            selected = list(selected)
            if len(selected) > target:
                selected = random.sample(selected, target)
            elif len(selected) < target:
                remaining = [f for f in imgs if f not in set(selected)]
                if remaining:
                    selected.extend(random.sample(remaining, min(target - len(selected), len(remaining))))

            sampler_state["total"] = len(selected)
            sampler_state["msg"] = "Copying..."
            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
            if req.include_labels:
                os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
            before_classes = {c: len(v) for c, v in class_images.items()}
            after_classes = {}
            for i, fp in enumerate(selected):
                if not sampler_state["running"]:
                    sampler_state.update(msg="Cancelled")  # DATA-11
                    return
                shutil.copy2(fp, os.path.join(output_dir, "images"))
                if req.include_labels:
                    stem = os.path.splitext(os.path.basename(fp))[0]
                    txt = os.path.join(lbl_dir, stem + ".txt")
                    if os.path.isfile(txt):
                        shutil.copy2(txt, os.path.join(output_dir, "labels"))
                for c, cimgs in class_images.items():
                    if fp in cimgs:
                        after_classes[c] = after_classes.get(c, 0) + 1
                sampler_state["progress"] = i + 1
            sampler_state["results"] = {"total": len(imgs), "sampled": len(selected),
                                         "target": target,
                                         "before": before_classes, "after": after_classes}
            sampler_state.update(running=False, msg="Complete")
        except Exception as e:
            sampler_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/data/sampler/status")
async def sampler_status():
    return sampler_state.snapshot() if hasattr(sampler_state, 'snapshot') else dict(sampler_state)


