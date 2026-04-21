"""/api/data/* 라우터."""
import os
import random
import shutil
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from server.state import explorer_state, splitter_state, converter_state, remapper_state, merger_state, sampler_state, executor
from server.utils import imread, glob_images, encode_jpeg

router = APIRouter()

# ── Data: Explorer API ──────────────────────────────────
explorer_state = {"running": False, "progress": 0, "total": 0, "msg": "", "data": None}

@router.post("/api/data/explorer")
async def data_explorer(req: dict):
    img_dir = req.get("img_dir", "")
    lbl_dir = req.get("label_dir", "")
    if not img_dir or not os.path.isdir(img_dir):
        return {"error": "Invalid image directory"}
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
            for i, fp in enumerate(imgs[:5000]):
                explorer_state["progress"] = i + 1
                stem = os.path.splitext(os.path.basename(fp))[0]
                txt = os.path.join(lbl_dir, stem + ".txt") if lbl_dir else ""
                boxes = []
                box_details = []
                if txt and os.path.isfile(txt):
                    with open(txt) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cid = int(parts[0])
                                boxes.append(cid)
                                class_counts[cid] = class_counts.get(cid, 0) + 1
                                if cid not in img_class_counts:
                                    img_class_counts[cid] = set()
                                img_class_counts[cid].add(i)
                                bw, bh = float(parts[3]), float(parts[4])
                                box_sizes.append({"w": bw, "h": bh})
                                box_details.append({"cid": cid, "cx": float(parts[1]), "cy": float(parts[2]), "w": bw, "h": bh})
                                box_aspect_ratios.append(round(bw / max(bh, 1e-6), 2))
                # image aspect ratio
                try:
                    im = imread(fp)
                    if im is not None:
                        h_, w_ = im.shape[:2]
                        aspect_ratios.append(round(w_ / max(h_, 1), 2))
                except:
                    pass
                file_info.append({"name": os.path.basename(fp), "path": fp, "boxes": len(boxes),
                                  "classes": list(set(boxes)), "box_details": box_details})
            img_class_count_dict = {k: len(v) for k, v in img_class_counts.items()}
            explorer_state["data"] = {
                "total": n, "shown": len(file_info), "files": file_info,
                "class_counts": class_counts, "img_class_counts": img_class_count_dict,
                "box_sizes": box_sizes, "aspect_ratios": aspect_ratios,
                "box_aspect_ratios": box_aspect_ratios
            }
            explorer_state.update(running=False, msg="Complete")
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

@router.post("/api/data/explorer/preview")
async def explorer_preview(req: dict):
    """Return base64 JPEG of image with bounding boxes overlaid."""
    img_path = req.get("img_path", "")
    lbl_dir = req.get("label_dir", "")
    if not img_path or not os.path.isfile(img_path):
        return {"error": "Image not found"}
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
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    boxes.append({"cid": cid, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    total_cls = max((b["cid"] for b in boxes), default=0) + 1
    palette = _generate_palette(max(total_cls, 1))
    for b in boxes:
        color = palette[b["cid"] % len(palette)]
        cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, 2)
        label = str(b["cid"])
        _draw_label(img, label, b["x1"], b["y1"], color, 0.6, 1, True)
    return {"image": encode_jpeg(img, 90), "width": w, "height": h, "box_count": len(boxes)}


# ── Data: Splitter API ─────────────────────────────────
class SplitterRequest(BaseModel):
    img_dir: str
    label_dir: str = ""
    output_dir: str
    train: float = 0.7
    val: float = 0.2
    test: float = 0.1
    strategy: str = "random"  # random, stratified, similarity

splitter_state = {"running": False, "msg": "", "progress": 0, "total": 0, "results": {}}

@router.post("/api/data/splitter")
async def data_splitter(req: SplitterRequest):
    if splitter_state["running"]:
        return {"error": "Already running"}
    splitter_state.update(running=True, msg="Splitting...", progress=0, total=0, results={})

    def _run():
        try:
            import random, shutil
            imgs = glob_images(req.img_dir)
            if not imgs:
                splitter_state.update(running=False, msg="No images found")
                return
            n = len(imgs)
            splitter_state["total"] = n

            # Normalize ratios — treat 0 as empty split
            ratios = {"train": max(req.train, 0), "val": max(req.val, 0), "test": max(req.test, 0)}
            total_ratio = sum(ratios.values())
            if total_ratio <= 0:
                splitter_state.update(running=False, msg="Error: All ratios are 0")
                return

            if req.strategy == "stratified" and req.label_dir:
                # Group images by class set
                class_groups = {}
                for fp in imgs:
                    stem = os.path.splitext(os.path.basename(fp))[0]
                    txt = os.path.join(req.label_dir, stem + ".txt")
                    classes = set()
                    if os.path.isfile(txt):
                        with open(txt) as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    classes.add(int(parts[0]))
                    key = tuple(sorted(classes)) if classes else (-1,)
                    class_groups.setdefault(key, []).append(fp)
                splits = {"train": [], "val": [], "test": []}
                # Accumulate fractional counts across groups to preserve global ratio
                frac_train = 0.0
                frac_val = 0.0
                norm_train = ratios["train"] / total_ratio
                norm_val = ratios["val"] / total_ratio
                for group_imgs in class_groups.values():
                    random.shuffle(group_imgs)
                    gn = len(group_imgs)
                    frac_train += gn * norm_train
                    frac_val += gn * norm_val
                    n_train = round(frac_train)
                    n_val = round(frac_val)
                    # Clamp so we don't exceed group size
                    n_train = min(n_train, gn)
                    n_val = min(n_val, gn - n_train)
                    splits["train"].extend(group_imgs[:n_train])
                    splits["val"].extend(group_imgs[n_train:n_train + n_val])
                    splits["test"].extend(group_imgs[n_train + n_val:])
                    frac_train -= n_train
                    frac_val -= n_val
            else:
                # Random split
                random.shuffle(imgs)
                n_train = round(n * ratios["train"] / total_ratio)
                n_val = round(n * ratios["val"] / total_ratio)
                # Clamp to avoid exceeding total
                n_val = min(n_val, n - n_train)
                splits = {"train": imgs[:n_train], "val": imgs[n_train:n_train + n_val], "test": imgs[n_train + n_val:]}

            # Copy files with progress
            total_files = sum(len(v) for v in splits.values())
            splitter_state["total"] = total_files
            done = 0
            for split_name, split_files in splits.items():
                if not split_files:
                    continue
                img_out = os.path.join(req.output_dir, split_name, "images")
                lbl_out = os.path.join(req.output_dir, split_name, "labels")
                os.makedirs(img_out, exist_ok=True)
                os.makedirs(lbl_out, exist_ok=True)
                for fp in split_files:
                    shutil.copy2(fp, img_out)
                    stem = os.path.splitext(os.path.basename(fp))[0]
                    txt = os.path.join(req.label_dir, stem + ".txt")
                    if os.path.isfile(txt):
                        shutil.copy2(txt, lbl_out)
                    done += 1
                    splitter_state["progress"] = done
            splitter_state["results"] = {k: len(v) for k, v in splits.items()}
            splitter_state.update(running=False, msg="Complete")
        except Exception as e:
            splitter_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/data/splitter/status")
async def splitter_status():
    return dict(splitter_state)


# ── Data: Converter API ────────────────────────────────
class ConverterRequest(BaseModel):
    input_dir: str
    output_dir: str
    from_fmt: str = "YOLO"
    to_fmt: str = "COCO JSON"

converter_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": {}}

@router.post("/api/data/converter")
async def data_converter(req: ConverterRequest):
    if converter_state["running"]:
        return {"error": "Already running"}
    converter_state.update(running=True, progress=0, total=0, msg="Converting...", results={})

    def _run():
        try:
            import json as _json
            os.makedirs(req.output_dir, exist_ok=True)
            label_files = sorted(glob_module.glob(os.path.join(req.input_dir, "*.txt")))
            img_files = glob_images(req.input_dir)
            converter_state["total"] = len(label_files) or len(img_files)

            if req.from_fmt == "YOLO" and "COCO" in req.to_fmt:
                coco = {"images": [], "annotations": [], "categories": []}
                ann_id = 1
                cats_seen = set()
                for i, txt in enumerate(label_files):
                    stem = os.path.splitext(os.path.basename(txt))[0]
                    # find matching image
                    img_path = None
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        candidate = os.path.join(req.input_dir, stem + ext)
                        if os.path.isfile(candidate):
                            img_path = candidate
                            break
                    w, h = 640, 640
                    if img_path:
                        img = imread(img_path)
                        if img is not None:
                            h, w = img.shape[:2]
                    coco["images"].append({"id": i+1, "file_name": stem + (os.path.splitext(img_path)[1] if img_path else ".jpg"), "width": w, "height": h})
                    with open(txt) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cid = int(parts[0])
                                cx, cy, bw, bh = map(float, parts[1:5])
                                x = (cx - bw/2) * w
                                y = (cy - bh/2) * h
                                bw_px = bw * w
                                bh_px = bh * h
                                coco["annotations"].append({"id": ann_id, "image_id": i+1, "category_id": cid, "bbox": [round(x,1), round(y,1), round(bw_px,1), round(bh_px,1)], "area": round(bw_px*bh_px,1), "iscrowd": 0})
                                ann_id += 1
                                cats_seen.add(cid)
                    converter_state["progress"] = i + 1
                for c in sorted(cats_seen):
                    coco["categories"].append({"id": c, "name": str(c)})
                with open(os.path.join(req.output_dir, "annotations.json"), "w") as f:
                    _json.dump(coco, f, indent=2)
                converter_state["results"] = {"images": len(coco["images"]), "annotations": ann_id - 1}

            elif "COCO" in req.from_fmt and req.to_fmt == "YOLO":
                json_files = glob_module.glob(os.path.join(req.input_dir, "*.json"))
                if not json_files:
                    converter_state.update(running=False, msg="No JSON files found")
                    return
                with open(json_files[0]) as f:
                    coco = _json.load(f)
                img_map = {img["id"]: img for img in coco.get("images", [])}
                converter_state["total"] = len(coco.get("annotations", []))
                per_image = {}
                for idx, ann in enumerate(coco.get("annotations", [])):
                    iid = ann["image_id"]
                    if iid not in per_image:
                        per_image[iid] = []
                    img_info = img_map.get(iid, {})
                    w, h = img_info.get("width", 640), img_info.get("height", 640)
                    bx, by, bw, bh = ann["bbox"]
                    cx = (bx + bw/2) / w
                    cy = (by + bh/2) / h
                    nw = bw / w
                    nh = bh / h
                    per_image[iid].append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                    converter_state["progress"] = idx + 1
                for iid, lines in per_image.items():
                    img_info = img_map.get(iid, {})
                    stem = os.path.splitext(img_info.get("file_name", str(iid)))[0]
                    with open(os.path.join(req.output_dir, stem + ".txt"), "w") as f:
                        f.write("\n".join(lines) + "\n")
                converter_state["results"] = {"images": len(per_image), "labels": sum(len(v) for v in per_image.values())}

            elif req.from_fmt == "YOLO" and "VOC" in req.to_fmt:
                from xml.etree.ElementTree import Element, SubElement, tostring
                from xml.dom.minidom import parseString
                count = 0
                for i, txt in enumerate(label_files):
                    stem = os.path.splitext(os.path.basename(txt))[0]
                    img_path = None
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        candidate = os.path.join(req.input_dir, stem + ext)
                        if os.path.isfile(candidate):
                            img_path = candidate
                            break
                    w, h = 640, 640
                    if img_path:
                        img = imread(img_path)
                        if img is not None:
                            h, w = img.shape[:2]
                    root = Element("annotation")
                    SubElement(root, "filename").text = stem + (os.path.splitext(img_path)[1] if img_path else ".jpg")
                    sz = SubElement(root, "size")
                    SubElement(sz, "width").text = str(w)
                    SubElement(sz, "height").text = str(h)
                    SubElement(sz, "depth").text = "3"
                    with open(txt) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cid = int(parts[0])
                                cx, cy, bw, bh = map(float, parts[1:5])
                                obj = SubElement(root, "object")
                                SubElement(obj, "name").text = str(cid)
                                bnd = SubElement(obj, "bndbox")
                                SubElement(bnd, "xmin").text = str(int((cx - bw/2) * w))
                                SubElement(bnd, "ymin").text = str(int((cy - bh/2) * h))
                                SubElement(bnd, "xmax").text = str(int((cx + bw/2) * w))
                                SubElement(bnd, "ymax").text = str(int((cy + bh/2) * h))
                                count += 1
                    xml_str = parseString(tostring(root)).toprettyxml(indent="  ")
                    with open(os.path.join(req.output_dir, stem + ".xml"), "w") as f:
                        f.write(xml_str)
                    converter_state["progress"] = i + 1
                converter_state["results"] = {"files": len(label_files), "objects": count}
            else:
                converter_state["results"] = {"error": f"Unsupported: {req.from_fmt} → {req.to_fmt}"}

            converter_state.update(running=False, msg="Complete")
        except Exception as e:
            converter_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/data/converter/status")
async def converter_status():
    return dict(converter_state)


# ── Data: Remapper API ─────────────────────────────────
class RemapperRequest(BaseModel):
    label_dir: str
    output_dir: str
    mapping: dict = {}  # {"old_id": "new_id"}
    auto_reindex: bool = True
    recursive: bool = False

remapper_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": {}}

@router.post("/api/data/remapper")
async def data_remapper(req: RemapperRequest):
    if remapper_state["running"]:
        return {"error": "Already running"}
    remapper_state.update(running=True, progress=0, total=0, msg="Remapping...", results={})

    def _run():
        try:
            os.makedirs(req.output_dir, exist_ok=True)
            label_files = sorted(glob_module.glob(os.path.join(req.label_dir, "*.txt")))
            if req.recursive:
                label_files += sorted(glob_module.glob(os.path.join(req.label_dir, "**", "*.txt"), recursive=True))
                label_files = list(dict.fromkeys(label_files))
            if not label_files:
                remapper_state.update(running=False, msg="No label files found")
                return
            mapping = {int(k): int(v) for k, v in req.mapping.items()} if req.mapping else {}
            remapper_state["total"] = len(label_files)
            count = 0
            for idx, txt in enumerate(label_files):
                lines = []
                with open(txt) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            if mapping:
                                if cid in mapping:
                                    cid = mapping[cid]
                                else:
                                    continue
                            lines.append(f"{cid} {' '.join(parts[1:])}")
                            count += 1
                with open(os.path.join(req.output_dir, os.path.basename(txt)), "w") as f:
                    f.write("\n".join(lines) + "\n" if lines else "")
                remapper_state["progress"] = idx + 1
            remapper_state["results"] = {"files": len(label_files), "labels": count}
            remapper_state.update(running=False, msg="Complete")
        except Exception as e:
            remapper_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/data/remapper/status")
async def remapper_status():
    return dict(remapper_state)


# ── Data: Merger API ───────────────────────────────────
class MergerRequest(BaseModel):
    datasets: list[str]
    output_dir: str
    dhash_threshold: int = 10
    recursive: bool = False

merger_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": {}}

@router.post("/api/data/merger")
async def data_merger(req: MergerRequest):
    if merger_state["running"]:
        return {"error": "Already running"}
    merger_state.update(running=True, progress=0, total=0, msg="Merging...", results={})

    def _run():
        try:
            import shutil
            import cv2
            import numpy as np

            def compute_dhash(path: str) -> int | None:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return None
                resized = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
                diff = resized[:, 1:] > resized[:, :-1]
                bits = diff.flatten()
                h = 0
                for b in bits:
                    h = (h << 1) | int(b)
                return h

            def hamming(a: int, b: int) -> int:
                return bin(a ^ b).count("1")

            os.makedirs(os.path.join(req.output_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(req.output_dir, "labels"), exist_ok=True)
            all_imgs = []
            for d in req.datasets:
                all_imgs.extend(glob_images(d, recursive=req.recursive))
            merger_state["total"] = len(all_imgs)
            seen_hashes: list[int] = []
            copied = 0
            dupes = 0
            for i, fp in enumerate(all_imgs):
                h = compute_dhash(fp)
                if h is None:
                    merger_state["progress"] = i + 1
                    continue
                is_dupe = any(hamming(h, s) <= req.dhash_threshold for s in seen_hashes)
                if is_dupe:
                    dupes += 1
                    merger_state["progress"] = i + 1
                    continue
                seen_hashes.append(h)
                dst = os.path.join(req.output_dir, "images", os.path.basename(fp))
                if os.path.exists(dst):
                    name, ext = os.path.splitext(os.path.basename(fp))
                    dst = os.path.join(req.output_dir, "images", f"{name}_{i}{ext}")
                shutil.copy2(fp, dst)
                # Copy label if exists
                stem = os.path.splitext(os.path.basename(fp))[0]
                parent = os.path.dirname(fp)
                for lbl_dir_name in ["labels", "../labels", "."]:
                    txt = os.path.join(parent, lbl_dir_name, stem + ".txt")
                    if os.path.isfile(txt):
                        shutil.copy2(txt, os.path.join(req.output_dir, "labels", os.path.basename(dst).rsplit(".", 1)[0] + ".txt"))
                        break
                copied += 1
                merger_state["progress"] = i + 1
            merger_state["results"] = {"total": len(all_imgs), "copied": copied, "duplicates": dupes}
            merger_state.update(running=False, msg="Complete")
        except Exception as e:
            merger_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/data/merger/status")
async def merger_status():
    return dict(merger_state)


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

sampler_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": {}}

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
    if sampler_state["running"]:
        return {"error": "Already running"}
    sampler_state.update(running=True, progress=0, total=0, msg="Scanning...", results={})

    def _run():
        try:
            import random, shutil
            import numpy as np
            random.seed(req.seed)
            np.random.seed(req.seed)
            imgs = glob_images(req.img_dir, recursive=req.recursive)
            if not imgs:
                sampler_state.update(running=False, msg="No images found")
                return

            lbl_dir = req.label_dir or req.img_dir
            # Parse labels: class→images, image→bbox features
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
                            if len(p) >= 5:
                                classes.add(int(p[0]))
                                centers.append([float(p[1]), float(p[2])])
                for c in classes:
                    class_images.setdefault(c, set()).add(fp)
                img_features[fp] = np.mean(centers, axis=0).tolist() if centers else [0.5, 0.5]

            selected = set()
            strategy = req.strategy.lower()
            if strategy == "random":
                selected = set(random.sample(imgs, min(req.target_count, len(imgs))))
            elif strategy == "stratified":
                total_assoc = sum(len(v) for v in class_images.values())
                for cid, cimgs in class_images.items():
                    n = max(1, int(req.target_count * len(cimgs) / max(total_assoc, 1)))
                    selected.update(random.sample(list(cimgs), min(n, len(cimgs))))
                remaining = [f for f in imgs if f not in selected]
                need = req.target_count - len(selected)
                if need > 0 and remaining:
                    selected.update(random.sample(remaining, min(need, len(remaining))))
            elif strategy == "balanced":
                if not class_images:
                    selected = set(random.sample(imgs, min(req.target_count, len(imgs))))
                else:
                    per_class = max(1, req.target_count // len(class_images))
                    for cid, cimgs in class_images.items():
                        pool = list(cimgs)
                        if len(pool) <= per_class:
                            selected.update(pool)
                        else:
                            picked = _farthest_point_sample(pool, img_features, per_class)
                            selected.update(picked)

            selected = list(selected)
            sampler_state["total"] = len(selected)
            sampler_state["msg"] = "Copying..."
            os.makedirs(os.path.join(req.output_dir, "images"), exist_ok=True)
            if req.include_labels:
                os.makedirs(os.path.join(req.output_dir, "labels"), exist_ok=True)
            before_classes = {c: len(v) for c, v in class_images.items()}
            after_classes = {}
            for i, fp in enumerate(selected):
                shutil.copy2(fp, os.path.join(req.output_dir, "images"))
                if req.include_labels and lbl_dir:
                    stem = os.path.splitext(os.path.basename(fp))[0]
                    txt = os.path.join(lbl_dir, stem + ".txt")
                    if os.path.isfile(txt):
                        shutil.copy2(txt, os.path.join(req.output_dir, "labels"))
                for c, cimgs in class_images.items():
                    if fp in cimgs:
                        after_classes[c] = after_classes.get(c, 0) + 1
                sampler_state["progress"] = i + 1
            sampler_state["results"] = {"total": len(imgs), "sampled": len(selected),
                                         "before": before_classes, "after": after_classes}
            sampler_state.update(running=False, msg="Complete")
        except Exception as e:
            sampler_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/data/sampler/status")
async def sampler_status():
    return dict(sampler_state)


