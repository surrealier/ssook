"""/api/quality/* 라우터."""
import os

import cv2
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from server.errors import route_errors
from server.path_safety import safe_image_dir
from server.state import anomaly_state, quality_state, dup_state, leaky_state, sim_state, executor
from server.utils import imread, glob_images

router = APIRouter()


# ── Shared Pydantic shapes ──────────────────────────────
class QualityDirRequest(BaseModel):
    img_dir: str
    label_dir: Optional[str] = ""
    recursive: bool = False
    threshold: Optional[int] = None
    metrics: Optional[list[str]] = None


class LeakyRequest(BaseModel):
    train_dir: Optional[str] = ""
    val_dir: Optional[str] = ""
    test_dir: Optional[str] = ""
    threshold: int = 0


class SimRequest(BaseModel):
    img_dir: str
    query_path: Optional[str] = ""
    top_k: int = 10
    recursive: bool = False

# ── Quality: Anomaly Detector API ──────────────────────
# NOTE: All *_state instances are imported from server.state — do NOT re-declare.
# Re-declaring breaks /api/force-stop because all_states[...] would still point
# to the original TaskState while routes mutate a different dict.

@router.post("/api/quality/anomaly")
async def quality_anomaly(req: QualityDirRequest):
    if anomaly_state["running"]:
        return {"error": "Already running"}
    img_dir = safe_image_dir(req.img_dir)
    label_dir = req.label_dir
    recursive = req.recursive
    anomaly_state.update(running=True, progress=0, total=0, msg="Scanning...", results=[])

    @route_errors(state=anomaly_state, scope="anomaly")
    def _run():
        imgs = glob_images(img_dir, recursive=recursive)
        anomaly_state["total"] = len(imgs)
        results = []
        for i, fp in enumerate(imgs):
            if not anomaly_state.get("running", True):
                anomaly_state.update(msg="Stopped by user")
                break
            stem = os.path.splitext(os.path.basename(fp))[0]
            txt = os.path.join(label_dir, stem + ".txt") if label_dir else ""
            if txt and os.path.isfile(txt):
                with open(txt) as f:
                    for ln, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cx, cy, bw, bh = map(float, parts[1:5])
                        issues = []
                        if cx - bw/2 < -0.01 or cy - bh/2 < -0.01 or cx + bw/2 > 1.01 or cy + bh/2 > 1.01:
                            issues.append("Out-of-bounds")
                        area = bw * bh
                        if area < 0.0001:
                            issues.append("Tiny box")
                        elif area > 0.9:
                            issues.append("Huge box")
                        ar = bw / (bh + 1e-9)
                        if ar > 20 or ar < 0.05:
                            issues.append("Extreme aspect")
                        if issues:
                            results.append({"file": os.path.basename(fp), "type": ", ".join(issues),
                                            "details": f"L{ln+1}: cls={parts[0]} cx={cx:.3f} cy={cy:.3f} w={bw:.3f} h={bh:.3f}",
                                            "severity": "High" if "Out-of-bounds" in issues else "Medium"})
            anomaly_state["progress"] = i + 1
        anomaly_state["results"] = results[:1000]
        anomaly_state.update(running=False, msg=f"Complete — {len(results)} issues found")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/quality/anomaly/status")
async def anomaly_status():
    return anomaly_state.snapshot() if hasattr(anomaly_state, "snapshot") else dict(anomaly_state)


# ── Quality: Image Quality Checker API ─────────────────

@router.post("/api/quality/image-quality")
async def quality_check(req: QualityDirRequest):
    if quality_state["running"]:
        return {"error": "Already running"}
    img_dir = safe_image_dir(req.img_dir)
    recursive = req.recursive
    quality_state.update(running=True, progress=0, total=0, msg="Checking...", results=[])

    @route_errors(state=quality_state, scope="quality")
    def _run():
        imgs = glob_images(img_dir, recursive=recursive)
        quality_state["total"] = len(imgs)
        results = []
        for i, fp in enumerate(imgs):
            if not quality_state.get("running", True):
                quality_state.update(msg="Stopped by user")
                break
            frame = imread(fp)
            if frame is None:
                quality_state["progress"] = i + 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2)
            brightness = round(float(gray.mean()), 2)
            entropy = round(float(-np.sum(np.histogram(gray, 256, [0,256])[0]/gray.size * np.log2(np.histogram(gray, 256, [0,256])[0]/gray.size + 1e-12))), 2)
            h, w = frame.shape[:2]
            aspect = round(w / h, 2)
            issues = []
            if blur < 50:
                issues.append("Blurry")
            if brightness < 40:
                issues.append("Dark")
            elif brightness > 220:
                issues.append("Overexposed")
            if aspect > 4 or aspect < 0.25:
                issues.append("Odd aspect")
            results.append({"file": os.path.basename(fp), "blur": blur, "brightness": brightness,
                            "entropy": entropy, "aspect": aspect, "issues": ", ".join(issues) or "OK"})
            quality_state["progress"] = i + 1
        quality_state["results"] = results[:1000]
        quality_state.update(running=False, msg=f"Complete — {len(results)} images checked")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/quality/image-quality/status")
async def quality_status():
    return quality_state.snapshot() if hasattr(quality_state, 'snapshot') else dict(quality_state)


# ── Quality: Near-Duplicate Detector API ───────────────
# Use the canonical hash from core.hashing so merger and duplicate detector
# agree on what "duplicate" means.
from core.hashing import compute_dhash as _core_dhash, hamming as _core_hamming


def _dhash(img, size=8):
    return _core_dhash(img, size=size)

@router.post("/api/quality/duplicate")
async def quality_duplicate(req: QualityDirRequest):
    if dup_state["running"]:
        return {"error": "Already running"}
    img_dir = safe_image_dir(req.img_dir)
    threshold = int(req.threshold if req.threshold is not None else 10)
    recursive = req.recursive
    dup_state.update(running=True, progress=0, total=0, msg="Hashing...", results=[])

    @route_errors(state=dup_state, scope="duplicate")
    def _run():
        imgs = glob_images(img_dir, recursive=recursive)
        dup_state["total"] = len(imgs)
        hashes = []
        for i, fp in enumerate(imgs):
            if not dup_state.get("running", True):
                dup_state.update(msg="Stopped by user")
                break
            frame = imread(fp)
            if frame is not None:
                hashes.append((os.path.basename(fp), _dhash(frame)))
            dup_state["progress"] = i + 1
        dup_state["msg"] = "Comparing..."
        results = []
        group = 1
        for i in range(len(hashes)):
            for j in range(i+1, len(hashes)):
                dist = _core_hamming(hashes[i][1], hashes[j][1])
                if dist <= threshold:
                    results.append({"group": group, "image_a": hashes[i][0], "image_b": hashes[j][0], "distance": dist})
                    group += 1
                if len(results) >= 500:
                    break
            if len(results) >= 500:
                break
        dup_state["results"] = results
        dup_state.update(running=False, msg=f"Complete — {len(results)} pairs found")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/quality/duplicate/status")
async def duplicate_status():
    return dup_state.snapshot() if hasattr(dup_state, 'snapshot') else dict(dup_state)


# ── Quality: Leaky Split Detector API ──────────────────

@router.post("/api/quality/leaky")
async def quality_leaky(req: LeakyRequest):
    if leaky_state["running"]:
        return {"error": "Already running"}
    dirs = {k: getattr(req, k) for k in ["train_dir", "val_dir", "test_dir"]}
    # Validate any provided dirs.
    for k, d in list(dirs.items()):
        if d:
            dirs[k] = safe_image_dir(d)
    threshold = int(req.threshold)
    leaky_state.update(running=True, progress=0, total=0, msg="Scanning...", results=[])

    @route_errors(state=leaky_state, scope="leaky")
    def _run():
        split_hashes = {}
        for name, d in dirs.items():
            if not d:
                continue
            imgs = glob_images(d)
            split_hashes[name] = {os.path.basename(fp): _dhash(imread(fp)) for fp in imgs if imread(fp) is not None}
        leaky_state["total"] = sum(len(v) for v in split_hashes.values())
        results = []
        names = list(split_hashes.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                dupes = 0
                files = []
                for fa, ha in split_hashes[names[i]].items():
                    for fb, hb in split_hashes[names[j]].items():
                        if ha is None or hb is None:
                            continue
                        dist = _core_hamming(ha, hb)
                        if dist <= threshold:
                            dupes += 1
                            if len(files) < 10:
                                files.append(f"{fa} ↔ {fb}")
                results.append({"pair": f"{names[i]} ↔ {names[j]}", "duplicates": dupes, "files": "; ".join(files)})
        leaky_state["results"] = results
        leaky_state.update(running=False, msg="Complete")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/quality/leaky/status")
async def leaky_status():
    return leaky_state.snapshot() if hasattr(leaky_state, 'snapshot') else dict(leaky_state)


# ── Quality: Similarity Search API ─────────────────────

@router.post("/api/quality/similarity")
async def quality_similarity(req: SimRequest):
    if sim_state["running"]:
        return {"error": "Already running"}
    img_dir = safe_image_dir(req.img_dir)
    query = req.query_path or ""
    top_k = int(req.top_k)
    sim_state.update(running=True, progress=0, total=0, msg="Building index...", results=[])

    @route_errors(state=sim_state, scope="similarity")
    def _run():
        imgs = glob_images(img_dir, recursive=req.recursive)
        sim_state["total"] = len(imgs)
        hashes = []
        for i, fp in enumerate(imgs):
            if not sim_state.get("running", True):
                sim_state.update(msg="Stopped by user")
                break
            frame = imread(fp)
            if frame is not None:
                hashes.append((os.path.basename(fp), _dhash(frame, 16)))
            sim_state["progress"] = i + 1
        if query and os.path.isfile(query):
            q_frame = imread(query)
            q_hash = _dhash(q_frame, 16) if q_frame is not None else 0
            ranked = sorted(hashes, key=lambda x: _core_hamming(x[1], q_hash))
            sim_state["results"] = [{"rank": i+1, "image": name, "distance": _core_hamming(h, q_hash)} for i, (name, h) in enumerate(ranked[:top_k])]
        else:
            sim_state["results"] = [{"rank": i+1, "image": name, "distance": 0} for i, (name, _) in enumerate(hashes[:top_k])]
        sim_state.update(running=False, msg="Complete")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/quality/similarity/status")
async def similarity_status():
    return sim_state.snapshot() if hasattr(sim_state, 'snapshot') else dict(sim_state)


