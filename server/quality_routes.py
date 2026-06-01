"""/api/quality/* 라우터."""
import os

import cv2
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from server.errors import route_errors
from server.path_safety import safe_image_dir, safe_label_dir, safe_image_file
from server.state import anomaly_state, quality_state, dup_state, leaky_state, sim_state, executor
from server.utils import imread, glob_images

router = APIRouter()


# ── Result caps ─────────────────────────────────────────
# Workers store at most CAP rows to keep status payloads (polled every
# second) small; the UI surfaces a "showing first N" note when the true
# count exceeds the cap so users never mistake a truncated view for the
# whole dataset (QUAL-09).
ANOMALY_CAP = 1000
QUALITY_CAP = 1000
DUP_CAP = 500


# ── Shared Pydantic shapes ──────────────────────────────
class QualityDirRequest(BaseModel):
    img_dir: str
    label_dir: Optional[str] = ""
    recursive: bool = False
    threshold: Optional[int] = None
    metrics: Optional[list[str]] = None
    # Image-quality knobs (QUAL-12). Defaults reproduce the previous
    # hardcoded behaviour exactly; `threshold` doubles as the blur cutoff
    # so the existing UI field finally does something.
    blur_threshold: Optional[float] = None
    dark_threshold: float = 40.0
    bright_threshold: float = 220.0
    entropy_threshold: float = 3.0
    aspect_min: float = 0.25
    aspect_max: float = 4.0


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
    # Validate label_dir at the boundary too — it feeds open() below, so a
    # raw path would let a crafted request read arbitrary <stem>.txt files
    # outside the path-safety surface (QUAL-07).
    label_dir = safe_label_dir(req.label_dir) if req.label_dir else ""
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
        total_found = len(results)
        truncated = total_found > ANOMALY_CAP
        anomaly_state["results"] = results[:ANOMALY_CAP]
        anomaly_state["truncated"] = truncated
        anomaly_state["total_found"] = total_found
        msg = f"Complete — {total_found} issues found"
        if truncated:
            msg += f" (showing first {ANOMALY_CAP})"
        anomaly_state.update(running=False, msg=msg)

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
    # `threshold` is the legacy UI field; treat it as the blur cutoff so it
    # finally has an effect, falling back to the original default (QUAL-12).
    blur_threshold = float(req.blur_threshold if req.blur_threshold is not None
                           else (req.threshold if req.threshold is not None else 50.0))
    dark_threshold = req.dark_threshold
    bright_threshold = req.bright_threshold
    entropy_threshold = req.entropy_threshold
    aspect_min, aspect_max = req.aspect_min, req.aspect_max
    quality_state.update(running=True, progress=0, total=0, msg="Checking...", results=[])

    @route_errors(state=quality_state, scope="quality")
    def _run():
        imgs = glob_images(img_dir, recursive=recursive)
        quality_state["total"] = len(imgs)
        results = []
        skipped = 0
        for i, fp in enumerate(imgs):
            if not quality_state.get("running", True):
                quality_state.update(msg="Stopped by user")
                break
            frame = imread(fp)
            if frame is None:
                skipped += 1
                quality_state["progress"] = i + 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2)
            brightness = round(float(gray.mean()), 2)
            # Single histogram pass — the previous code computed it twice
            # (QUAL-11). p is the per-bin probability; 1e-12 guards log2(0).
            hist = np.histogram(gray, 256, [0, 256])[0] / gray.size
            entropy = round(float(-np.sum(hist * np.log2(hist + 1e-12))), 2)
            h, w = frame.shape[:2]
            aspect = round(w / h, 2)
            issues = []
            if blur < blur_threshold:
                issues.append("Blurry")
            if brightness < dark_threshold:
                issues.append("Dark")
            elif brightness > bright_threshold:
                issues.append("Overexposed")
            # Low entropy ⇒ flat/featureless frame (was computed but unused).
            # Machine code drives the i18n key quality.low_detail.
            if entropy < entropy_threshold:
                issues.append("low_detail")
            if aspect > aspect_max or aspect < aspect_min:
                issues.append("Odd aspect")
            results.append({"file": os.path.basename(fp), "blur": blur, "brightness": brightness,
                            "entropy": entropy, "aspect": aspect, "issues": ", ".join(issues) or "OK"})
            quality_state["progress"] = i + 1
        total_found = len(results)
        truncated = total_found > QUALITY_CAP
        quality_state["results"] = results[:QUALITY_CAP]
        quality_state["truncated"] = truncated
        quality_state["total_found"] = total_found
        quality_state["skipped"] = skipped
        msg = f"Complete — {total_found} images checked"
        if skipped:
            msg += f", {skipped} unreadable skipped"
        if truncated:
            msg += f" (showing first {QUALITY_CAP})"
        quality_state.update(running=False, msg=msg)

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/quality/image-quality/status")
async def quality_status():
    return quality_state.snapshot() if hasattr(quality_state, 'snapshot') else dict(quality_state)


# ── Quality: Near-Duplicate Detector API ───────────────
# Use the canonical hash from core.hashing so merger and duplicate detector
# agree on what "duplicate" means.
from core.hashing import (
    compute_dhash as _core_dhash,
    hamming as _core_hamming,
    find_near_duplicates as _find_near_duplicates,
    cluster_near_duplicates as _cluster_near_duplicates,
)


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
        names: list[str] = []
        hashes: list[int] = []
        skipped = 0
        for i, fp in enumerate(imgs):
            if not dup_state.get("running", True):
                dup_state.update(msg="Stopped by user")
                break
            frame = imread(fp)
            if frame is None:
                skipped += 1
            else:
                names.append(os.path.basename(fp))
                hashes.append(_dhash(frame))
            dup_state["progress"] = i + 1
        dup_state["msg"] = "Comparing..."
        # Group ids over connected components so A~B~C cluster under ONE id
        # instead of a per-pair counter that never expressed clusters (QUAL-05).
        groups = _cluster_near_duplicates(hashes, threshold)
        pairs = _find_near_duplicates(hashes, threshold)
        results = [{"group": groups[i], "image_a": names[i], "image_b": names[j],
                    "distance": _core_hamming(hashes[i], hashes[j])}
                   for i, j in pairs[:DUP_CAP]]
        total_found = len(pairs)
        truncated = total_found > DUP_CAP
        dup_state["results"] = results
        dup_state["truncated"] = truncated
        dup_state["total_found"] = total_found
        dup_state["skipped"] = skipped
        msg = f"Complete — {total_found} pairs found"
        if skipped:
            msg += f", {skipped} unreadable skipped"
        if truncated:
            msg += f" (showing first {DUP_CAP})"
        dup_state.update(running=False, msg=msg)

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
        # Glob every split up front so the progress bar has a real `total`
        # before hashing starts (was set only afterwards → bar stuck at 0%).
        split_files = {name: glob_images(d) for name, d in dirs.items() if d}
        leaky_state["total"] = sum(len(v) for v in split_files.values())
        done = 0
        skipped = 0
        split_hashes: dict[str, dict[str, int]] = {}
        for name, imgs in split_files.items():
            hashes: dict[str, int] = {}
            for fp in imgs:
                if not leaky_state.get("running", True):
                    leaky_state.update(msg="Stopped by user")
                    break
                # Decode once — the old comprehension called imread twice
                # (filter + hash), doubling disk I/O and decode cost (QUAL-03).
                frame = imread(fp)
                if frame is None:
                    skipped += 1
                else:
                    hashes[os.path.basename(fp)] = _dhash(frame)
                done += 1
                leaky_state["progress"] = done
            split_hashes[name] = hashes
            if not leaky_state.get("running", True):
                break
        results = []
        names = list(split_hashes.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                dupes = 0
                files = []
                for fa, ha in split_hashes[names[i]].items():
                    for fb, hb in split_hashes[names[j]].items():
                        dist = _core_hamming(ha, hb)
                        if dist <= threshold:
                            dupes += 1
                            if len(files) < 10:
                                files.append(f"{fa} ↔ {fb}")
                results.append({"pair": f"{names[i]} ↔ {names[j]}", "duplicates": dupes, "files": "; ".join(files)})
        leaky_state["results"] = results
        leaky_state["skipped"] = skipped
        msg = "Complete"
        if skipped:
            msg += f" — {skipped} unreadable skipped"
        leaky_state.update(running=False, msg=msg)

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/quality/leaky/status")
async def leaky_status():
    return leaky_state.snapshot() if hasattr(leaky_state, 'snapshot') else dict(leaky_state)


# ── Quality: Similarity Search API ─────────────────────

# Similarity hashes at the project-wide 64-bit width (QUAL-02). The previous
# 256-bit (size=16) put it in a different threshold space from every other tool.
_SIM_HASH_SIZE = 8


@router.post("/api/quality/similarity")
async def quality_similarity(req: SimRequest):
    if sim_state["running"]:
        return {"error": "Already running"}
    img_dir = safe_image_dir(req.img_dir)
    # Validate the query at the boundary — must_exist + image extension —
    # now that the frontend actually sends query_path (QUAL-01/QUAL-08).
    query = safe_image_file(req.query_path) if req.query_path else ""
    top_k = int(req.top_k)
    recursive = req.recursive
    sim_state.update(running=True, progress=0, total=0, msg="Building index...", results=[])

    @route_errors(state=sim_state, scope="similarity")
    def _run():
        # Reuse the cached hash list when the corpus (dir, recursion, width)
        # is unchanged so repeat queries against the same folder skip the
        # full re-glob + re-decode (QUAL-04). Invalidated when the key differs.
        cache_key = (img_dir, recursive, _SIM_HASH_SIZE)
        cached = sim_state.get("index")
        if cached and cached.get("key") == cache_key:
            hashes = cached["hashes"]
            skipped = cached.get("skipped", 0)
            sim_state["total"] = len(hashes)
            sim_state["progress"] = len(hashes)
        else:
            imgs = glob_images(img_dir, recursive=recursive)
            sim_state["total"] = len(imgs)
            hashes = []
            skipped = 0
            for i, fp in enumerate(imgs):
                if not sim_state.get("running", True):
                    sim_state.update(msg="Stopped by user")
                    return
                frame = imread(fp)
                if frame is None:
                    skipped += 1
                else:
                    hashes.append((os.path.basename(fp), _dhash(frame, _SIM_HASH_SIZE)))
                sim_state["progress"] = i + 1
            sim_state["index"] = {"key": cache_key, "hashes": hashes, "skipped": skipped}

        if query:
            q_frame = imread(query)
            if q_frame is None:
                # No silent q_hash=0 fallback — that ranked every image by
                # distance-to-zero and looked like a real result (QUAL-10).
                sim_state.update(running=False, msg="Error: query image could not be decoded")
                return
            q_hash = _dhash(q_frame, _SIM_HASH_SIZE)
            ranked = sorted(hashes, key=lambda x: _core_hamming(x[1], q_hash))
            sim_state["results"] = [{"rank": i+1, "image": name, "distance": _core_hamming(h, q_hash)} for i, (name, h) in enumerate(ranked[:top_k])]
        else:
            sim_state["results"] = [{"rank": i+1, "image": name, "distance": 0} for i, (name, _) in enumerate(hashes[:top_k])]
        msg = "Complete"
        if skipped:
            msg += f" — {skipped} unreadable skipped"
        sim_state.update(running=False, msg=msg)

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/quality/similarity/status")
async def similarity_status():
    return sim_state.snapshot() if hasattr(sim_state, 'snapshot') else dict(sim_state)


