"""/api/quality/* 라우터."""
import os

import cv2
import numpy as np
from fastapi import APIRouter

from server.state import anomaly_state, quality_state, dup_state, leaky_state, sim_state, executor
from server.utils import imread, glob_images

router = APIRouter()

# ── Quality: Anomaly Detector API ──────────────────────
anomaly_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": []}

@router.post("/api/quality/anomaly")
async def quality_anomaly(req: dict):
    if anomaly_state["running"]:
        return {"error": "Already running"}
    anomaly_state.update(running=True, progress=0, total=0, msg="Scanning...", results=[])
    img_dir = req.get("img_dir", "")
    label_dir = req.get("label_dir", "")
    recursive = req.get("recursive", False)

    def _run():
        try:
            imgs = glob_images(img_dir, recursive=recursive)
            anomaly_state["total"] = len(imgs)
            results = []
            for i, fp in enumerate(imgs):
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
                            # OOB check
                            if cx - bw/2 < -0.01 or cy - bh/2 < -0.01 or cx + bw/2 > 1.01 or cy + bh/2 > 1.01:
                                issues.append("Out-of-bounds")
                            # Size outlier
                            area = bw * bh
                            if area < 0.0001:
                                issues.append("Tiny box")
                            elif area > 0.9:
                                issues.append("Huge box")
                            # Aspect ratio
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
        except Exception as e:
            anomaly_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/quality/anomaly/status")
async def anomaly_status():
    return dict(anomaly_state)


# ── Quality: Image Quality Checker API ─────────────────
quality_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": []}

@router.post("/api/quality/image-quality")
async def quality_check(req: dict):
    if quality_state["running"]:
        return {"error": "Already running"}
    quality_state.update(running=True, progress=0, total=0, msg="Checking...", results=[])
    img_dir = req.get("img_dir", "")
    recursive = req.get("recursive", False)

    def _run():
        try:
            imgs = glob_images(img_dir, recursive=recursive)
            quality_state["total"] = len(imgs)
            results = []
            for i, fp in enumerate(imgs):
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
        except Exception as e:
            quality_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/quality/image-quality/status")
async def quality_status():
    return dict(quality_state)


# ── Quality: Near-Duplicate Detector API ───────────────
dup_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": []}

def _dhash(img, size=8):
    resized = cv2.resize(img, (size+1, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    diff = gray[:, 1:] > gray[:, :-1]
    return sum(2**i for i, v in enumerate(diff.flatten()) if v)

@router.post("/api/quality/duplicate")
async def quality_duplicate(req: dict):
    if dup_state["running"]:
        return {"error": "Already running"}
    dup_state.update(running=True, progress=0, total=0, msg="Hashing...", results=[])
    img_dir = req.get("img_dir", "")
    threshold = int(req.get("threshold", 10))
    recursive = req.get("recursive", False)

    def _run():
        try:
            imgs = glob_images(img_dir, recursive=recursive)
            dup_state["total"] = len(imgs)
            hashes = []
            for i, fp in enumerate(imgs):
                frame = imread(fp)
                if frame is not None:
                    hashes.append((os.path.basename(fp), _dhash(frame)))
                dup_state["progress"] = i + 1
            dup_state["msg"] = "Comparing..."
            results = []
            group = 1
            for i in range(len(hashes)):
                for j in range(i+1, len(hashes)):
                    dist = bin(hashes[i][1] ^ hashes[j][1]).count('1')
                    if dist <= threshold:
                        results.append({"group": group, "image_a": hashes[i][0], "image_b": hashes[j][0], "distance": dist})
                        group += 1
                    if len(results) >= 500:
                        break
                if len(results) >= 500:
                    break
            dup_state["results"] = results
            dup_state.update(running=False, msg=f"Complete — {len(results)} pairs found")
        except Exception as e:
            dup_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/quality/duplicate/status")
async def duplicate_status():
    return dict(dup_state)


# ── Quality: Leaky Split Detector API ──────────────────
leaky_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": []}

@router.post("/api/quality/leaky")
async def quality_leaky(req: dict):
    if leaky_state["running"]:
        return {"error": "Already running"}
    leaky_state.update(running=True, progress=0, total=0, msg="Scanning...", results=[])
    dirs = {k: req.get(k, "") for k in ["train_dir", "val_dir", "test_dir"]}
    threshold = int(req.get("threshold", 10))

    def _run():
        try:
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
                            dist = bin(ha ^ hb).count('1')
                            if dist <= threshold:
                                dupes += 1
                                if len(files) < 10:
                                    files.append(f"{fa} ↔ {fb}")
                    results.append({"pair": f"{names[i]} ↔ {names[j]}", "duplicates": dupes, "files": "; ".join(files)})
            leaky_state["results"] = results
            leaky_state.update(running=False, msg="Complete")
        except Exception as e:
            leaky_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/quality/leaky/status")
async def leaky_status():
    return dict(leaky_state)


# ── Quality: Similarity Search API ─────────────────────
sim_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": [], "index": None}

@router.post("/api/quality/similarity")
async def quality_similarity(req: dict):
    if sim_state["running"]:
        return {"error": "Already running"}
    sim_state.update(running=True, progress=0, total=0, msg="Building index...", results=[])
    img_dir = req.get("img_dir", "")
    query = req.get("query", "")
    top_k = int(req.get("top_k", 10))

    def _run():
        try:
            imgs = glob_images(img_dir)
            sim_state["total"] = len(imgs)
            hashes = []
            for i, fp in enumerate(imgs):
                frame = imread(fp)
                if frame is not None:
                    hashes.append((os.path.basename(fp), _dhash(frame, 16)))
                sim_state["progress"] = i + 1
            if query and os.path.isfile(query):
                q_frame = imread(query)
                q_hash = _dhash(q_frame, 16) if q_frame is not None else 0
                ranked = sorted(hashes, key=lambda x: bin(x[1] ^ q_hash).count('1'))
                sim_state["results"] = [{"rank": i+1, "image": name, "distance": bin(h ^ q_hash).count('1')} for i, (name, h) in enumerate(ranked[:top_k])]
            else:
                sim_state["results"] = [{"rank": i+1, "image": name, "distance": 0} for i, (name, _) in enumerate(hashes[:top_k])]
            sim_state.update(running=False, msg="Complete")
        except Exception as e:
            sim_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/quality/similarity/status")
async def similarity_status():
    return dict(sim_state)


