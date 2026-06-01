"""/api/clip/*, /api/embedder/*, /api/segmentation/*, /api/batch/*, /api/quantize/*, /api/inspector/*, /api/profiler/*, /api/tracking/*, /api/hf/*, /api/force-stop/* 라우터."""
import os
import time
import uuid
import asyncio

import cv2
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

import threading
from collections import OrderedDict

from core.model_loader import load_model as _load_model
from core.inference import run_inference, letterbox, preprocess, _padded_to_tensor, run_segmentation
from server.errors import route_errors
from server.path_safety import safe_image_file, safe_image_dir, safe_model_file, safe_path, UnsafePathError
from server.state import clip_state, embedder_state, seg_state, quant_state, all_states, executor
from server.utils import imread, encode_jpeg, glob_images, overlay_segmentation, generate_palette

router = APIRouter()

# ── CLIP Zero-Shot API ─────────────────────────────────
class CLIPRequest(BaseModel):
    image_encoder: str
    text_encoder: str
    img_dir: str
    labels: str  # comma-separated
    # CLIP zero-shot accuracy is calibrated against "a photo of a {label}."
    # prompts. Bare labels degrade accuracy; allow overriding the template.
    prompt_template: str = "a photo of a {}."

# NOTE: clip_state is imported from server.state — do NOT re-declare here.
# Re-declaring would break /api/force-stop/clip since all_states["clip"] would
# still point to the original TaskState while routes mutate a different dict.

@router.post("/api/clip/run")
async def run_clip(req: CLIPRequest):
    if clip_state["running"]:
        return {"error": "Already running"}
    # Validate user paths synchronously before submitting the worker so bad
    # input is rejected at the boundary rather than failing deep in ORT.
    image_encoder = safe_model_file(req.image_encoder)
    text_encoder = safe_model_file(req.text_encoder)
    img_dir = safe_image_dir(req.img_dir)
    clip_state.update(running=True, progress=0, total=0, msg="Starting...",
                      results=[], skipped=0)

    @route_errors(state=clip_state, scope="clip")
    def _run():
        from core.clip_inference import CLIPModel, simple_tokenize
        model = CLIPModel(image_encoder, text_encoder)
        labels = [l.strip() for l in req.labels.split(",") if l.strip()]
        if not labels:
            clip_state.update(running=False, msg="No labels provided")
            return
        # Wrap labels in the prompt template for zero-shot calibration.
        template = req.prompt_template if "{}" in req.prompt_template else "a photo of a {}."
        text_embs = []
        for label in labels:
            tokens = simple_tokenize(template.format(label))
            text_embs.append(model.encode_text(tokens))
        imgs = glob_images(img_dir)
        if not imgs:
            clip_state.update(running=False, msg="No images found")
            return
        clip_state["total"] = len(imgs)
        clip_state["msg"] = "Running CLIP inference..."
        label_correct = {l: 0 for l in labels}
        label_total = {l: 0 for l in labels}
        skipped = 0
        detail_log = []
        for idx, fp in enumerate(imgs):
            if not clip_state.get("running", True):
                clip_state.update(msg="Stopped by user")
                return
            frame = imread(fp)
            if frame is None:
                skipped += 1
                clip_state["skipped"] = skipped
                clip_state["progress"] = idx + 1
                continue
            ranked = model.zero_shot_classify(frame, text_embs, labels)
            parent = os.path.basename(os.path.dirname(fp)).lower()
            top_label = ranked[0][0].lower() if ranked else ""
            correct = False
            for l in labels:
                if parent == l.lower():
                    label_total[l] += 1
                    if top_label == l.lower():
                        label_correct[l] += 1
                        correct = True
            detail_log.append({
                "file": os.path.basename(fp),
                "gt": parent,
                "pred": ranked[0][0] if ranked else "—",
                "score": round(ranked[0][1], 4) if ranked else 0,
                "correct": correct,
                "top3": [(r[0], round(r[1], 4)) for r in ranked[:3]],
            })
            clip_state["progress"] = idx + 1
        results = []
        total_correct = 0
        total_count = 0
        for l in labels:
            tc = label_total[l]
            cc = label_correct[l]
            total_correct += cc
            total_count += tc
            acc = round(cc / tc * 100, 2) if tc > 0 else 0
            results.append({"label": l, "total": tc, "correct": cc, "accuracy": acc})
        overall_acc = round(total_correct / total_count * 100, 2) if total_count > 0 else 0
        results.append({"label": "Overall", "total": total_count, "correct": total_correct, "accuracy": overall_acc})
        clip_state["results"] = results
        clip_state["detail"] = detail_log[:500]
        msg = "Complete"
        if skipped:
            msg += f" ({skipped} unreadable image(s) skipped)"
        clip_state.update(running=False, msg=msg, skipped=skipped)

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/clip/status")
async def clip_status():
    return clip_state.snapshot() if hasattr(clip_state, "snapshot") else dict(clip_state)


# ── Embedder Evaluation API ────────────────────────────
class EmbedderRequest(BaseModel):
    model_path: str
    model_type: str = "yolo"
    img_dir: str
    top_k: int = 5

# NOTE: embedder_state is imported from server.state — do NOT re-declare here.

@router.post("/api/embedder/run")
async def run_embedder(req: EmbedderRequest):
    if embedder_state["running"]:
        return {"error": "Already running"}
    # Validate user paths synchronously before submitting the worker.
    model_path = safe_model_file(req.model_path)
    img_dir = safe_image_dir(req.img_dir)
    embedder_state.update(running=True, progress=0, total=0, msg="Starting...",
                          results=[], detail=[], skipped=0)

    def _run():
        try:
            mi = _load_model(model_path, model_type=req.model_type)
            # 폴더 구조: img_dir/class_name/image.jpg (없으면 단일 클래스로 처리)
            class_dirs = [d for d in os.listdir(img_dir)
                          if os.path.isdir(os.path.join(img_dir, d))]
            import numpy as np
            # Collect files (class_name, filepath); the index into `all_files`
            # surviving embeddings is the stable global index we reuse below.
            all_files = []
            if class_dirs:
                for cls in sorted(class_dirs):
                    cls_path = os.path.join(img_dir, cls)
                    files = glob_images(cls_path)
                    for f in files:
                        all_files.append((cls, f))
            else:
                # 폴더 구조 없음 — 루트의 이미지를 단일 클래스로
                files = glob_images(img_dir)
                for f in files:
                    all_files.append(("default", f))
            if not all_files:
                embedder_state.update(running=False, msg="No images found")
                return
            embedder_state["total"] = len(all_files)
            embedder_state["msg"] = "Extracting embeddings..."
            # Parallel arrays indexed by a stable global index (gi).
            emb_labels: list[str] = []
            emb_fnames: list[str] = []
            emb_list: list = []
            class_to_gis: dict[str, list[int]] = {}  # class -> global indices
            skipped = 0
            for idx, (cls, fp) in enumerate(all_files):
                frame = imread(fp)
                if frame is None:
                    skipped += 1
                    embedder_state["skipped"] = skipped
                    embedder_state["progress"] = idx + 1
                    continue
                padded, ratio, pad = letterbox(frame, mi.input_size)
                tensor = _padded_to_tensor(padded, mi.input_size)
                if mi.batch_size > 1:
                    tensor = np.repeat(tensor, mi.batch_size, axis=0)
                out = mi.session.run(None, {mi.input_name: tensor})
                emb = out[0][0].flatten().astype(np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-9)
                gi = len(emb_list)
                emb_list.append(emb)
                emb_labels.append(cls)
                emb_fnames.append(os.path.basename(fp))
                class_to_gis.setdefault(cls, []).append(gi)
                embedder_state["progress"] = idx + 1

            # Compute retrieval metrics. Vectorize the full N×N similarity once
            # (all_embs @ all_embs.T) and mask the diagonal (self-match) to -inf
            # via the stable global index instead of O(N²·D) np.array_equal.
            embedder_state["msg"] = "Computing metrics..."
            results = []
            detail_log = []
            if not emb_list:
                embedder_state.update(running=False, msg="No readable images")
                return
            all_embs_arr = np.array(emb_list)
            sim_matrix = all_embs_arr @ all_embs_arr.T  # (N, N)
            np.fill_diagonal(sim_matrix, -np.inf)
            for cls in sorted(class_to_gis.keys()):
                gis = class_to_gis[cls]
                n = len(gis)
                if n < 2:
                    results.append({"class": cls, "retrieval_1": 0, "retrieval_k": 0, "avg_cosine": 0})
                    continue
                r1_correct = 0
                rk_correct = 0
                cosines = []
                for gi in gis:
                    sims = sim_matrix[gi]
                    top_indices = np.argsort(sims)[::-1][:req.top_k]
                    top_labels = [emb_labels[j] for j in top_indices]
                    hit = top_labels[0] == cls
                    if hit:
                        r1_correct += 1
                    if cls in top_labels:
                        rk_correct += 1
                    # Mean cosine to same-class peers (excludes self via diagonal mask).
                    same_sims = [sims[j] for j in gis if j != gi]
                    if same_sims:
                        cosines.append(np.mean(same_sims))
                    if len(detail_log) < 200:
                        detail_log.append({
                            "file": emb_fnames[gi],
                            "gt": cls,
                            "top1": top_labels[0],
                            "top1_file": emb_fnames[top_indices[0]] if len(top_indices) else "?",
                            "top1_sim": round(float(sims[top_indices[0]]), 4) if len(top_indices) else 0,
                            "correct": hit,
                            "top_k": [(emb_labels[j], round(float(sims[j]), 4)) for j in top_indices[:3]],
                        })
                results.append({
                    "class": cls,
                    "retrieval_1": round(r1_correct / n * 100, 2),
                    "retrieval_k": round(rk_correct / n * 100, 2),
                    "avg_cosine": round(float(np.mean(cosines)) if cosines else 0, 4),
                })
            # Overall
            if results:
                avg_r1 = round(np.mean([r["retrieval_1"] for r in results]), 2)
                avg_rk = round(np.mean([r["retrieval_k"] for r in results]), 2)
                avg_cos = round(float(np.mean([r["avg_cosine"] for r in results])), 4)
                results.append({"class": "Overall", "retrieval_1": avg_r1, "retrieval_k": avg_rk, "avg_cosine": avg_cos})
            embedder_state["results"] = results
            embedder_state["detail"] = detail_log
            msg = "Complete"
            if skipped:
                msg += f" ({skipped} unreadable image(s) skipped)"
            embedder_state.update(running=False, msg=msg, skipped=skipped)
        except Exception as e:
            # Kept inline (broad try block spans most of the worker); upgrade
            # to @route_errors when the worker is refactored.
            import logging as _log
            _log.getLogger("ssook.embedder").exception("embedder worker failed")
            embedder_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/embedder/status")
async def embedder_status():
    return embedder_state.snapshot() if hasattr(embedder_state, "snapshot") else dict(embedder_state)

class EmbedderCompareRequest(BaseModel):
    model_path: str
    img_paths: list[str]


@router.post("/api/embedder/compare")
async def embedder_compare(req: EmbedderCompareRequest):
    """Compare embeddings of multiple selected images using the loaded embedder model."""
    if len(req.img_paths) < 2:
        return {"error": "Need at least 2 images"}
    safe_model_file(req.model_path)
    safe_paths = [safe_image_file(p) for p in req.img_paths]

    import onnxruntime as ort
    session = ort.InferenceSession(req.model_path)
    inp = session.get_inputs()[0]
    h = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 224
    w = int(inp.shape[3]) if isinstance(inp.shape[3], int) else 224
    embeddings = []
    names = []
    for fp in safe_paths:
        img = imread(fp)
        if img is None:
            continue
        img = cv2.resize(img, (w, h))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis], dtype=np.float32) / 255.0
        out = session.run(None, {inp.name: tensor})
        emb = out[0].flatten().astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        embeddings.append(emb)
        names.append(os.path.basename(fp))
    if len(embeddings) < 2:
        return {"error": "Could not read enough images"}
    matrix = []
    for i in range(len(embeddings)):
        row = [round(float(np.dot(embeddings[i], embeddings[j])), 4) for j in range(len(embeddings))]
        matrix.append(row)
    return {"names": names, "matrix": matrix}


class BatchAugmentRequest(BaseModel):
    img_dir: str
    aug_type: str = "Flip"
    # Preview-only augmentation. Labels are NOT transformed, so this endpoint
    # only renders a visual preview (the UI shows an aug.preview_only note).
    angle: float = 15.0       # Rotate degrees
    alpha: float = 1.3        # Brightness contrast scale
    beta: float = 30.0        # Brightness additive shift


# ── Batch: Augmentation Preview API ────────────────────
@router.post("/api/batch/augmentation")
async def batch_augmentation(req: BatchAugmentRequest):
    img_dir = safe_image_dir(req.img_dir)
    imgs = glob_images(img_dir)
    if not imgs:
        return {"error": "No images found"}
    import random
    # imgs is non-empty here, so random.choice is safe.
    fp = random.choice(imgs)
    frame = imread(fp)
    if frame is None:
        return {"error": "Cannot read image"}
    original = encode_jpeg(frame)
    aug_type = req.aug_type
    if aug_type == "Flip":
        aug = cv2.flip(frame, 1)
    elif aug_type == "Rotate":
        M = cv2.getRotationMatrix2D((frame.shape[1]//2, frame.shape[0]//2), req.angle, 1.0)
        aug = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    elif aug_type == "Brightness":
        aug = cv2.convertScaleAbs(frame, alpha=req.alpha, beta=req.beta)
    elif "Mosaic" in aug_type:
        h, w = frame.shape[:2]
        half_h, half_w = h//2, w//2
        samples = [imread(random.choice(imgs)) for _ in range(4)]
        samples = [cv2.resize(s, (half_w, half_h)) if s is not None else np.zeros((half_h, half_w, 3), dtype=np.uint8) for s in samples]
        top = np.hstack([samples[0], samples[1]])
        bot = np.hstack([samples[2], samples[3]])
        aug = np.vstack([top, bot])
    else:
        aug = frame.copy()
    augmented = encode_jpeg(aug)
    # preview_only signals the UI that labels were not transformed.
    return {"original": original, "augmented": augmented,
            "file": os.path.basename(fp), "preview_only": True}


# ── Calibration / Quantization API ─────────────────────
class QuantizeRequest(BaseModel):
    model_path: str
    method: str = "dynamic"          # dynamic | static | fp16
    output_path: str = ""
    calibration_dir: str = ""
    max_images: int = 100
    weight_type: str = "uint8"       # uint8 | int8
    activation_type: str = "uint8"
    quant_format: str = "QDQ"        # QDQ | QOperator
    per_channel: bool = True

# NOTE: quant_state is imported from server.state — do NOT re-declare here.

@router.post("/api/quantize")
async def run_quantize(req: QuantizeRequest):
    if quant_state["running"]:
        return {"error": "Quantization already running"}
    # Validate model + output (.onnx, may not exist) + optional calibration dir.
    # Job-submission route returns {error} with 200, matching its UI poller.
    try:
        model_path = safe_model_file(req.model_path)
        out = req.output_path
        if not out:
            base, ext = os.path.splitext(model_path)
            out = f"{base}_{req.method}{ext}"
        out = safe_path(out, allowed_exts={".onnx"}, must_be_file=False)
        calibration_dir = safe_image_dir(req.calibration_dir) if req.calibration_dir else ""
    except UnsafePathError as e:
        return {"error": str(e), "code": f"PATH_{e.code}"}

    quant_state.update(running=True, progress=0, total=0, msg="Starting...", results={})

    def _run():
        try:
            from core.quantizer import quantize_dynamic, quantize_static, convert_fp16
            if req.method == "dynamic":
                quant_state["msg"] = "Dynamic quantization..."
                result = quantize_dynamic(model_path, out, req.weight_type)
            elif req.method == "static":
                if not calibration_dir:
                    quant_state.update(running=False, msg="Error: calibration directory not found")
                    return
                def _prog(cur, tot):
                    quant_state.update(progress=cur, total=tot, msg=f"Calibrating {cur}/{tot}...")
                result = quantize_static(
                    model_path, out, calibration_dir, req.max_images,
                    req.per_channel, req.weight_type, req.activation_type,
                    req.quant_format, on_progress=_prog)
            elif req.method == "fp16":
                quant_state["msg"] = "FP16 conversion..."
                result = convert_fp16(model_path, out)
            else:
                quant_state.update(running=False, msg=f"Error: unknown method {req.method}")
                return
            quant_state.update(running=False, msg="Complete", results=result)
        except Exception as e:
            quant_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True, "output_path": out}

@router.get("/api/quantize/status")
async def quantize_status():
    return quant_state.snapshot() if hasattr(quant_state, "snapshot") else dict(quant_state)


class InspectModelRequest(BaseModel):
    path: str


class ProfileModelRequest(BaseModel):
    path: str
    # Defaults raised for stable percentiles; warmup>=10, num_runs>=50.
    num_runs: int = 50
    warmup: int = 10
    provider: Optional[str] = None   # None/"auto" → app auto-EP (GPU-first)


class InferImageRequest(BaseModel):
    model_path: str
    image_path: str
    conf: float = 0.25
    model_type: Optional[str] = None
    kpt_conf: float = 0.5   # pose keypoint visibility threshold


# ── Phase 1: Model Inspector API ───────────────────────
@router.post("/api/inspector/inspect")
async def api_inspect_model(req: InspectModelRequest):
    path = safe_model_file(req.path)
    from core.model_inspector import inspect_model, inspection_to_dict
    from core.model_cache import get_or_compute

    def _do(p):
        return inspection_to_dict(inspect_model(p))

    return get_or_compute(path, _do)


# ── Phase 1: Model Profiler API ────────────────────────
@router.post("/api/profiler/run")
async def api_profile_model(req: ProfileModelRequest):
    path = safe_model_file(req.path)
    from core.model_profiler import profile_model, profile_to_dict
    # Enforce minimums so a UI default can't request unstable percentiles.
    num_runs = max(req.num_runs, 50)
    warmup = max(req.warmup, 10)
    result = profile_model(path, num_runs=num_runs, warmup=warmup, provider=req.provider)
    return profile_to_dict(result)


# ── Phase 1: Pose Estimation API ───────────────────────
@router.post("/api/infer/pose")
async def api_infer_pose(req: InferImageRequest):
    model_path = safe_model_file(req.model_path)
    image_path = safe_image_file(req.image_path)
    model_type = req.model_type or "pose_yolo"
    frame = imread(image_path)
    if frame is None:
        return {"error": "Cannot read image"}
    mi = _load_model(model_path, model_type=model_type)
    from core.inference import run_pose, COCO_SKELETON, COCO_KPT_NAMES
    result = run_pose(mi, frame, req.conf)
    vis = frame.copy()
    for i in range(len(result.boxes)):
        x1, y1, x2, y2 = result.boxes[i].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        kpts = result.keypoints[i]
        kpt_conf = req.kpt_conf
        for j, (kx, ky, kc) in enumerate(kpts):
            if kc > kpt_conf:
                cv2.circle(vis, (int(kx), int(ky)), 3, (0, 0, 255), -1)
        for a, b in COCO_SKELETON:
            if kpts[a][2] > kpt_conf and kpts[b][2] > kpt_conf:
                cv2.line(vis, (int(kpts[a][0]), int(kpts[a][1])),
                         (int(kpts[b][0]), int(kpts[b][1])), (255, 255, 0), 2)
    return {
        "image": encode_jpeg(vis),
        "num_persons": len(result.boxes),
        "infer_ms": round(result.infer_ms, 2),
        "detections": [
            {"box": result.boxes[i].tolist(),
             "score": round(float(result.scores[i]), 3),
             "keypoints": result.keypoints[i].tolist()}
            for i in range(len(result.boxes))
        ],
    }


# ── Phase 1: Instance Segmentation API ─────────────────
@router.post("/api/infer/instance-seg")
async def api_infer_instance_seg(req: InferImageRequest):
    model_path = safe_model_file(req.model_path)
    image_path = safe_image_file(req.image_path)
    model_type = req.model_type or "instseg_yolo"
    frame = imread(image_path)
    if frame is None:
        return {"error": "Cannot read image"}
    mi = _load_model(model_path, model_type=model_type)
    from core.inference import run_instance_seg
    result = run_instance_seg(mi, frame, req.conf)
    vis = frame.copy()
    colors = generate_palette(max(len(result.masks), 1))
    for i in range(len(result.masks)):
        mask = result.masks[i]
        color = colors[i % len(colors)]
        overlay = vis.copy()
        overlay[mask > 0] = color
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
        x1, y1, x2, y2 = result.boxes[i].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"cls{result.class_ids[i]} {result.scores[i]:.2f}"
        cv2.putText(vis, label, (x1, max(y1-4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return {
        "image": encode_jpeg(vis),
        "num_instances": len(result.boxes),
        "infer_ms": round(result.infer_ms, 2),
        "detections": [
            {"box": result.boxes[i].tolist(),
             "score": round(float(result.scores[i]), 3),
             "class_id": int(result.class_ids[i])}
            for i in range(len(result.boxes))
        ],
    }


# ── Phase 1: Tracking API ──────────────────────────────
# Module-level tracker registry. Bounded LRU (evict oldest past the cap) and
# guarded by a lock — create/update/reset run on different request threads and
# the dict grew unboundedly before (one entry per create, never removed).
_TRACKER_CAP = 32
_trackers: "OrderedDict[str, object]" = OrderedDict()
_trackers_lock = threading.Lock()


def _tracker_get(tid: str):
    with _trackers_lock:
        tracker = _trackers.get(tid)
        if tracker is not None:
            _trackers.move_to_end(tid)  # mark most-recently-used
        return tracker


class TrackerCreateRequest(BaseModel):
    tracker_type: str = "bytetrack"


class TrackerResetRequest(BaseModel):
    tracker_id: str


class TrackerUpdateRequest(BaseModel):
    tracker_id: str
    model_path: str
    image_path: str
    conf: float = 0.25


@router.post("/api/tracking/create")
async def api_tracking_create(req: TrackerCreateRequest):
    from core.tracking import create_tracker
    tid = str(uuid.uuid4())[:8]
    with _trackers_lock:
        _trackers[tid] = create_tracker(req.tracker_type)
        _trackers.move_to_end(tid)
        while len(_trackers) > _TRACKER_CAP:
            _trackers.popitem(last=False)  # evict least-recently-used
    return {"tracker_id": tid, "type": req.tracker_type}


@router.post("/api/tracking/reset")
async def api_tracking_reset(req: TrackerResetRequest):
    tracker = _tracker_get(req.tracker_id)
    if tracker is not None:
        tracker.reset()
        return {"status": "ok"}
    return {"error": "Tracker not found"}


@router.post("/api/tracking/update")
async def api_tracking_update(req: TrackerUpdateRequest):
    """Run detection on a frame, feed it to the tracker, return active tracks."""
    tracker = _tracker_get(req.tracker_id)
    if tracker is None:
        return {"error": "Tracker not found"}
    model_path = safe_model_file(req.model_path)
    image_path = safe_image_file(req.image_path)
    frame = imread(image_path)
    if frame is None:
        return {"error": "Cannot read image"}
    mi = _load_model(model_path)
    result = run_inference(mi, frame, req.conf)
    boxes = np.asarray(result.boxes, dtype=np.float32) if len(result.boxes) else np.empty((0, 4), np.float32)
    scores = np.asarray(result.scores, dtype=np.float32) if len(result.scores) else np.empty((0,), np.float32)
    class_ids = np.asarray(result.class_ids, dtype=np.int64) if len(result.class_ids) else np.empty((0,), np.int64)
    # tracker.update mutates tracker state — serialise per tracker via the lock.
    with _trackers_lock:
        tracks = tracker.update(boxes, scores, class_ids)
    return {
        "tracks": [
            {
                "id": int(t.id),
                "box": [float(v) for v in t.box],
                "score": round(float(t.score), 3),
                "class_id": int(t.class_id),
                "trajectory": [[float(x), float(y)] for x, y in t.trajectory],
            }
            for t in tracks
        ],
    }


# ── HuggingFace Hub API ────────────────────────────────
class HFSearchRequest(BaseModel):
    query: str = ""
    task: str = ""
    limit: int = 20


class HFFilesRequest(BaseModel):
    repo_id: str


class HFDownloadRequest(BaseModel):
    repo_id: str
    filename: str


@router.post("/api/hf/search")
async def hf_search(req: HFSearchRequest):
    from core.hf_downloader import search_models
    return {"results": search_models(req.query, req.task, req.limit)}


@router.post("/api/hf/files")
async def hf_files(req: HFFilesRequest):
    from core.hf_downloader import list_onnx_files
    return {"files": list_onnx_files(req.repo_id)}


@router.post("/api/hf/download")
async def hf_download(req: HFDownloadRequest):
    from core.hf_downloader import download_model
    path = await asyncio.get_event_loop().run_in_executor(
        executor, download_model, req.repo_id, req.filename
    )
    return {"path": path}


@router.get("/api/hf/cached")
async def hf_cached():
    try:
        from core.hf_downloader import list_cached
        return {"models": list_cached()}
    except Exception as e:
        return {"error": str(e)}


class SegmentationRunRequest(BaseModel):
    model_path: str
    img_dir: str
    # Frontend may send either gt_mask_dir or label_dir — accept both.
    gt_mask_dir: str = ""
    label_dir: str = ""
    num_classes: int = 80
    conf: float = 0.25


# Mask extensions probed when locating a GT mask by image stem (SPEC-08).
_MASK_EXTS = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")


def _find_gt_mask(gt_dir: str, stem: str, src_ext: str):
    """Return the first existing GT mask path for `stem`, else None.

    Mirrors eval's ext-flexible lookup so non-.png masks are not silently
    skipped. Probes the source image's own extension last as a fallback.
    """
    for ext in _MASK_EXTS:
        candidate = os.path.join(gt_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    candidate = os.path.join(gt_dir, stem + src_ext)
    return candidate if os.path.isfile(candidate) else None


# ── Segmentation Evaluation API ─────────────────────────
@router.post("/api/segmentation/run")
async def api_segmentation_run(req: SegmentationRunRequest):
    """Run segmentation evaluation (async with progress)."""
    model_path = safe_model_file(req.model_path)
    img_dir = safe_image_dir(req.img_dir)
    gt_dir_raw = req.gt_mask_dir or req.label_dir
    gt_mask_dir = safe_image_dir(gt_dir_raw)
    num_classes = req.num_classes

    seg_state.update(running=True, progress=0, total=0, msg="Starting...",
                     results=[], skipped=0)

    @route_errors(state=seg_state, scope="seg")
    def _run():
        images = glob_images(img_dir)
        seg_state["total"] = len(images)
        if not images:
            seg_state.update(running=False, msg="No images found")
            return

        mi = _load_model(model_path, model_type="segmentation")
        from core.evaluation import evaluate_segmentation

        # Per-image: predict → resize to GT (nearest) → per-class IoU/Dice,
        # then average per class across images. Honors stop_flag + progress.
        all_ious: dict[int, list] = {}
        all_dices: dict[int, list] = {}
        evaluated = 0
        skipped = 0
        for idx, fp in enumerate(images):
            if not seg_state.get("running", True):
                seg_state.update(msg="Stopped by user")
                return
            frame = imread(fp)
            if frame is None:
                skipped += 1
                seg_state.update(progress=idx + 1, skipped=skipped)
                continue
            stem = os.path.splitext(os.path.basename(fp))[0]
            src_ext = os.path.splitext(fp)[1]
            gt_path = _find_gt_mask(gt_mask_dir, stem, src_ext)
            if gt_path is None:
                skipped += 1
                seg_state.update(progress=idx + 1, skipped=skipped)
                continue
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                skipped += 1
                seg_state.update(progress=idx + 1, skipped=skipped)
                continue
            seg_res = run_segmentation(mi, frame)
            if not seg_res or seg_res.mask is None:
                skipped += 1
                seg_state.update(progress=idx + 1, skipped=skipped)
                continue
            pred_mask = seg_res.mask
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
            metrics = evaluate_segmentation(pred_mask, gt_mask, num_classes)
            for k, v in metrics.items():
                if k == "__overall__":
                    continue
                all_ious.setdefault(k, []).append(v["iou"])
                all_dices.setdefault(k, []).append(v["dice"])
            evaluated += 1
            seg_state.update(progress=idx + 1,
                             msg=f"Evaluating {idx + 1}/{len(images)}...")

        # Average per-class IoU/Dice (NaN IoU = class absent in that image).
        detail = {}
        valid_ious, valid_dices = [], []
        for k in sorted(all_ious.keys()):
            ious = [x for x in all_ious[k] if not np.isnan(x)]
            dices = all_dices[k]
            mean_iou = float(np.mean(ious)) if ious else 0.0
            mean_dice = float(np.mean(dices)) if dices else 0.0
            detail[str(k)] = {"iou": round(mean_iou, 6), "dice": round(mean_dice, 6)}
            if ious:
                valid_ious.append(mean_iou)
                valid_dices.append(mean_dice)
        miou = float(np.mean(valid_ious)) if valid_ious else 0.0
        mdice = float(np.mean(valid_dices)) if valid_dices else 0.0
        results = {
            "mIoU": round(miou * 100, 4),
            "mDice": round(mdice * 100, 4),
            "detail": detail,
            "num_images": evaluated,
        }
        msg = "Complete"
        if skipped:
            msg += f" ({skipped} image(s) skipped: no GT mask / unreadable)"
        seg_state.update(running=False, msg=msg, results=results, skipped=skipped)

    executor.submit(_run)
    return {"ok": True, "msg": "Segmentation evaluation started"}


@router.get("/api/segmentation/status")
async def api_segmentation_status():
    """Get segmentation evaluation progress."""
    return seg_state.snapshot() if hasattr(seg_state, 'snapshot') else dict(seg_state)


# ── Infer Save Crops (single image) ────────────────────
@router.post("/api/infer/save-crops")
async def api_infer_save_crops(req: dict):
    """Run inference on a single image and save detection crops."""
    model_path = req.get("model_path", "")
    image_path = req.get("image_path", "")
    conf = req.get("conf", 0.25)

    if not model_path or not image_path:
        return {"error": "model_path and image_path required"}

    frame = imread(image_path)
    if frame is None:
        return {"error": f"Cannot read image: {image_path}"}

    try:
        mi = _load_model(model_path)
        result = run_inference(mi, frame, conf)
        if not hasattr(result, 'boxes') or len(result.boxes) == 0:
            return {"ok": True, "count": 0, "path": ""}

        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("snapshots", f"crops_{ts}")
        os.makedirs(out_dir, exist_ok=True)

        h, w = frame.shape[:2]
        names = getattr(mi, 'names', {}) or {}
        saved = 0
        for i, (box, score, cid) in enumerate(zip(result.boxes, result.scores, result.class_ids)):
            x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            cls_name = names.get(int(cid), str(int(cid)))
            crop = frame[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(out_dir, f"{i:03d}_{cls_name}_{score:.2f}.jpg"), crop)
            saved += 1
        return {"ok": True, "path": out_dir, "count": saved}
    except Exception as e:
        return {"error": str(e)}
