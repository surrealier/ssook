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

from core.model_loader import load_model as _load_model
from core.inference import run_inference, letterbox, preprocess, _padded_to_tensor, run_segmentation
from server.errors import route_errors
from server.path_safety import safe_image_file, safe_image_dir, safe_model_file
from server.state import clip_state, embedder_state, seg_state, quant_state, all_states, executor
from server.utils import imread, encode_jpeg, glob_images, overlay_segmentation, generate_palette

router = APIRouter()

# ── CLIP Zero-Shot API ─────────────────────────────────
class CLIPRequest(BaseModel):
    image_encoder: str
    text_encoder: str
    img_dir: str
    labels: str  # comma-separated

# NOTE: clip_state is imported from server.state — do NOT re-declare here.
# Re-declaring would break /api/force-stop/clip since all_states["clip"] would
# still point to the original TaskState while routes mutate a different dict.

@router.post("/api/clip/run")
async def run_clip(req: CLIPRequest):
    if clip_state["running"]:
        return {"error": "Already running"}
    clip_state.update(running=True, progress=0, total=0, msg="Starting...", results=[])

    @route_errors(state=clip_state, scope="clip")
    def _run():
        from core.clip_inference import CLIPModel, simple_tokenize
        model = CLIPModel(req.image_encoder, req.text_encoder)
        labels = [l.strip() for l in req.labels.split(",") if l.strip()]
        if not labels:
            clip_state.update(running=False, msg="No labels provided")
            return
        text_embs = []
        for label in labels:
            tokens = simple_tokenize(label)
            text_embs.append(model.encode_text(tokens))
        imgs = glob_images(req.img_dir)
        if not imgs:
            clip_state.update(running=False, msg="No images found")
            return
        clip_state["total"] = len(imgs)
        clip_state["msg"] = "Running CLIP inference..."
        label_correct = {l: 0 for l in labels}
        label_total = {l: 0 for l in labels}
        detail_log = []
        for idx, fp in enumerate(imgs):
            if not clip_state.get("running", True):
                clip_state.update(msg="Stopped by user")
                return
            frame = imread(fp)
            if frame is None:
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
        clip_state.update(running=False, msg="Complete")

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
    embedder_state.update(running=True, progress=0, total=0, msg="Starting...", results=[], detail=[])

    def _run():
        try:
            from core.model_loader import load_model as _load
            mi = _load_model(req.model_path, model_type=req.model_type)
            # 폴더 구조: img_dir/class_name/image.jpg (없으면 단일 클래스로 처리)
            class_dirs = [d for d in os.listdir(req.img_dir)
                          if os.path.isdir(os.path.join(req.img_dir, d))]
            # Collect embeddings per class
            class_embeddings = {}  # class_name -> list of embeddings
            all_files = []
            if class_dirs:
                for cls in sorted(class_dirs):
                    cls_path = os.path.join(req.img_dir, cls)
                    files = glob_images(cls_path)
                    for f in files:
                        all_files.append((cls, f))
            else:
                # 폴더 구조 없음 — 루트의 이미지를 단일 클래스로
                files = glob_images(req.img_dir)
                for f in files:
                    all_files.append(("default", f))
            if not all_files:
                embedder_state.update(running=False, msg="No images found")
                return
            embedder_state["total"] = len(all_files)
            embedder_state["msg"] = "Extracting embeddings..."
            embeddings = []  # (class_name, embedding)
            for idx, (cls, fp) in enumerate(all_files):
                frame = imread(fp)
                if frame is None:
                    embedder_state["progress"] = idx + 1
                    continue
                # Use model to get embedding (last layer output, flattened)
                from core.inference import preprocess, _padded_to_tensor, letterbox
                padded, ratio, pad = letterbox(frame, mi.input_size)
                tensor = _padded_to_tensor(padded, mi.input_size)
                import numpy as np
                if mi.batch_size > 1:
                    tensor = np.repeat(tensor, mi.batch_size, axis=0)
                out = mi.session.run(None, {mi.input_name: tensor})
                emb = out[0][0].flatten().astype(np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-9)
                embeddings.append((cls, emb))
                if cls not in class_embeddings:
                    class_embeddings[cls] = []
                class_embeddings[cls].append(emb)
                embedder_state["progress"] = idx + 1
            # Compute retrieval metrics
            import numpy as np
            embedder_state["msg"] = "Computing metrics..."
            results = []
            detail_log = []
            all_embs_arr = np.array([e for _, e in embeddings])
            all_labels = [c for c, _ in embeddings]
            all_fnames = [os.path.basename(fp) for _, fp in all_files[:len(embeddings)]]
            for cls in sorted(class_embeddings.keys()):
                cls_embs = class_embeddings[cls]
                n = len(cls_embs)
                if n < 2:
                    results.append({"class": cls, "retrieval_1": 0, "retrieval_k": 0, "avg_cosine": 0})
                    continue
                r1_correct = 0
                rk_correct = 0
                cosines = []
                for i, emb in enumerate(cls_embs):
                    sims = np.dot(all_embs_arr, emb)
                    self_indices = [j for j, (c, e) in enumerate(embeddings) if c == cls and np.array_equal(e, emb)]
                    for si in self_indices:
                        sims[si] = -1
                    top_indices = np.argsort(sims)[::-1][:req.top_k]
                    top_labels = [all_labels[j] for j in top_indices]
                    hit = top_labels[0] == cls
                    if hit:
                        r1_correct += 1
                    if cls in top_labels:
                        rk_correct += 1
                    same_sims = [sims[j] for j, (c, _) in enumerate(embeddings) if c == cls and j not in self_indices]
                    if same_sims:
                        cosines.append(np.mean(same_sims))
                    # Per-image detail (limit 200)
                    if len(detail_log) < 200:
                        query_idx = self_indices[0] if self_indices else -1
                        detail_log.append({
                            "file": all_fnames[query_idx] if query_idx >= 0 else "?",
                            "gt": cls,
                            "top1": top_labels[0],
                            "top1_file": all_fnames[top_indices[0]] if top_indices[0] < len(all_fnames) else "?",
                            "top1_sim": round(float(sims[top_indices[0]]), 4),
                            "correct": hit,
                            "top_k": [(all_labels[j], round(float(sims[j]), 4)) for j in top_indices[:3]],
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
            embedder_state.update(running=False, msg="Complete")
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


# ── Batch: Augmentation Preview API ────────────────────
@router.post("/api/batch/augmentation")
async def batch_augmentation(req: BatchAugmentRequest):
    img_dir = safe_image_dir(req.img_dir)
    imgs = glob_images(img_dir)
    if not imgs:
        return {"error": "No images found"}
    import random
    fp = random.choice(imgs)
    frame = imread(fp)
    if frame is None:
        return {"error": "Cannot read image"}
    original = encode_jpeg(frame)
    aug_type = req.aug_type
    if aug_type == "Flip":
        aug = cv2.flip(frame, 1)
    elif aug_type == "Rotate":
        M = cv2.getRotationMatrix2D((frame.shape[1]//2, frame.shape[0]//2), 15, 1.0)
        aug = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    elif aug_type == "Brightness":
        aug = cv2.convertScaleAbs(frame, alpha=1.3, beta=30)
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
    return {"original": original, "augmented": augmented, "file": os.path.basename(fp)}


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
    if not req.model_path or not os.path.isfile(req.model_path):
        return {"error": "Model file not found"}

    out = req.output_path
    if not out:
        base, ext = os.path.splitext(req.model_path)
        out = f"{base}_{req.method}{ext}"

    quant_state.update(running=True, progress=0, total=0, msg="Starting...", results={})

    def _run():
        try:
            from core.quantizer import quantize_dynamic, quantize_static, convert_fp16
            if req.method == "dynamic":
                quant_state["msg"] = "Dynamic quantization..."
                result = quantize_dynamic(req.model_path, out, req.weight_type)
            elif req.method == "static":
                if not req.calibration_dir or not os.path.isdir(req.calibration_dir):
                    quant_state.update(running=False, msg="Error: calibration directory not found")
                    return
                def _prog(cur, tot):
                    quant_state.update(progress=cur, total=tot, msg=f"Calibrating {cur}/{tot}...")
                result = quantize_static(
                    req.model_path, out, req.calibration_dir, req.max_images,
                    req.per_channel, req.weight_type, req.activation_type,
                    req.quant_format, on_progress=_prog)
            elif req.method == "fp16":
                quant_state["msg"] = "FP16 conversion..."
                result = convert_fp16(req.model_path, out)
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
    num_runs: int = 20


class InferImageRequest(BaseModel):
    model_path: str
    image_path: str
    conf: float = 0.25
    model_type: Optional[str] = None


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
    result = profile_model(path, num_runs=req.num_runs)
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
        for j, (kx, ky, kc) in enumerate(kpts):
            if kc > 0.5:
                cv2.circle(vis, (int(kx), int(ky)), 3, (0, 0, 255), -1)
        for a, b in COCO_SKELETON:
            if kpts[a][2] > 0.5 and kpts[b][2] > 0.5:
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
_trackers = {}


class TrackerCreateRequest(BaseModel):
    tracker_type: str = "bytetrack"


class TrackerResetRequest(BaseModel):
    tracker_id: str


@router.post("/api/tracking/create")
async def api_tracking_create(req: TrackerCreateRequest):
    from core.tracking import create_tracker
    tid = str(uuid.uuid4())[:8]
    _trackers[tid] = create_tracker(req.tracker_type)
    return {"tracker_id": tid, "type": req.tracker_type}


@router.post("/api/tracking/reset")
async def api_tracking_reset(req: TrackerResetRequest):
    if req.tracker_id in _trackers:
        _trackers[req.tracker_id].reset()
        return {"status": "ok"}
    return {"error": "Tracker not found"}


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
    gt_mask_dir: str
    num_classes: int = 80
    conf: float = 0.25


# ── Segmentation Evaluation API ─────────────────────────
@router.post("/api/segmentation/run")
async def api_segmentation_run(req: SegmentationRunRequest):
    """Run segmentation evaluation (async with progress)."""
    model_path = safe_model_file(req.model_path)
    img_dir = safe_image_dir(req.img_dir)
    gt_mask_dir = safe_image_dir(req.gt_mask_dir)

    seg_state.update(running=True, progress=0, total=0, msg="Starting...", results=[])

    @route_errors(state=seg_state, scope="seg")
    def _run():
        images = glob_images(img_dir)
        seg_state["total"] = len(images)
        if not images:
            seg_state.update(running=False, msg="No images found")
            return

        mi = _load_model(model_path, model_type="segmentation")
        from core.evaluation import evaluate_segmentation
        results = evaluate_segmentation(
            mi, images, gt_mask_dir, req.num_classes, req.conf,
            progress_cb=lambda i, msg: seg_state.update(progress=i, msg=msg),
            stop_flag=lambda: not seg_state["running"],
        )
        seg_state.update(running=False, msg="Complete", results=results)

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
