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
from server.path_safety import (
    safe_image_file, safe_image_dir, safe_model_file, safe_video_file,
    safe_path, UnsafePathError,
)
from server.state import (
    clip_state, embedder_state, seg_state, quant_state,
    TaskState, all_states, executor,
)
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
    # Validate user paths synchronously before submitting the worker so bad
    # input is rejected at the boundary rather than failing deep in ORT.
    image_encoder = safe_model_file(req.image_encoder)
    text_encoder = safe_model_file(req.text_encoder)
    img_dir = safe_image_dir(req.img_dir)
    # Atomic check-and-set: two concurrent POSTs (double-click / parallel tabs)
    # would otherwise both pass a read-then-write guard and race clip_state.
    if not clip_state.try_start(progress=0, total=0, msg="Starting...",
                                results=[], detail=[], skipped=0, matched=0):
        return {"error": "Already running"}

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
        # GT label = each image's parent-folder name (ImageNet-style
        # img_dir/<class>/*.jpg layout). A non-recursive glob only matched
        # files directly under img_dir — whose parent IS img_dir — so GT never
        # matched a label and accuracy was always 0. Scan recursively and skip
        # images sitting directly in img_dir (no class subfolder = no GT).
        img_dir_real = os.path.realpath(img_dir)
        imgs = [fp for fp in glob_images(img_dir, recursive=True)
                if os.path.realpath(os.path.dirname(fp)) != img_dir_real]
        if not imgs:
            clip_state.update(
                running=False,
                msg="No class-subfolder images found (expected img_dir/<label>/*.jpg)")
            return
        clip_state["total"] = len(imgs)
        clip_state["msg"] = "Running CLIP inference..."
        label_correct = {l: 0 for l in labels}
        label_total = {l: 0 for l in labels}
        skipped = 0
        detail_log = []
        # Batched encode: a 10k-image run was 10k sequential ORT calls. Process
        # in chunks of CHUNK images, stacking each chunk into one image-encoder
        # forward and scoring against the cached text-embedding matrix. Cancel is
        # re-checked per chunk so force-stop stays responsive.
        _CHUNK = 32
        processed = 0
        for start in range(0, len(imgs), _CHUNK):
            if not clip_state.get("running", True):
                clip_state.update(msg="Stopped by user")
                return
            chunk_fps = imgs[start:start + _CHUNK]
            frames = []
            kept_fps = []
            for fp in chunk_fps:
                frame = imread(fp)
                if frame is None:
                    skipped += 1
                    continue
                frames.append(frame)
                kept_fps.append(fp)
            processed += len(chunk_fps)
            if frames:
                ranked_batch = model.zero_shot_classify_batch(frames, text_embs, labels)
                for fp, ranked in zip(kept_fps, ranked_batch):
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
            clip_state.update(progress=processed, skipped=skipped)
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
        # `matched` = images whose parent folder matched a label. 0 means the
        # folder layout didn't line up with any label — surface it so the user
        # sees a layout mismatch instead of a silent 0% accuracy.
        clip_state["matched"] = total_count
        if total_count == 0:
            msg = "Complete — 0 images matched any label folder (expected img_dir/<label>/*.jpg)"
        else:
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
    # Validate user paths synchronously before submitting the worker.
    model_path = safe_model_file(req.model_path)
    img_dir = safe_image_dir(req.img_dir)
    # Atomic start guard — see clip_state.try_start rationale.
    if not embedder_state.try_start(progress=0, total=0, msg="Starting...",
                                    results=[], detail=[], skipped=0):
        return {"error": "Already running"}

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
            # Batched extraction: stack CHUNK preprocessed frames into one forward
            # instead of one ORT call per image. Fixed-batch models (mi.batch_size>1)
            # set the chunk to the model batch and the last short chunk is padded
            # then trimmed. The stable global index (gi) is preserved so the
            # diagonal-masked self-similarity below stays correct.
            _CHUNK = mi.batch_size if mi.batch_size > 1 else 32
            for start in range(0, len(all_files), _CHUNK):
                if not embedder_state.get("running", True):
                    embedder_state.update(msg="Stopped by user")
                    return
                chunk = all_files[start:start + _CHUNK]
                tensors = []
                kept = []  # (cls, fname)
                for cls, fp in chunk:
                    frame = imread(fp)
                    if frame is None:
                        skipped += 1
                        continue
                    padded, ratio, pad = letterbox(frame, mi.input_size)
                    tensors.append(_padded_to_tensor(padded, mi.input_size).copy())
                    kept.append((cls, os.path.basename(fp)))
                if tensors:
                    batch = np.concatenate(tensors, axis=0)
                    real = batch.shape[0]
                    if mi.batch_size > 1 and real < mi.batch_size:
                        batch = np.concatenate(
                            [batch, np.repeat(batch[-1:], mi.batch_size - real, axis=0)], axis=0)
                    out = mi.session.run(None, {mi.input_name: batch})[0]
                    out = out.reshape(out.shape[0], -1)[:real].astype(np.float32)
                    norms = np.linalg.norm(out, axis=1, keepdims=True)
                    out = out / (norms + 1e-9)
                    for (cls, fname), emb in zip(kept, out):
                        gi = len(emb_list)
                        emb_list.append(emb)
                        emb_labels.append(cls)
                        emb_fnames.append(fname)
                        class_to_gis.setdefault(cls, []).append(gi)
                embedder_state.update(progress=min(start + _CHUNK, len(all_files)),
                                      skipped=skipped)

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
    # Validate + normalize the model path and feed the SESSION the validated
    # path. Previously the safe_model_file return was discarded and the raw
    # request string opened the ORT session, defeating the boundary check.
    model_path = safe_model_file(req.model_path)
    safe_paths = [safe_image_file(p) for p in req.img_paths]

    import onnxruntime as ort
    session = ort.InferenceSession(model_path)
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
    angle: float = 15.0       # Rotate degrees
    alpha: float = 1.3        # Brightness contrast scale
    beta: float = 30.0        # Brightness additive shift
    # Optional YOLO label dir. When set and the op supports label co-transform
    # (Flip/Rotate/Brightness), the boxes are transformed alongside the image
    # and returned, and preview_only flips to false.
    label_dir: str = ""


def _find_yolo_label(label_dir: str, img_dir: str, fp: str):
    """Locate a YOLO .txt label for image `fp`: prefer label_dir, else next to image."""
    stem = os.path.splitext(os.path.basename(fp))[0]
    candidates = []
    if label_dir:
        candidates.append(os.path.join(label_dir, stem + ".txt"))
    candidates.append(os.path.join(os.path.dirname(fp), stem + ".txt"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


# ── Batch: Augmentation Preview API ────────────────────
@router.post("/api/batch/augmentation")
async def batch_augmentation(req: BatchAugmentRequest):
    img_dir = safe_image_dir(req.img_dir)
    label_dir = safe_image_dir(req.label_dir) if req.label_dir else ""
    imgs = glob_images(img_dir)
    if not imgs:
        return {"error": "No images found"}
    import random
    from core import augmentation as aug_mod
    # imgs is non-empty here, so random.choice is safe.
    fp = random.choice(imgs)
    frame = imread(fp)
    if frame is None:
        return {"error": "Cannot read image"}
    aug_type = req.aug_type

    # Locate a YOLO label and parse boxes (None when no label / unsupported op).
    label_path = _find_yolo_label(label_dir, img_dir, fp)
    has_labels = (label_path is not None
                  and aug_type in aug_mod.LABEL_TRANSFORM_OPS)
    class_ids = np.zeros(0, dtype=np.int32)
    boxes = np.zeros((0, 4), dtype=np.float32)
    if has_labels:
        with open(label_path, "r", encoding="utf-8", errors="replace") as f:
            class_ids, boxes = aug_mod.parse_yolo_label(f.read())

    if aug_type == "Mosaic" or "Mosaic" in aug_type:
        # Mosaic composites 4 images — a single label set has no meaning, so it
        # stays an image-only preview. Resample distinct readable tiles (avoid
        # black quadrants); fall back to the base frame when the dir is tiny.
        h, w = frame.shape[:2]
        half_h, half_w = h // 2, w // 2
        readable = [frame]
        attempts = 0
        while len(readable) < 4 and attempts < 24:
            cand = imread(random.choice(imgs))
            if cand is not None:
                readable.append(cand)
            attempts += 1
        while len(readable) < 4:
            readable.append(frame)
        samples = [cv2.resize(s, (half_w, half_h)) for s in readable[:4]]
        top = np.hstack([samples[0], samples[1]])
        bot = np.hstack([samples[2], samples[3]])
        aug = np.vstack([top, bot])
        original = encode_jpeg(frame)
        return {"original": original, "augmented": encode_jpeg(aug),
                "file": os.path.basename(fp), "preview_only": True}

    # Single-image op: co-transform image + boxes together.
    aug_img, aug_boxes, aug_cids = aug_mod.augment_image_and_boxes(
        frame, boxes, class_ids, aug_type,
        angle=req.angle, alpha=req.alpha, beta=req.beta)

    # Draw boxes on both previews when labels were transformed.
    orig_vis = aug_mod.draw_boxes_on_image(frame, boxes) if has_labels else frame
    aug_vis = aug_mod.draw_boxes_on_image(aug_img, aug_boxes) if has_labels else aug_img

    resp = {
        "original": encode_jpeg(orig_vis),
        "augmented": encode_jpeg(aug_vis),
        "file": os.path.basename(fp),
        # preview_only=false only when labels were actually co-transformed.
        "preview_only": not has_labels,
    }
    if has_labels:
        # Transformed YOLO labels for the frontend to render/save.
        resp["labels"] = [
            {"class_id": int(c), "box": [round(float(v), 6) for v in b]}
            for c, b in zip(aug_cids, aug_boxes)
        ]
        resp["label_text"] = aug_mod.format_yolo_label(aug_cids, aug_boxes)
        resp["num_boxes"] = len(aug_boxes)
    return resp


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
    # Single-image mode: set image_path. Directory/batch mode: set image_dir
    # (image_path may be left empty). Both kept so existing single-image callers
    # are unchanged; image_dir is the additive batch entry point.
    image_path: str = ""
    image_dir: str = ""
    conf: float = 0.25
    model_type: Optional[str] = None
    kpt_conf: float = 0.5   # pose keypoint visibility threshold
    # Bound the directory scan so a huge folder can't exhaust memory in one
    # response (per-image overlays are heavy). 0 / negative → no extra cap
    # beyond max_images below.
    max_images: int = 200


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

    def _work():
        result = profile_model(path, num_runs=num_runs, warmup=warmup, provider=req.provider)
        return profile_to_dict(result)

    # profile_model does warmup + 50..500 sess.run + a second ORT profiling
    # session — tens of seconds on large models. Run off the event loop so it
    # doesn't pin every other tab's requests for the whole benchmark.
    return await asyncio.get_event_loop().run_in_executor(executor, _work)


# ── Phase 1: Pose Estimation API ───────────────────────
def _draw_pose(frame, result, kpt_conf):
    """Draw boxes + keypoints + skeleton onto a copy of the frame."""
    from core.inference import COCO_SKELETON
    vis = frame.copy()
    for i in range(len(result.boxes)):
        x1, y1, x2, y2 = result.boxes[i].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        kpts = result.keypoints[i]
        for kx, ky, kc in kpts:
            if kc > kpt_conf:
                cv2.circle(vis, (int(kx), int(ky)), 3, (0, 0, 255), -1)
        for a, b in COCO_SKELETON:
            if kpts[a][2] > kpt_conf and kpts[b][2] > kpt_conf:
                cv2.line(vis, (int(kpts[a][0]), int(kpts[a][1])),
                         (int(kpts[b][0]), int(kpts[b][1])), (255, 255, 0), 2)
    return vis


def _pose_result_dict(frame, result, kpt_conf):
    vis = _draw_pose(frame, result, kpt_conf)
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


@router.post("/api/infer/pose")
async def api_infer_pose(req: InferImageRequest):
    model_path = safe_model_file(req.model_path)
    model_type = req.model_type or "pose_yolo"
    # Directory/batch mode when image_dir is set; otherwise single-image mode.
    image_dir = safe_image_dir(req.image_dir) if req.image_dir else ""
    image_path = safe_image_file(req.image_path) if req.image_path else ""
    if not image_dir and not image_path:
        return {"error": "image_path or image_dir required"}

    def _work_single():
        frame = imread(image_path)
        if frame is None:
            return {"error": "Cannot read image"}
        mi = _load_model(model_path, model_type=model_type)
        from core.inference import run_pose
        result = run_pose(mi, frame, req.conf)
        return _pose_result_dict(frame, result, req.kpt_conf)

    def _work_batch():
        from core.inference import run_pose_batch
        files = glob_images(image_dir)
        if req.max_images and req.max_images > 0:
            files = files[:req.max_images]
        if not files:
            return {"error": "No images found"}
        mi = _load_model(model_path, model_type=model_type)
        frames, kept = [], []
        skipped = 0
        for fp in files:
            f = imread(fp)
            if f is None:
                skipped += 1
                continue
            frames.append(f)
            kept.append(fp)
        if not frames:
            return {"error": "No readable images"}
        results = run_pose_batch(mi, frames, req.conf)
        per_image = []
        for fp, frame, result in zip(kept, frames, results):
            item = _pose_result_dict(frame, result, req.kpt_conf)
            item["file"] = os.path.basename(fp)
            per_image.append(item)
        total_persons = sum(p["num_persons"] for p in per_image)
        return {"results": per_image, "count": len(per_image),
                "skipped": skipped, "total_persons": total_persons}

    # Model load + ORT run + cv2 drawing are blocking; offload off the event
    # loop so concurrent requests (status pollers, MJPEG stream) aren't frozen.
    work = _work_batch if image_dir else _work_single
    return await asyncio.get_event_loop().run_in_executor(executor, work)


# ── Phase 1: Instance Segmentation API ─────────────────
def _draw_instance_seg(frame, result):
    """Draw per-instance mask overlay + boxes + labels onto a copy of the frame."""
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
    return vis


def _instance_seg_result_dict(frame, result):
    vis = _draw_instance_seg(frame, result)
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


@router.post("/api/infer/instance-seg")
async def api_infer_instance_seg(req: InferImageRequest):
    model_path = safe_model_file(req.model_path)
    model_type = req.model_type or "instseg_yolo"
    image_dir = safe_image_dir(req.image_dir) if req.image_dir else ""
    image_path = safe_image_file(req.image_path) if req.image_path else ""
    if not image_dir and not image_path:
        return {"error": "image_path or image_dir required"}

    def _work_single():
        frame = imread(image_path)
        if frame is None:
            return {"error": "Cannot read image"}
        mi = _load_model(model_path, model_type=model_type)
        from core.inference import run_instance_seg
        result = run_instance_seg(mi, frame, req.conf)
        return _instance_seg_result_dict(frame, result)

    def _work_batch():
        from core.inference import run_instance_seg_batch
        files = glob_images(image_dir)
        if req.max_images and req.max_images > 0:
            files = files[:req.max_images]
        if not files:
            return {"error": "No images found"}
        mi = _load_model(model_path, model_type=model_type)
        frames, kept = [], []
        skipped = 0
        for fp in files:
            f = imread(fp)
            if f is None:
                skipped += 1
                continue
            frames.append(f)
            kept.append(fp)
        if not frames:
            return {"error": "No readable images"}
        results = run_instance_seg_batch(mi, frames, req.conf)
        per_image = []
        for fp, frame, result in zip(kept, frames, results):
            item = _instance_seg_result_dict(frame, result)
            item["file"] = os.path.basename(fp)
            per_image.append(item)
        total_instances = sum(p["num_instances"] for p in per_image)
        return {"results": per_image, "count": len(per_image),
                "skipped": skipped, "total_instances": total_instances}

    # ORT run + per-instance cv2.resize/overlay loop are blocking; offload off
    # the event loop so concurrent requests aren't frozen (same as pose).
    work = _work_batch if image_dir else _work_single
    return await asyncio.get_event_loop().run_in_executor(executor, work)


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


# ── Tracking: full-sequence runner (background job) ────
# The interactive /api/tracking/update endpoint above processes ONE typed frame
# at a time against a shared LRU tracker — it can't demonstrate tracking over a
# sequence. This job runs a whole folder of frames (or a video) through a
# per-JOB tracker (created fresh in the worker so concurrent runs never share a
# tracker) and reports stable IDs + trajectories.
tracking_state = TaskState()
all_states["tracking"] = tracking_state

# Bound the snapshot the status poller serializes: keep per-frame track rows
# (small) for every processed frame but only a handful of full overlay images
# (heavy base64 JPEGs). Evenly spaced previews give the UI a filmstrip without
# ballooning memory on a long sequence.
_TRACK_MAX_FRAMES = 1000
_TRACK_OVERLAY_PREVIEWS = 12
# Prune dead tracks every N frames so the tracker's list can't grow unbounded
# on very long sequences (update() prunes per frame too, but this is the
# explicit P2 memory bound from the backlog).
_TRACK_PRUNE_EVERY = 200


class TrackSequenceRequest(BaseModel):
    model_path: str
    model_type: Optional[str] = None
    # Provide EITHER image_dir (a folder of frames in sorted order) OR
    # video_path. image_dir takes precedence when both are set.
    image_dir: str = ""
    video_path: str = ""
    tracker: str = "bytetrack"     # "sort" | "bytetrack"
    conf: float = 0.25
    # Cap frames so an enormous folder/video can't run unbounded. Clamped to
    # _TRACK_MAX_FRAMES regardless of what the client requests.
    max_frames: int = 300


def _track_frame_source(image_dir: str, video_path: str, max_frames: int):
    """Yield (frame_idx, frame) from a frame folder or a video, lazily.

    Folder frames are read in sorted (glob_images) order. Video frames are
    decoded one at a time so memory stays flat regardless of sequence length.
    """
    if image_dir:
        files = glob_images(image_dir)[:max_frames]
        for idx, fp in enumerate(files):
            frame = imread(fp)
            if frame is None:
                continue
            yield idx, frame
        return
    cap = cv2.VideoCapture(video_path)
    try:
        idx = 0
        while idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


@router.post("/api/tracking/run-sequence")
async def api_tracking_run_sequence(req: TrackSequenceRequest):
    """Track a folder of frames (or a video) end-to-end as a background job."""
    # Validate paths synchronously at the boundary before submitting the worker.
    try:
        model_path = safe_model_file(req.model_path)
        image_dir = safe_image_dir(req.image_dir) if req.image_dir else ""
        video_path = safe_video_file(req.video_path) if req.video_path else ""
    except UnsafePathError as e:
        return {"error": str(e), "code": f"PATH_{e.code}"}
    if not image_dir and not video_path:
        return {"error": "image_dir or video_path required"}
    if req.tracker not in ("sort", "bytetrack"):
        return {"error": f"Unknown tracker '{req.tracker}' (expected sort|bytetrack)"}

    max_frames = max(1, min(req.max_frames, _TRACK_MAX_FRAMES))
    # Atomic start guard — a second POST while running returns the error.
    if not tracking_state.try_start(
            progress=0, total=max_frames, msg="Starting...",
            results={}, frames=[], previews=[], summary={}):
        return {"error": "Tracking already running"}

    @route_errors(state=tracking_state, scope="tracking")
    def _run():
        from core.tracking import create_tracker, draw_track_overlay, prune_dead_tracks
        # Per-JOB tracker: created fresh inside the worker so two concurrent
        # run-sequence jobs (or this job + the interactive /update tracker) never
        # mutate a shared tracker.
        tracker = create_tracker(req.tracker)
        mi = _load_model(model_path, model_type=req.model_type) if req.model_type \
            else _load_model(model_path)
        names = getattr(mi, "names", {}) or {}

        # Pick frame indices that should get a heavy overlay preview up front so
        # they're spaced evenly across the sequence (filmstrip, not first-N).
        preview_at = set(
            int(i * (max_frames - 1) / max(_TRACK_OVERLAY_PREVIEWS - 1, 1))
            for i in range(_TRACK_OVERLAY_PREVIEWS)
        ) if max_frames > 1 else {0}

        frame_rows: list[dict] = []
        previews: list[dict] = []
        unique_ids: set[int] = set()
        processed = 0
        for frame_idx, frame in _track_frame_source(image_dir, video_path, max_frames):
            # Cooperative stop: re-check running each frame so force-stop and a
            # second-tab stop break the loop promptly.
            if not tracking_state.get("running", True):
                tracking_state.update(msg="Stopped by user")
                break
            result = run_inference(mi, frame, req.conf)
            boxes = np.asarray(result.boxes, dtype=np.float32) \
                if len(result.boxes) else np.empty((0, 4), np.float32)
            scores = np.asarray(result.scores, dtype=np.float32) \
                if len(result.scores) else np.empty((0,), np.float32)
            class_ids = np.asarray(result.class_ids, dtype=np.int64) \
                if len(result.class_ids) else np.empty((0,), np.int64)
            tracks = tracker.update(boxes, scores, class_ids)
            for t in tracks:
                unique_ids.add(int(t.id))
            frame_rows.append({
                "frame_idx": frame_idx,
                "tracks": [
                    {"id": int(t.id), "box": [round(float(v), 1) for v in t.box],
                     "class_id": int(t.class_id), "score": round(float(t.score), 3)}
                    for t in tracks
                ],
            })
            if frame_idx in preview_at:
                vis = draw_track_overlay(frame, tracks, names=names)
                previews.append({"frame_idx": frame_idx, "image": encode_jpeg(vis)})
            processed += 1
            # Bound tracker memory on long runs (P2): explicit periodic prune.
            if processed % _TRACK_PRUNE_EVERY == 0:
                prune_dead_tracks(tracker)
            # Publish progress + a fresh shallow list (snapshot() is shallow, so
            # handing the poller a new list avoids racing in-place mutation).
            tracking_state.update(progress=processed,
                                  msg=f"Tracking frame {processed}/{max_frames}...",
                                  frames=list(frame_rows))

        summary = {
            "num_frames": processed,
            "num_unique_ids": len(unique_ids),
            "tracker": req.tracker,
            "source": "image_dir" if image_dir else "video",
        }
        stopped = not tracking_state.get("running", True)
        tracking_state.update(
            running=False,
            msg="Stopped" if stopped else "Complete",
            frames=list(frame_rows),
            previews=previews,
            summary=summary,
            results={"summary": summary, "frames": list(frame_rows),
                     "previews": previews})

    executor.submit(_run)
    return {"ok": True}


@router.get("/api/tracking/sequence-status")
async def api_tracking_sequence_status():
    return tracking_state.snapshot() if hasattr(tracking_state, "snapshot") \
        else dict(tracking_state)


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

# Overlays are heavy (one base64 PNG per image), so only the N worst-scoring
# images get one — that's where the user wants to look anyway. Bound the
# per-image metrics list too so a huge directory can't balloon the snapshot.
_SEG_OVERLAY_CAP = 12
_SEG_DETAIL_CAP = 1000


def _encode_png(img: np.ndarray) -> str:
    """Base64-encode a BGR image as PNG (lossless overlays for the modal)."""
    import base64
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode() if ok else ""


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

    # Atomic check-and-set — two concurrent POSTs would otherwise both submit
    # workers writing to the single seg_state. detail=[] / images=[] reset so a
    # prior run's per-image overlays don't bleed into this one.
    if not seg_state.try_start(progress=0, total=0, msg="Starting...",
                               results=[], detail=[], images=[], skipped=0):
        return {"error": "Segmentation evaluation already running"}

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
        # Per-image records carry the predicted/source masks so the worst-N can
        # be turned into overlays AFTER the loop (avoids encoding all images).
        per_image: list[dict] = []
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
            img_ious = []
            for k, v in metrics.items():
                if k == "__overall__":
                    continue
                all_ious.setdefault(k, []).append(v["iou"])
                all_dices.setdefault(k, []).append(v["dice"])
                if not np.isnan(v["iou"]) and v["gt_px"] > 0:
                    img_ious.append(v["iou"])
            img_iou = float(np.mean(img_ious)) if img_ious else 0.0
            gt_classes_present = int(len([c for c in np.unique(gt_mask) if c != 0]))
            if len(per_image) < _SEG_DETAIL_CAP:
                per_image.append({
                    "file": os.path.basename(fp),
                    "iou": round(img_iou, 4),
                    "classes": gt_classes_present,
                    # frame/pred kept only long enough to render worst-N overlays.
                    "_frame": frame,
                    "_pred": pred_mask,
                })
            evaluated += 1
            seg_state.update(progress=idx + 1,
                             msg=f"Evaluating {idx + 1}/{len(images)}...")

        # Average per-class IoU/Dice (NaN IoU = class absent in that image).
        per_class: list[dict] = []
        valid_ious, valid_dices = [], []
        for k in sorted(all_ious.keys()):
            ious = [x for x in all_ious[k] if not np.isnan(x)]
            dices = all_dices[k]
            mean_iou = float(np.mean(ious)) if ious else 0.0
            mean_dice = float(np.mean(dices)) if dices else 0.0
            per_class.append({
                "class": int(k),
                "iou": round(mean_iou, 6),
                "dice": round(mean_dice, 6),
            })
            if ious:
                valid_ious.append(mean_iou)
                valid_dices.append(mean_dice)
        miou = float(np.mean(valid_ious)) if valid_ious else 0.0
        mdice = float(np.mean(valid_dices)) if valid_dices else 0.0

        # Public per-image rows (no in-memory frame/pred — those were kept only
        # to render overlays). `path` mirrors the original spec field name.
        images_out: list[dict] = [
            {"path": rec["file"], "file": rec["file"], "iou": rec["iou"],
             "classes": rec["classes"], "overlay": ""}
            for rec in per_image
        ]
        # Overlay the worst-IoU images so the user lands on the failures first.
        by_file = {e["file"]: e for e in images_out}
        for rec in sorted(per_image, key=lambda r: r["iou"])[:_SEG_OVERLAY_CAP]:
            vis = overlay_segmentation(rec["_frame"], rec["_pred"])
            by_file[rec["file"]]["overlay"] = _encode_png(vis)

        results = {
            "metrics": {
                "mIoU": round(miou * 100, 4),
                "mDice": round(mdice * 100, 4),
                "per_class": per_class,
            },
            # Kept for backwards-compat with older UI that read these top-level.
            "mIoU": round(miou * 100, 4),
            "mDice": round(mdice * 100, 4),
            "num_images": evaluated,
        }
        msg = "Complete"
        if skipped:
            msg += f" ({skipped} image(s) skipped: no GT mask / unreadable)"
        # `images` (per-image rows + worst-N overlays) lives at the top level of
        # the status snapshot; `detail` mirrors it for the existing modal poller.
        seg_state.update(running=False, msg=msg, results=results,
                         images=images_out, detail=images_out, skipped=skipped)

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
