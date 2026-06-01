"""/api/model/*, /api/infer/*, /api/gt/*, /api/video/info 라우터."""
import asyncio
import base64
import os
import time as _time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from core.app_config import AppConfig
from core.inference import run_inference, run_classification, run_segmentation
from core.model_loader import load_model as _load_core
from server.errors import route_errors
from server.model_manager import ensure_model, get_model, get_model_meta, load_fresh
from server.path_safety import safe_label_dir, safe_image_dir, safe_image_file, safe_model_file
from server.state import executor, vlm_state
from server.utils import imread, generate_palette, get_color, draw_label, overlay_segmentation, glob_images

router = APIRouter()


# ── Pydantic models ─────────────────────────────────────
class ModelLoadRequest(BaseModel):
    path: str


class ModelClassesRequest(BaseModel):
    path: str
    model_type: str = "yolo"


class InferRequest(BaseModel):
    model_path: str
    image_path: Optional[str] = None
    conf: float = 0.25
    clip_labels: Optional[str] = None
    clip_text_encoder: Optional[str] = None
    # model_type overrides the global cfg.model_type for this request — the VLM
    # tab sends 'vlm' so we route to the VLM path instead of detection.
    model_type: Optional[str] = None
    vlm_prompt: Optional[str] = None
    vlm_task: Optional[str] = "caption"  # caption | vqa | grounding
    vlm_text_encoder: Optional[str] = None  # CLIP text encoder for VLM backend
    vlm_candidates: Optional[str] = None  # comma-separated answer candidates (VQA)
    # Pluggable VLM backend selection.
    backend: str = "clip"  # clip | transformers | openai
    model_id: Optional[str] = None  # HF repo / served model name (transformers, openai)
    endpoint_url: Optional[str] = None  # OpenAI-compatible server base URL (openai)
    api_key: Optional[str] = None  # bearer token (openai) — NEVER logged
    max_new_tokens: int = 128
    temperature: float = 0.0


class VideoInfoRequest(BaseModel):
    path: str


class VLMBatchRequest(BaseModel):
    model_path: str
    text_encoder: Optional[str] = None  # required only for the clip backend
    img_dir: str
    prompt: Optional[str] = None
    task: str = "caption"  # caption | vqa | grounding
    candidates: Optional[str] = None
    max_images: int = 50
    # Pluggable VLM backend selection (mirrors InferRequest).
    backend: str = "clip"  # clip | transformers | openai
    model_id: Optional[str] = None
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None  # bearer token (openai) — NEVER logged
    max_new_tokens: int = 128
    temperature: float = 0.0


# ── Model load / info ───────────────────────────────────
@router.post("/api/model/load")
async def load_model(req: ModelLoadRequest):
    return await asyncio.get_event_loop().run_in_executor(executor, _load_model_sync, req)


def _load_model_sync(req: ModelLoadRequest):
    try:
        cfg = AppConfig()
        return load_fresh(req.path, cfg.model_type, cfg)
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/model/info")
async def model_info():
    return {"loaded": get_model() is not None, "info": get_model_meta()}


@router.post("/api/model/classes")
async def model_classes(req: ModelClassesRequest):
    try:
        mi = _load_core(req.path, model_type=req.model_type)
        return {"names": mi.names or {}}
    except Exception as e:
        return {"error": str(e)}


class GtClassesRequest(BaseModel):
    label_dir: str


@router.post("/api/gt/classes")
async def gt_classes(req: GtClassesRequest):
    # Empty input → empty result (unchanged friendly fallback).
    if not req.label_dir:
        return {"classes": []}
    # Boundary check: must exist, must be a dir, no traversal.
    label_dir = safe_label_dir(req.label_dir)
    class_ids: set[int] = set()
    for f in os.listdir(label_dir):
        if not f.endswith(".txt"):
            continue
        with open(os.path.join(label_dir, f)) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_ids.add(int(parts[0]))
                    except ValueError:
                        pass
    return {"classes": sorted(class_ids)}


class ClassifyRequest(BaseModel):
    path: str


@router.post("/api/model/classify")
async def model_classify(req: ClassifyRequest):
    """Auto-detect task_type from an ONNX file's I/O signature.

    Used by the frontend's drag-and-drop loader to route the user to the
    right tab without forcing them to pick a model_type manually.
    """
    path = safe_model_file(req.path)
    from core.model_classifier import classify
    return classify(path)


@router.post("/api/model/find-partner")
async def model_find_partner(req: ClassifyRequest):
    """Find the matching CLIP partner encoder in the same directory.

    Removes a manual step in the VLM tab: drop the image encoder, and
    ssook auto-populates the text encoder field if it can find one.
    """
    path = safe_model_file(req.path)
    from core.model_pairing import find_partner
    return find_partner(path)


# ── Class label catalogues ──────────────────────────────
@router.get("/api/classes/catalog")
async def classes_catalog():
    """List built-in label catalogues (COCO80, VOC20, ImageNet1k...)."""
    from core.class_catalog import list_catalogs
    return {"catalogs": list_catalogs()}


@router.get("/api/classes/catalog/{name}")
async def classes_catalog_get(name: str):
    """Fetch the full label list for one catalogue."""
    from core.class_catalog import get, as_class_names
    labels = get(name)
    if labels is None:
        return {"error": f"Unknown catalog: {name}", "code": "CATALOG_UNKNOWN"}
    return {"name": name, "labels": labels, "class_names": as_class_names(name)}


class SuggestCatalogRequest(BaseModel):
    num_classes: int


@router.post("/api/classes/suggest")
async def classes_suggest(req: SuggestCatalogRequest):
    """Suggest a catalogue based on the model's class count.

    Frontend: after loading a model with no class_names, call this with
    the detected count and offer a 1-click apply.
    """
    from core.class_catalog import suggest, get
    name = suggest(req.num_classes)
    if name is None:
        return {"name": None, "labels": None,
                "msg": f"No built-in catalogue matches num_classes={req.num_classes}"}
    return {"name": name, "labels": get(name)}


@router.post("/api/model/infer-shapes")
async def infer_shapes(req: ModelLoadRequest):
    try:
        import onnxruntime as ort
        from core.inference import preprocess, preprocess_sequential
        session = ort.InferenceSession(req.path)
        inp = session.get_inputs()[0]
        h = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        w = int(inp.shape[3]) if isinstance(inp.shape[3], int) else 640
        bs = int(inp.shape[0]) if isinstance(inp.shape[0], int) and inp.shape[0] > 0 else 1
        in_ch = int(inp.shape[1]) if isinstance(inp.shape[1], int) else 3
        if in_ch == 9:
            dummy_frames = [np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(3)]
            tensor, _, _ = preprocess_sequential(dummy_frames, (h, w))
        else:
            dummy = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            tensor = preprocess(dummy, (h, w))
        if bs > 1:
            tensor = np.repeat(tensor, bs, axis=0)
        outputs = session.run(None, {inp.name: tensor})
        result = [{"index": i, "name": session.get_outputs()[i].name, "shape": list(o.shape)}
                  for i, o in enumerate(outputs)]
        return {"ok": True, "input_shape": list(inp.shape), "outputs": result}
    except Exception as e:
        return {"error": str(e)}


# ── Inference ────────────────────────────────────────────
@router.post("/api/infer/image")
async def infer_image(req: InferRequest):
    return await asyncio.get_event_loop().run_in_executor(executor, _infer_image_sync, req)


def _vlm_spec(req: InferRequest) -> dict:
    """Build a make_backend spec from the request, choosing the right encoder.

    vlm_text_encoder takes precedence over clip_text_encoder for the legacy
    CLIP field name.
    """
    return {
        "backend": req.backend,
        "model_path": req.model_path,
        "text_encoder": req.vlm_text_encoder or req.clip_text_encoder,
        "model_id": req.model_id,
        "endpoint_url": req.endpoint_url,
        "api_key": req.api_key,
    }


def _infer_image_vlm(req: InferRequest):
    """Single-image VLM path. Validates paths up front, then dispatches to the
    selected backend. Reached when model_type resolves to 'vlm'.
    """
    from core.vlm_inference import make_backend
    backend_name = (req.backend or "clip").lower()
    # Path safety FIRST — before any imread/session — and use the safe paths.
    try:
        safe_img = safe_image_file(req.image_path) if req.image_path else None
        # Only the clip backend points model_path/text_encoder at local ONNX
        # files; transformers uses an HF id, openai uses an HTTP endpoint.
        if backend_name == "clip":
            safe_model_file(req.model_path)
            text_encoder = req.vlm_text_encoder or req.clip_text_encoder
            if text_encoder:
                safe_model_file(text_encoder)
    except Exception as e:
        return {"error": f"Invalid path: {e}"}

    if not safe_img:
        return {"error": "image_path is required for VLM inference"}
    frame = imread(safe_img)
    if frame is None:
        return {"error": "Cannot read image"}

    t0 = _time.perf_counter()
    try:
        backend = make_backend(_vlm_spec(req))
        task = (req.vlm_task or "caption").lower()
        prompt = req.vlm_prompt or ""
        if task == "vqa":
            cands = [c for c in (req.vlm_candidates or "").split(",") if c.strip()]
            text_result = backend.answer(frame, prompt, candidates=cands or None,
                                         max_new_tokens=req.max_new_tokens,
                                         temperature=req.temperature)
            overlay = f"A: {text_result[:60]}"
        else:  # caption / grounding (grounding falls back to caption)
            text_result = backend.describe(frame, prompt,
                                           max_new_tokens=req.max_new_tokens,
                                           temperature=req.temperature)
            overlay = text_result[:80]
    except Exception as e:
        return {"error": f"VLM inference failed: {e}"}
    infer_ms = (_time.perf_counter() - t0) * 1000
    vis = frame.copy()
    cv2.putText(vis, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"image": base64.b64encode(buf).decode(), "detections": 0,
            "vlm_result": text_result, "vlm_task": task,
            "infer_ms": round(infer_ms, 2), "classes": {}}


def _infer_image_sync(req: InferRequest):
    try:
        cfg = AppConfig()
        # Per-request model_type wins over the global cfg default; the VLM tab
        # sends 'vlm' so we route to the VLM path instead of detection.
        model_type = req.model_type or cfg.model_type
        if model_type == "vlm":
            return _infer_image_vlm(req)
        model = ensure_model(req.model_path, model_type, cfg)
        frame = imread(req.image_path)
        if frame is None:
            return {"error": "Cannot read image"}
        names = model.names or {}

        if model.task_type == "embedding":
            from core.clip_inference import CLIPModel, simple_tokenize
            t0 = _time.perf_counter()
            clip_model = CLIPModel(req.model_path, req.clip_text_encoder or None)
            if req.clip_labels:
                labels = [l.strip() for l in req.clip_labels.split(',') if l.strip()]
                text_embs = [clip_model.encode_text(simple_tokenize(l)) for l in labels]
                ranked = clip_model.zero_shot_classify(frame, text_embs, labels)
                infer_ms = (_time.perf_counter() - t0) * 1000
                vis = frame.copy()
                y = 30
                for label, score in ranked[:5]:
                    cv2.putText(vis, f"{label}: {score*100:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y += 28
                _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return {"image": base64.b64encode(buf).decode(), "detections": 0,
                        "clip_result": [{"label": l, "score": round(s, 4)} for l, s in ranked],
                        "infer_ms": round(infer_ms, 2), "classes": {}}
            else:
                emb = clip_model.encode_image(frame)
                infer_ms = (_time.perf_counter() - t0) * 1000
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return {"image": base64.b64encode(buf).decode(), "detections": 0,
                        "embedding": f"dim={len(emb)}, norm={float(np.linalg.norm(emb)):.4f}",
                        "infer_ms": round(infer_ms, 2), "classes": {}}

        if model.task_type == "vlm":
            # ONNX model auto-detected as VLM but the request omitted
            # model_type='vlm'; route through the unified backend path.
            return _infer_image_vlm(req)

        if model.task_type == "classification":
            result = run_classification(model, frame)
            top_k = result.top_k[:5]
            vis = frame.copy()
            y = 30
            for cid, conf in top_k:
                cv2.putText(vis, f"{names.get(cid, str(cid))}: {conf:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y += 30
            _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            best = names.get(result.class_id, str(result.class_id))
            return {"image": base64.b64encode(buf).decode(), "detections": 0,
                    "classification": f"{best} ({result.confidence:.3f})",
                    "top_k": [{"class": names.get(c, str(c)), "score": round(s, 4)} for c, s in top_k],
                    "infer_ms": round(result.infer_ms, 2), "classes": {}}

        if model.task_type == "segmentation":
            result = run_segmentation(model, frame)
            vis = overlay_segmentation(frame, result.mask)
            _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            unique_cls = [int(c) for c in np.unique(result.mask) if c > 0]
            return {"image": base64.b64encode(buf).decode(), "detections": len(unique_cls),
                    "segmentation": f"{result.num_classes} classes, {len(unique_cls)} present",
                    "infer_ms": round(result.infer_ms, 2),
                    "classes": {c: names.get(c, str(c)) for c in unique_cls}}

        # Detection (default)
        result = run_inference(model, frame, cfg.conf_threshold)
        thickness = cfg.box_thickness
        label_size = cfg.label_size
        total_cls = len(names)
        for box, score, cid in zip(result.boxes, result.scores, result.class_ids):
            cid_int = int(cid)
            style = cfg.get_class_style(cid_int)
            if not style.enabled:
                continue
            x1, y1, x2, y2 = map(int, box)
            t_val = style.thickness or thickness
            color = get_color(style, cid_int, total_cls)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, t_val)
            parts = []
            if cfg.show_labels:
                parts.append(names.get(cid_int, str(cid_int)))
            if cfg.show_confidence:
                parts.append(f"{score:.2f}")
            if parts:
                draw_label(frame, " ".join(parts), x1, y1, color, label_size, max(1, t_val - 1), cfg.show_label_bg)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return {"image": base64.b64encode(buf).decode(), "detections": len(result.boxes),
                "infer_ms": round(result.infer_ms, 2),
                "classes": {int(cid): names.get(int(cid), str(int(cid)))
                            for cid in np.unique(result.class_ids)} if len(result.class_ids) else {}}
    except Exception as e:
        return {"error": str(e)}


# ── VLM batch (async) ────────────────────────────────────
@router.post("/api/vlm/batch")
async def vlm_batch(req: VLMBatchRequest):
    """Run VLM (caption/VQA) over a folder in the background.

    Status is exposed via GET /api/vlm/status. Force-stop via
    POST /api/force-stop/vlm.
    """
    if vlm_state.get("running"):
        return {"error": "VLM batch already running", "code": "VLM_BUSY"}
    backend_name = (req.backend or "clip").lower()
    # Path safety: only the clip backend reads local ONNX files; transformers
    # uses an HF id and openai an HTTP endpoint.
    model_path = req.model_path
    text_encoder = req.text_encoder
    if backend_name == "clip":
        model_path = safe_model_file(req.model_path)
        if not req.text_encoder:
            return {"error": "clip backend requires text_encoder", "code": "VLM_CONFIG"}
        text_encoder = safe_model_file(req.text_encoder)
    img_dir = safe_image_dir(req.img_dir)
    imgs = glob_images(img_dir)
    if not imgs:
        return {"error": "No images found", "code": "EMPTY_DIR"}
    imgs = imgs[: max(1, min(req.max_images, 500))]

    spec = {
        "backend": backend_name,
        "model_path": model_path,
        "text_encoder": text_encoder,
        "model_id": req.model_id,
        "endpoint_url": req.endpoint_url,
        "api_key": req.api_key,
    }
    vlm_state.update(running=True, progress=0, total=len(imgs), msg="Starting...", results=[])

    @route_errors(state=vlm_state, scope="vlm")
    def _run():
        from core.vlm_inference import make_backend
        from core.run_record import RunRecorder
        from core.paths import tmp_dir
        out_dir = tmp_dir("vlm")
        # NOTE: api_key is intentionally excluded from the run record inputs.
        with RunRecorder(run_type="vlm_batch", output_dir=out_dir,
                         inputs={"backend": backend_name, "task": req.task,
                                 "prompt": req.prompt, "candidates": req.candidates,
                                 "model_id": req.model_id,
                                 "endpoint_url": req.endpoint_url,
                                 "img_dir": img_dir, "max_images": len(imgs)},
                         model_path=model_path if backend_name == "clip" else None) as rec:
            vlm_state["trace_id"] = rec.trace_id
            backend = make_backend(spec)
            task = (req.task or "caption").lower()
            cands = [c for c in (req.candidates or "").split(",") if c.strip()]
            results = []
            for i, fp in enumerate(imgs):
                if not vlm_state.get("running", True):
                    rec.note(f"Stopped at {i}/{len(imgs)}")
                    vlm_state.update(msg="Stopped by user")
                    break
                frame = imread(fp)
                if frame is None:
                    vlm_state["progress"] = i + 1
                    continue
                t0 = _time.perf_counter()
                try:
                    if task == "vqa":
                        text = backend.answer(frame, req.prompt or "", candidates=cands or None,
                                              max_new_tokens=req.max_new_tokens,
                                              temperature=req.temperature)
                    else:
                        text = backend.describe(frame, req.prompt or "",
                                                max_new_tokens=req.max_new_tokens,
                                                temperature=req.temperature)
                except Exception as e:
                    text = f"(error: {e})"
                ms = round((_time.perf_counter() - t0) * 1000, 1)
                results.append({"file": os.path.basename(fp), "result": text, "ms": ms})
                vlm_state["progress"] = i + 1
                vlm_state["msg"] = f"{i+1}/{len(imgs)}"
                vlm_state["results"] = results
            vlm_state.update(running=False, msg="Complete")

    executor.submit(_run)
    return {"ok": True, "total": len(imgs)}


@router.get("/api/vlm/backends")
async def vlm_backends():
    """List available VLM backends so the UI can grey out missing deps.

    `cuda` reports whether the transformers backend can run on GPU; guarded
    because torch is an optional dependency.
    """
    from core.vlm_inference import list_backends
    cuda = False
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    return {"backends": list_backends(), "cuda": cuda}


@router.get("/api/vlm/status")
async def vlm_status():
    return vlm_state.snapshot() if hasattr(vlm_state, "snapshot") else dict(vlm_state)


# ── Video Info ───────────────────────────────────────────
@router.post("/api/video/info")
async def video_info(req: VideoInfoRequest):
    try:
        ext = os.path.splitext(req.path)[1].lower()
        if ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            frame = imread(req.path)
            if frame is None:
                return {"error": "Cannot read image"}
            h, w = frame.shape[:2]
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return {"width": w, "height": h, "fps": 0, "total_frames": 1, "duration": "0:00",
                    "first_frame": base64.b64encode(buf).decode()}
        cap = cv2.VideoCapture(req.path)
        if not cap.isOpened():
            return {"error": "Cannot open video"}
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        first_frame_b64 = None
        ret, frame = cap.read()
        if ret:
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            first_frame_b64 = base64.b64encode(buf).decode()
        cap.release()
        dur_s = total / fps if fps > 0 else 0
        mins, secs = divmod(int(dur_s), 60)
        return {"width": w, "height": h, "fps": round(fps, 2), "total_frames": total,
                "duration": f"{mins}:{secs:02d}", "first_frame": first_frame_b64}
    except Exception as e:
        return {"error": str(e)}
