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
from server.model_manager import ensure_model, get_model, get_model_meta, load_fresh
from server.state import executor
from server.utils import imread, generate_palette, get_color, draw_label, overlay_segmentation

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
    vlm_prompt: Optional[str] = None


class VideoInfoRequest(BaseModel):
    path: str


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


@router.post("/api/gt/classes")
async def gt_classes(req: dict):
    label_dir = req.get("label_dir", "")
    if not os.path.isdir(label_dir):
        return {"classes": []}
    class_ids = set()
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


@router.post("/api/model/infer-shapes")
async def infer_shapes(req: ModelLoadRequest):
    try:
        import onnxruntime as ort
        from core.inference import preprocess
        session = ort.InferenceSession(req.path)
        inp = session.get_inputs()[0]
        h = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        w = int(inp.shape[3]) if isinstance(inp.shape[3], int) else 640
        bs = int(inp.shape[0]) if isinstance(inp.shape[0], int) and inp.shape[0] > 0 else 1
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


def _infer_image_sync(req: InferRequest):
    try:
        cfg = AppConfig()
        model = ensure_model(req.model_path, cfg.model_type, cfg)
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
            vis = frame.copy()
            prompt = req.vlm_prompt or "Describe this image."
            cv2.putText(vis, f"VLM: {prompt[:60]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return {"image": base64.b64encode(buf).decode(), "detections": 0,
                    "vlm_result": f"VLM inference not yet implemented (prompt: {prompt})",
                    "infer_ms": 0, "classes": {}}

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
