"""
ssook Web Server — FastAPI backend serving the web UI
and exposing core/ functionality as REST API.
"""
import os
import sys
import platform
import asyncio
import base64
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _generate_palette(n):
    """HSV 균등 분포로 n개의 BGR 색상 생성"""
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        hsv = np.uint8([[[hue, 220, 220]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(x) for x in bgr))
    return colors


_palette_cache = []


def _get_color(style, cid, total):
    """Return BGR color for a class: style.color > palette > green fallback."""
    global _palette_cache
    if style.color:
        return tuple(style.color)
    if total > 0:
        if len(_palette_cache) < total:
            _palette_cache = _generate_palette(total)
        return _palette_cache[cid % len(_palette_cache)]
    return (0, 255, 0)


def _draw_label(frame, text, x1, y1, color, font_scale, font_thick, show_bg):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
    ty = max(y1 - 4, th + 4)
    if show_bg:
        cv2.rectangle(frame, (x1, ty - th - baseline - 2), (x1 + tw + 2, ty + 2), color, -1)
        lum = color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299
        txt_color = (0, 0, 0) if lum > 128 else (255, 255, 255)
    else:
        txt_color = color
    cv2.putText(frame, text, (x1 + 1, ty - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, font_thick, cv2.LINE_AA)

app = FastAPI(title="ssook", version="1.1.0")

# ── Static files ────────────────────────────────────────
WEB_DIR = ROOT / "web"
app.mount("/css", StaticFiles(directory=WEB_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=WEB_DIR / "js"), name="js")
app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


# ── Config API ──────────────────────────────────────────
@app.get("/api/config")
async def get_config():
    try:
        from core.app_config import AppConfig
        cfg = AppConfig()
        # 클래스별 스타일 (#5)
        cs = {}
        for cid, s in cfg.class_styles.items():
            cs[str(cid)] = {
                "enabled": s.enabled,
                "color": list(s.color) if s.color else None,
                "thickness": s.thickness,
            }
        mt = {k: v for k, v in _get_model_types().items()}
        for name in cfg.custom_model_types:
            mt[f"custom:{name}"] = name
        return {
            "model_type": cfg.model_type,
            "conf_threshold": cfg.conf_threshold,
            "batch_size": cfg.batch_size,
            "box_thickness": cfg.box_thickness,
            "label_size": cfg.label_size,
            "show_labels": cfg.show_labels,
            "show_confidence": cfg.show_confidence,
            "default_model_path": cfg.default_model_path,
            "model_types": mt,
            "class_styles": cs,
        }
    except Exception as e:
        return {"error": str(e)}


def _get_model_types():
    from core.model_loader import MODEL_TYPES
    return MODEL_TYPES


class ConfigUpdate(BaseModel):
    conf_threshold: Optional[float] = None
    model_type: Optional[str] = None
    batch_size: Optional[int] = None
    box_thickness: Optional[int] = None
    label_size: Optional[float] = None
    show_labels: Optional[bool] = None
    show_confidence: Optional[bool] = None
    show_label_bg: Optional[bool] = None
    default_model_path: Optional[str] = None


@app.post("/api/config")
async def save_config(cfg: ConfigUpdate):
    try:
        from core.app_config import AppConfig
        app_cfg = AppConfig()
        for k, v in cfg.dict(exclude_none=True).items():
            setattr(app_cfg, k, v)
        app_cfg.save()
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}


# ── Class Styles API (#5) ───────────────────────────────
class ClassStyleUpdate(BaseModel):
    class_id: int
    enabled: Optional[bool] = None
    color: Optional[list] = None       # [B, G, R] or null
    thickness: Optional[int] = None


@app.post("/api/config/class-style")
async def save_class_style(req: ClassStyleUpdate):
    try:
        from core.app_config import AppConfig, ClassStyle
        cfg = AppConfig()
        style = cfg.get_class_style(req.class_id)
        if req.enabled is not None:
            style.enabled = req.enabled
        if req.color is not None:
            style.color = tuple(req.color) if req.color else None
        if req.thickness is not None:
            style.thickness = req.thickness if req.thickness > 0 else None
        cfg.set_class_style(req.class_id, style)
        cfg.save()
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}


# ── Custom Model Type API (#4) ──────────────────────────
class CustomModelTypeRequest(BaseModel):
    name: str
    model_path: str
    output_index: int = 0
    attr_roles: list = []
    dim_roles: list = []
    has_objectness: bool = False
    nms: bool = True
    conf_threshold: float = 0.25
    class_names: Optional[dict] = None


@app.post("/api/config/custom-model-type")
async def save_custom_model_type(req: CustomModelTypeRequest):
    try:
        from core.app_config import AppConfig, CustomModelType
        cfg = AppConfig()
        cmt = CustomModelType(
            name=req.name,
            output_index=req.output_index,
            dim_roles=req.dim_roles,
            attr_roles=req.attr_roles,
            has_objectness=req.has_objectness,
            nms=req.nms,
            conf_threshold=req.conf_threshold,
            class_names={int(k): v for k, v in req.class_names.items()} if req.class_names else None,
        )
        cfg.custom_model_types[req.name] = cmt
        cfg.save()
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/config/custom-model-type/test")
async def test_custom_model_type(req: CustomModelTypeRequest):
    """테스트 추론: 모델 로드 → 더미 이미지 → custom postprocess → 결과 반환"""
    try:
        import onnxruntime as ort
        from core.inference import letterbox, preprocess, postprocess_custom
        from core.app_config import CustomModelType
        session = ort.InferenceSession(req.model_path)
        inp = session.get_inputs()[0]
        h = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        w = int(inp.shape[3]) if isinstance(inp.shape[3], int) else 640
        # 더미 이미지
        dummy = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        _, ratio, pad = letterbox(dummy, (h, w))
        tensor = preprocess(dummy, (h, w))
        outputs = session.run(None, {inp.name: tensor})
        cmt = CustomModelType(
            name=req.name, output_index=req.output_index,
            dim_roles=req.dim_roles, attr_roles=req.attr_roles,
            has_objectness=req.has_objectness, nms=req.nms,
            conf_threshold=req.conf_threshold,
        )
        result = postprocess_custom(outputs, cmt, req.conf_threshold, ratio, pad, dummy.shape)
        shapes = [str(list(o.shape)) for o in outputs]
        return {"ok": True, "detections": len(result.boxes), "output_shapes": shapes}
    except Exception as e:
        return {"error": str(e)}



@app.get("/api/config/custom-model-types")
async def list_custom_model_types():
    from core.app_config import AppConfig
    cfg = AppConfig()
    return {name: {"attr_roles": cmt.attr_roles, "dim_roles": cmt.dim_roles,
                    "class_names": cmt.class_names}
            for name, cmt in cfg.custom_model_types.items()}


# ── Evaluation async API (#6, #7) ──────────────────────
_eval_state = {"running": False, "progress": 0, "total": 0, "msg": "",
               "model_name": "", "results": []}


class EvalAsyncRequest(BaseModel):
    models: list                       # [{path, model_type, class_mapping?}, ...]
    img_dir: str
    label_dir: str
    conf: float = 0.25
    class_mapping: Optional[dict] = None  # {gt_id: name, ...}
    per_model_mappings: Optional[dict] = None  # {model_name: {model_cls_id: gt_cls_id}}
    mapped_only: bool = True


@app.post("/api/evaluation/run-async")
async def run_evaluation_async(req: EvalAsyncRequest):
    """비동기 평가 실행 (#6 pbar + #7 모델타입/클래스 지정)"""
    if _eval_state["running"]:
        return {"error": "Evaluation already running"}

    _eval_state.update(running=True, progress=0, total=1, msg="Starting...", model_name="", results=[])

    def _run():
        import glob
        from core.model_loader import load_model as _load_model
        from core.inference import run_inference
        from core.evaluation import evaluate_dataset, evaluate_map50_95

        _eval_state.update(progress=0, msg="Loading images...")

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        img_files = []
        for e in exts:
            img_files.extend(glob.glob(os.path.join(req.img_dir, e)))
        img_files.sort()
        if not img_files:
            _eval_state.update(running=False, msg="No images found")
            return

        # GT 로드
        gt_data = {}
        for fp in img_files:
            stem = os.path.splitext(os.path.basename(fp))[0]
            txt = os.path.join(req.label_dir, stem + ".txt")
            boxes = []
            if os.path.isfile(txt):
                with open(txt) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            boxes.append((int(parts[0]), *map(float, parts[1:5])))
            gt_data[stem] = boxes

        total_work = len(img_files) * len(req.models)
        _eval_state["total"] = total_work
        done = 0

        for entry in req.models:
            model_path = entry if isinstance(entry, str) else entry.get("path", "")
            model_type = "yolo" if isinstance(entry, str) else entry.get("model_type", "yolo")
            name = os.path.basename(model_path)
            _eval_state["model_name"] = name

            try:
                mi = _load_model(model_path, model_type=model_type)
            except Exception as exc:
                _eval_state["results"].append({"name": name, "error": str(exc)})
                done += len(img_files)
                _eval_state["progress"] = done
                continue

            # per-model class mapping
            pm = (req.per_model_mappings or {}).get(name, {})
            # convert string keys to int
            mapping = {int(k): int(v) for k, v in pm.items()} if pm else {}
            mapped_only = req.mapped_only

            # GT 필터링 (매핑된 클래스만)
            if mapping and mapped_only:
                allowed_gt = set(mapping.values())
                gt_eval = {s: [b for b in boxes if b[0] in allowed_gt]
                           for s, boxes in gt_data.items()}
            else:
                gt_eval = gt_data

            pred_data = {}
            for fp in img_files:
                frame = cv2.imread(fp)
                if frame is None:
                    done += 1
                    continue
                h, w = frame.shape[:2]
                res = run_inference(mi, frame, req.conf)
                stem = os.path.splitext(os.path.basename(fp))[0]
                boxes = []
                for box, score, cid in zip(res.boxes, res.scores, res.class_ids):
                    cid = int(cid)
                    # 클래스 리매핑
                    if mapping:
                        if cid in mapping:
                            cid = mapping[cid]
                        elif mapped_only:
                            continue
                    x1, y1, x2, y2 = box
                    cx = ((x1+x2)/2)/w; cy = ((y1+y2)/2)/h
                    bw = (x2-x1)/w; bh = (y2-y1)/h
                    boxes.append((cid, cx, cy, bw, bh, float(score)))
                pred_data[stem] = boxes
                done += 1
                _eval_state["progress"] = done
                _eval_state["msg"] = f"{name}: {done}/{total_work}"

            res50 = evaluate_dataset(gt_eval, pred_data, 0.5)
            map5095 = evaluate_map50_95(gt_eval, pred_data)
            ov = res50.get("__overall__", {})
            # per-class detail (JSON serializable)
            detail = {}
            for cid, v in res50.items():
                if cid == "__overall__":
                    continue
                detail[str(cid)] = {
                    "ap": round(v.get("ap", 0), 6),
                    "precision": round(v.get("precision", 0), 6),
                    "recall": round(v.get("recall", 0), 6),
                    "f1": round(v.get("f1", 0), 6),
                    "tp": v.get("tp", 0), "fp": v.get("fp", 0), "fn": v.get("fn", 0),
                }
            _eval_state["results"].append({
                "name": name,
                "map50": round(ov.get("ap", 0) * 100, 4),
                "map5095": round(map5095 * 100, 4),
                "precision": round(ov.get("precision", 0) * 100, 4),
                "recall": round(ov.get("recall", 0) * 100, 4),
                "f1": round(ov.get("f1", 0) * 100, 4),
                "detail": detail,
            })

        _eval_state.update(running=False, msg="Complete")

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True}


@app.get("/api/evaluation/status")
async def evaluation_status():
    return {
        "running": _eval_state["running"],
        "progress": _eval_state["progress"],
        "total": _eval_state["total"],
        "msg": _eval_state["msg"],
        "model_name": _eval_state["model_name"],
        "results": _eval_state["results"],
    }


# ── Model API ───────────────────────────────────────────
class ModelLoadRequest(BaseModel):
    path: str


@app.post("/api/model/load")
async def load_model(req: ModelLoadRequest):
    global _loaded_model, _loaded_model_meta
    try:
        from core.model_loader import load_model as _load
        from core.app_config import AppConfig
        cfg = AppConfig()
        model_type = cfg.model_type
        custom_name = ""
        if model_type.startswith("custom:"):
            custom_name = model_type.split(":", 1)[1]
            model_type = "custom"
        info = _load(req.path, model_type=model_type)
        if custom_name:
            info.custom_type_name = custom_name
            cmt = cfg.custom_model_types.get(custom_name)
            if cmt and cmt.class_names:
                info.names = cmt.class_names
        _loaded_model = info
        inp = info.session.get_inputs()[0]
        out = info.session.get_outputs()[0]
        meta = {
            "ok": True,
            "name": os.path.basename(req.path),
            "input_shape": str(inp.shape),
            "output_shape": str(out.shape),
            "input_size": list(info.input_size) if info.input_size else None,
            "num_classes": len(info.names) if info.names else 0,
            "names": info.names or {},
            "task": info.task_type or "",
            "layout": info.output_layout or "",
            "model_type": info.model_type or "",
            "batch_size": info.batch_size,
        }
        _loaded_model_meta = meta
        return meta
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/model/info")
async def model_info():
    return {"loaded": _loaded_model is not None, "info": _loaded_model_meta}


class ModelClassesRequest(BaseModel):
    path: str
    model_type: str = "yolo"


@app.post("/api/model/classes")
async def model_classes(req: ModelClassesRequest):
    """모델의 클래스 이름 목록 반환 (전역 모델 변경 없음)"""
    try:
        from core.model_loader import load_model as _load
        mi = _load(req.path, model_type=req.model_type)
        return {"names": mi.names or {}}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/gt/classes")
async def gt_classes(req: dict):
    """GT 라벨 폴더에서 고유 클래스 ID 스캔"""
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


@app.post("/api/model/infer-shapes")
async def infer_shapes(req: ModelLoadRequest):
    """모델을 더미 입력으로 실행하여 실제 출력 shape 반환"""
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
        result = []
        for i, o in enumerate(outputs):
            result.append({"index": i, "name": session.get_outputs()[i].name, "shape": list(o.shape)})
        return {"ok": True, "input_shape": list(inp.shape), "outputs": result}
    except Exception as e:
        return {"error": str(e)}


# ── Shared model state ──────────────────────────────────
_loaded_model = None      # ModelInfo object
_loaded_model_meta = None  # dict for JSON response


# ── Auto-load default model on startup (#11) ────────────
@app.on_event("startup")
async def _auto_load_default_model():
    global _loaded_model, _loaded_model_meta
    try:
        from core.app_config import AppConfig
        from core.model_loader import load_model as _load
        cfg = AppConfig()
        p = cfg.default_model_path
        if p and os.path.isfile(p):
            model_type = cfg.model_type
            if model_type.startswith("custom:"):
                model_type = "custom"
            info = _load(p, model_type=model_type)
            _loaded_model = info
            inp = info.session.get_inputs()[0]
            out = info.session.get_outputs()[0]
            _loaded_model_meta = {
                "ok": True, "name": os.path.basename(p),
                "input_shape": str(inp.shape), "output_shape": str(out.shape),
                "input_size": list(info.input_size), "num_classes": len(info.names or {}),
                "names": info.names or {}, "task": info.task_type or "",
                "layout": info.output_layout or "", "model_type": info.model_type or "",
                "batch_size": info.batch_size,
            }
            print(f"[Startup] Default model loaded: {os.path.basename(p)}")
    except Exception as e:
        print(f"[Startup] Default model load failed: {e}")


# ── Inference API ───────────────────────────────────────
class InferRequest(BaseModel):
    model_path: str
    image_path: Optional[str] = None
    conf: float = 0.25


@app.post("/api/infer/image")
async def infer_image(req: InferRequest):
    """Run inference on a single image, return annotated JPEG + detections."""
    global _loaded_model, _loaded_model_meta
    try:
        from core.model_loader import load_model as _load
        from core.inference import run_inference, run_classification
        from core.app_config import AppConfig

        cfg = AppConfig()
        model_type = cfg.model_type
        custom_name = ""
        if model_type.startswith("custom:"):
            custom_name = model_type.split(":", 1)[1]
            model_type = "custom"
        if _loaded_model is None or _loaded_model.path != req.model_path or _loaded_model.model_type != model_type:
            _loaded_model = _load(req.model_path, model_type=model_type)
            if custom_name:
                _loaded_model.custom_type_name = custom_name
                cmt = cfg.custom_model_types.get(custom_name)
                if cmt and cmt.class_names:
                    _loaded_model.names = cmt.class_names
            _loaded_model_meta = {"name": os.path.basename(req.model_path)}

        frame = cv2.imread(req.image_path)
        if frame is None:
            return {"error": "Cannot read image"}

        names = _loaded_model.names or {}

        if _loaded_model.task_type == "classification":
            result = run_classification(_loaded_model, frame)
            top_k = result.top_k[:5]
            y = 30
            vis = frame.copy()
            for cid, conf in top_k:
                label = f"{names.get(cid, str(cid))}: {conf:.3f}"
                cv2.putText(vis, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y += 30
            _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            best = names.get(result.class_id, str(result.class_id))
            return {
                "image": base64.b64encode(buf).decode(),
                "detections": 0,
                "classification": f"{best} ({result.confidence:.3f})",
                "top_k": [{"class": names.get(c, str(c)), "score": round(s, 4)} for c, s in top_k],
                "infer_ms": round(result.infer_ms, 2),
                "classes": {},
            }

        # Detection
        result = run_inference(_loaded_model, frame, cfg.conf_threshold)
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
            color = _get_color(style, cid_int, total_cls)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, t_val)
            parts = []
            if cfg.show_labels:
                parts.append(names.get(cid_int, str(cid_int)))
            if cfg.show_confidence:
                parts.append(f"{score:.2f}")
            if parts:
                _draw_label(frame, " ".join(parts), x1, y1, color, label_size, max(1, t_val - 1), cfg.show_label_bg)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return {
            "image": base64.b64encode(buf).decode(),
            "detections": len(result.boxes),
            "infer_ms": round(result.infer_ms, 2),
            "classes": {int(cid): names.get(int(cid), str(int(cid)))
                        for cid in np.unique(result.class_ids)} if len(result.class_ids) else {},
        }
    except Exception as e:
        return {"error": str(e)}


# ── Video streaming (MJPEG) ─────────────────────────────
_video_sessions = {}  # session_id -> dict with state


class VideoStartRequest(BaseModel):
    model_path: str
    video_path: str
    conf: float = 0.25


@app.post("/api/viewer/start")
async def viewer_start(req: VideoStartRequest):
    """Start a video inference session, returns session_id."""
    global _loaded_model, _loaded_model_meta
    try:
        from core.model_loader import load_model as _load
        from core.app_config import AppConfig
        cfg = AppConfig()
        model_type = cfg.model_type
        custom_name = ""
        if model_type.startswith("custom:"):
            custom_name = model_type.split(":", 1)[1]
            model_type = "custom"
        if _loaded_model is None or _loaded_model.path != req.model_path or _loaded_model.model_type != model_type:
            _loaded_model = _load(req.model_path, model_type=model_type)
            if custom_name:
                _loaded_model.custom_type_name = custom_name
                cmt = cfg.custom_model_types.get(custom_name)
                if cmt and cmt.class_names:
                    _loaded_model.names = cmt.class_names
            _loaded_model_meta = {"name": os.path.basename(req.model_path)}

        cap = cv2.VideoCapture(req.video_path)
        if not cap.isOpened():
            return {"error": "Cannot open video"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        sid = str(uuid.uuid4())[:8]
        _video_sessions[sid] = {
            "cap": cap, "model": _loaded_model, "conf": req.conf,
            "playing": True, "paused": False,
            "fps": fps, "speed": 1.0,
            "total": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_idx": 0, "last_detections": 0, "last_infer_ms": 0,
            "last_frame": None, "last_result": None,
            "seek_to": None, "step_request": None,
            "video_path": req.video_path,
        }
        return {"session_id": sid, "fps": fps,
                "total_frames": _video_sessions[sid]["total"]}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/viewer/stream/{session_id}")
async def viewer_stream(session_id: str):
    """MJPEG stream of inference results."""
    sess = _video_sessions.get(session_id)
    if not sess:
        return {"error": "Invalid session"}

    def generate():
        from core.inference import run_inference, run_classification
        from core.app_config import AppConfig
        cap = sess["cap"]
        model = sess["model"]
        names = model.names or {}

        try:
            while sess.get("playing", False):
                # Handle seek
                seek_to = sess.get("seek_to")
                if seek_to is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
                    sess["seek_to"] = None

                # Handle step
                step = sess.get("step_request")
                if step is not None:
                    cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur + step))
                    sess["step_request"] = None

                # Handle pause
                if sess.get("paused", False):
                    time.sleep(0.05)
                    continue

                speed = sess.get("speed", 1.0)
                skip = max(0, int(speed) - 1)
                target_delay = 1.0 / sess["fps"]
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    sess["playing"] = False
                    break
                # frame skip
                for _ in range(skip):
                    r2, _ = cap.read()
                    if not r2:
                        break

                sess["frame_idx"] = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                cfg = AppConfig()

                if model.task_type == "classification":
                    result = run_classification(model, frame)
                    sess["last_detections"] = 0
                    sess["last_infer_ms"] = round(result.infer_ms, 2)
                    sess["last_frame"] = frame.copy()
                    sess["last_result"] = None
                    vis = frame.copy()
                    y = 30
                    for cid, conf in result.top_k[:5]:
                        label = f"{names.get(cid, str(cid))}: {conf:.3f}"
                        cv2.putText(vis, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        y += 30
                    _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
                else:
                    result = run_inference(model, frame, cfg.conf_threshold)
                    sess["last_detections"] = len(result.boxes)
                    sess["last_infer_ms"] = round(result.infer_ms, 2)
                    sess["last_frame"] = frame.copy()
                    sess["last_result"] = result

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
                        color = _get_color(style, cid_int, total_cls)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, t_val)
                        parts = []
                        if cfg.show_labels:
                            parts.append(names.get(cid_int, str(cid_int)))
                        if cfg.show_confidence:
                            parts.append(f"{score:.2f}")
                        if parts:
                            _draw_label(frame, " ".join(parts), x1, y1, color, label_size, max(1, t_val - 1), cfg.show_label_bg)
                    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')

                elapsed = time.time() - t0
                if elapsed < target_delay:
                    time.sleep(target_delay - elapsed)
        except Exception as exc:
            import traceback
            print(f"[MJPEG ERROR] {exc}")
            traceback.print_exc()
            sess["playing"] = False

    return StreamingResponse(generate(),
                             media_type='multipart/x-mixed-replace; boundary=frame')


@app.get("/api/viewer/status/{session_id}")
async def viewer_status(session_id: str):
    sess = _video_sessions.get(session_id)
    if not sess:
        return {"error": "Invalid session"}
    return {
        "playing": sess["playing"],
        "paused": sess.get("paused", False),
        "frame_idx": sess["frame_idx"],
        "total": sess["total"],
        "detections": sess["last_detections"],
        "infer_ms": sess["last_infer_ms"],
        "speed": sess.get("speed", 1.0),
    }


@app.post("/api/viewer/stop/{session_id}")
async def viewer_stop(session_id: str):
    sess = _video_sessions.pop(session_id, None)
    if sess:
        sess["playing"] = False
        sess["cap"].release()
    return {"ok": True}


@app.post("/api/viewer/pause/{session_id}")
async def viewer_pause(session_id: str):
    sess = _video_sessions.get(session_id)
    if sess:
        sess["paused"] = not sess.get("paused", False)
    return {"paused": sess.get("paused", False) if sess else False}


class SeekRequest(BaseModel):
    frame: int


@app.post("/api/viewer/seek/{session_id}")
async def viewer_seek(session_id: str, req: SeekRequest):
    sess = _video_sessions.get(session_id)
    if sess:
        sess["seek_to"] = req.frame
    return {"ok": True}


class SpeedRequest(BaseModel):
    speed: float


@app.post("/api/viewer/speed/{session_id}")
async def viewer_speed(session_id: str, req: SpeedRequest):
    sess = _video_sessions.get(session_id)
    if sess:
        sess["speed"] = req.speed
    return {"ok": True}


class StepRequest(BaseModel):
    delta: int = 1


@app.post("/api/viewer/step/{session_id}")
async def viewer_step(session_id: str, req: StepRequest):
    sess = _video_sessions.get(session_id)
    if sess:
        sess["step_request"] = req.delta
    return {"ok": True}


@app.post("/api/viewer/snapshot/{session_id}")
async def viewer_snapshot(session_id: str):
    sess = _video_sessions.get(session_id)
    if not sess or sess.get("last_frame") is None:
        return {"error": "No frame available"}
    os.makedirs("snapshots", exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("snapshots", f"snapshot_{ts}.jpg")
    frame = sess["last_frame"].copy()
    # Draw detections on snapshot
    result = sess.get("last_result")
    if result is not None:
        from core.app_config import AppConfig
        cfg = AppConfig()
        names = sess["model"].names or {}
        total_cls = len(names)
        for box, score, cid in zip(result.boxes, result.scores, result.class_ids):
            cid_int = int(cid)
            style = cfg.get_class_style(cid_int)
            if not style.enabled:
                continue
            x1, y1, x2, y2 = map(int, box)
            t_val = style.thickness or cfg.box_thickness
            color = _get_color(style, cid_int, total_cls)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, t_val)
            parts = []
            if cfg.show_labels:
                parts.append(names.get(cid_int, str(cid_int)))
            if cfg.show_confidence:
                parts.append(f"{score:.2f}")
            if parts:
                cv2.putText(frame, " ".join(parts), (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, cfg.label_size, color, max(1, t_val - 1))
    cv2.imwrite(path, frame)
    return {"ok": True, "path": path}


# ── Video Info ──────────────────────────────────────────
class VideoInfoRequest(BaseModel):
    path: str


@app.post("/api/video/info")
async def video_info(req: VideoInfoRequest):
    try:
        cap = cv2.VideoCapture(req.path)
        if not cap.isOpened():
            return {"error": "Cannot open video"}
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 첫 프레임 캡처 (#1)
        first_frame_b64 = None
        ret, frame = cap.read()
        if ret:
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            first_frame_b64 = base64.b64encode(buf).decode()
        cap.release()
        dur_s = total / fps if fps > 0 else 0
        mins, secs = divmod(int(dur_s), 60)
        return {
            "width": w, "height": h, "fps": round(fps, 2),
            "total_frames": total, "duration": f"{mins}:{secs:02d}",
            "first_frame": first_frame_b64,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Hardware Stats ──────────────────────────────────────
@app.get("/api/system/hw")
async def system_hw():
    import psutil
    proc = psutil.Process(os.getpid())
    info = {
        "cpu": round(proc.cpu_percent(interval=0), 1),
        "ram_mb": round(proc.memory_info().rss / 1024 / 1024),
    }
    try:
        import subprocess, sys as _sys
        flags = 0x08000000 if _sys.platform == "win32" else 0
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2, creationflags=flags,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        info.update(gpu_name=parts[0], gpu_util=int(parts[1]),
                    gpu_mem_used=int(parts[2]), gpu_mem_total=int(parts[3]),
                    gpu_temp=int(parts[4]))
    except Exception:
        info.update(gpu_name="N/A", gpu_util=0, gpu_mem_used=0, gpu_mem_total=0, gpu_temp=0)
    return info


# ── Benchmark API (async) ──────────────────────────────
class BenchmarkRequest(BaseModel):
    models: list[str]
    iterations: int = 100
    input_size: int = 640


_bench_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": []}

# ── Analysis states ─────────────────────────────────────
_compare_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": [], "images": []}
_error_analysis_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": {}}
_conf_opt_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": []}
_embedding_state = {"running": False, "msg": "", "image": None}


@app.post("/api/benchmark/run")
async def run_benchmark(req: BenchmarkRequest):
    if _bench_state["running"]:
        return {"error": "Benchmark already running"}

    _bench_state.update(running=True, progress=0, total=1, msg="Starting...", results=[])

    def _run():
        from core.benchmark_runner import BenchmarkConfig, run_benchmark_core
        _bench_state.update(progress=0, total=0, msg="Starting...", results=[])
        configs = []
        for path in req.models:
            configs.append(BenchmarkConfig(
                model_path=path, iterations=req.iterations,
                warmup=300, src_hw=(1080, 1920),
            ))
        _bench_state["total"] = sum(c.warmup + c.iterations for c in configs)

        def on_progress(done, total, msg):
            _bench_state["progress"] = done
            _bench_state["msg"] = msg

        def on_result(r):
            _bench_state["results"].append({
                "name": r.model_name, "provider": r.provider,
                "fps": round(r.fps, 1),
                "avg": round(r.mean_total_ms, 2),
                "pre_ms": round(r.mean_pre_ms, 2),
                "infer_ms": round(r.mean_infer_ms, 2),
                "post_ms": round(r.mean_post_ms, 2),
                "min": round(r.min_ms, 2),
                "max": round(r.max_ms, 2),
                "std": round(r.std_ms, 2),
                "p50": round(r.p50_ms, 2),
                "p95": round(r.p95_ms, 2),
                "p99": round(r.p99_ms, 2),
                "cpu_pct": round(r.cpu_pct, 1),
                "ram_mb": round(r.ram_mb),
                "gpu_pct": r.gpu_pct,
                "gpu_mem_used": r.gpu_mem_used,
                "gpu_mem_total": r.gpu_mem_total,
            })

        def on_error(msg):
            _bench_state["results"].append({"error": msg})

        try:
            run_benchmark_core(configs, on_progress, on_result, on_error, lambda: False)
        except Exception as e:
            _bench_state["msg"] = f"Error: {e}"
        _bench_state["running"] = False
        _bench_state["msg"] = "Complete"

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "msg": "Benchmark started"}


@app.get("/api/benchmark/status")
async def benchmark_status():
    return {
        "running": _bench_state["running"],
        "progress": _bench_state["progress"],
        "total": _bench_state["total"],
        "msg": _bench_state["msg"],
        "results": _bench_state["results"],
    }


def _get_gpu_info():
    try:
        import subprocess, sys as _sys
        flags = 0x08000000 if _sys.platform == "win32" else 0
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,pci.bus_id",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2, creationflags=flags,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        return {"gpu_name": parts[0], "gpu_driver": parts[1],
                "gpu_memory_gb": round(int(parts[2]) / 1024, 1), "pci_bus": parts[3]}
    except Exception:
        return {"gpu_name": "N/A", "gpu_driver": "N/A", "gpu_memory_gb": 0, "pci_bus": "N/A"}


@app.get("/api/benchmark/export-csv")
async def benchmark_export_csv():
    """벤치마크 결과를 CSV로 다운로드 (시스템 정보 포함)"""
    import io, csv, psutil
    if not _bench_state["results"]:
        return {"error": "No results"}
    buf = io.StringIO()
    w = csv.writer(buf)
    # System info header
    gpu = _get_gpu_info()
    ort_ver = "N/A"
    try:
        import onnxruntime; ort_ver = onnxruntime.__version__
    except ImportError:
        pass
    w.writerow(["OS", f"{platform.system()} {platform.release()} {platform.version()}"])
    w.writerow(["CPU", platform.processor()])
    w.writerow(["RAM Total (GB)", round(psutil.virtual_memory().total / (1024**3), 1)])
    w.writerow(["GPU", gpu["gpu_name"]])
    w.writerow(["Python", platform.python_version()])
    w.writerow(["ONNX Runtime", ort_ver])
    w.writerow([])
    # Benchmark results
    keys = list(_bench_state["results"][0].keys())
    dw = csv.DictWriter(buf, fieldnames=keys)
    dw.writeheader()
    dw.writerows(_bench_state["results"])
    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=benchmark_results.csv"})


@app.get("/api/evaluation/export-csv")
async def evaluation_export_csv():
    """평가 결과를 CSV로 다운로드"""
    import io, csv
    if not _eval_state["results"]:
        return {"error": "No results"}
    buf = io.StringIO()
    keys = list(_eval_state["results"][0].keys())
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    w.writerows(_eval_state["results"])
    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=evaluation_results.csv"})


# ── System Info ─────────────────────────────────────────
@app.get("/api/system/info")
async def system_info():
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }
    try:
        import onnxruntime
        info["ort"] = onnxruntime.__version__
    except ImportError:
        info["ort"] = "N/A"
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda or "N/A"
    except ImportError:
        info["torch"] = "N/A"
        info["cuda"] = "N/A"
    try:
        import subprocess, sys as _sys
        flags = 0x08000000 if _sys.platform == "win32" else 0
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            text=True, timeout=2, creationflags=flags,
        )
        info["gpu_name"] = out.strip()
    except Exception:
        info["gpu_name"] = "N/A"
    return info


# ── File System API (for file/dir selection dialogs) ────
class FileSelectRequest(BaseModel):
    filters: Optional[str] = None


@app.post("/api/fs/select")
async def select_file(req: FileSelectRequest):
    """Return a file selection dialog via tkinter (fallback)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title="Select File",
            filetypes=[("All files", "*.*")] if not req.filters else _parse_filters(req.filters),
        )
        root.destroy()
        return {"path": path or ""}
    except Exception as e:
        return {"error": str(e), "path": ""}


@app.post("/api/fs/select-multi")
async def select_files(req: FileSelectRequest):
    """Return multiple file selection dialog via tkinter."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        paths = filedialog.askopenfilenames(
            title="Select Files",
            filetypes=[("All files", "*.*")] if not req.filters else _parse_filters(req.filters),
        )
        root.destroy()
        return {"paths": list(paths) if paths else []}
    except Exception as e:
        return {"error": str(e), "paths": []}


@app.post("/api/fs/select-dir")
async def select_dir():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askdirectory(title="Select Directory")
        root.destroy()
        return {"path": path or ""}
    except Exception as e:
        return {"error": str(e), "path": ""}


class ListDirRequest(BaseModel):
    path: str
    exts: Optional[list[str]] = None


@app.post("/api/fs/list")
async def list_dir(req: ListDirRequest):
    p = ROOT / req.path if not os.path.isabs(req.path) else Path(req.path)
    if not p.is_dir():
        return {"error": "Not a directory", "files": [], "entries": []}
    files = []
    for item in sorted(p.iterdir()):
        if item.is_file():
            if req.exts and item.suffix.lower() not in req.exts:
                continue
            files.append({"name": item.name, "path": str(item.resolve())})
    return {"files": files}


# ── Evaluation API ──────────────────────────────────────
class EvalRequest(BaseModel):
    models: list[str]
    img_dir: str
    label_dir: str
    conf: float = 0.25


@app.post("/api/evaluation/run")
async def run_evaluation(req: EvalRequest):
    """Run multi-model evaluation against GT labels."""
    try:
        from core.model_loader import load_model as _load_model
        from core.inference import run_inference
        from core.evaluation import evaluate_dataset, evaluate_map50_95
        import glob, cv2

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        img_files = []
        for e in exts:
            img_files.extend(glob.glob(os.path.join(req.img_dir, e)))
        img_files.sort()

        # Load GT
        gt_data = {}
        for fp in img_files:
            stem = os.path.splitext(os.path.basename(fp))[0]
            txt = os.path.join(req.label_dir, stem + ".txt")
            boxes = []
            if os.path.isfile(txt):
                with open(txt) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            boxes.append((int(parts[0]), *map(float, parts[1:5])))
            gt_data[stem] = boxes

        results = []
        for model_path in req.models:
            name = os.path.basename(model_path)
            try:
                mi = _load_model(model_path)
                pred_data = {}
                for fp in img_files:
                    frame = cv2.imread(fp)
                    if frame is None:
                        continue
                    h, w = frame.shape[:2]
                    res = run_inference(mi, frame, req.conf)
                    stem = os.path.splitext(os.path.basename(fp))[0]
                    boxes = []
                    for box, score, cid in zip(res.boxes, res.scores, res.class_ids):
                        x1, y1, x2, y2 = box
                        cx = ((x1+x2)/2)/w; cy = ((y1+y2)/2)/h
                        bw = (x2-x1)/w; bh = (y2-y1)/h
                        boxes.append((int(cid), cx, cy, bw, bh, float(score)))
                    pred_data[stem] = boxes

                res50 = evaluate_dataset(gt_data, pred_data, 0.5)
                map5095 = evaluate_map50_95(gt_data, pred_data)
                ov = res50.get("__overall__", {})
                results.append({
                    "name": name,
                    "map50": round(ov.get("ap", 0) * 100, 4),
                    "map5095": round(map5095 * 100, 4),
                    "precision": round(ov.get("precision", 0) * 100, 4),
                    "recall": round(ov.get("recall", 0) * 100, 4),
                    "f1": round(ov.get("f1", 0) * 100, 4),
                })
            except Exception as e:
                results.append({"name": name, "error": str(e),
                                "map50": 0, "map5095": 0, "precision": 0, "recall": 0, "f1": 0})
        return results
    except Exception as e:
        return {"error": str(e)}


def _parse_filters(s: str):
    """Parse Qt-style filter string to tkinter format."""
    pairs = []
    for part in s.split(";;"):
        part = part.strip()
        if "(" in part and ")" in part:
            label = part[:part.index("(")].strip()
            exts = part[part.index("(") + 1:part.index(")")].strip()
            pairs.append((label, exts))
        else:
            pairs.append(("Files", part))
    return pairs or [("All files", "*.*")]


# ── Helper: glob images ─────────────────────────────────
def _glob_images(img_dir):
    import glob
    files = []
    for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(glob.glob(os.path.join(img_dir, e)))
    files.sort()
    return files


def _encode_jpeg(img, quality=80):
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def _draw_detections(frame, result, names):
    vis = frame.copy()
    total_cls = len(names)
    for box, score, cid in zip(result.boxes, result.scores, result.class_ids):
        cid_int = int(cid)
        color = _get_color(type('S', (), {'color': None})(), cid_int, total_cls)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(cid_int, str(cid_int))} {score:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 4, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


# ── 1. Model Compare API ───────────────────────────────
class ModelCompareRequest(BaseModel):
    model_a: str
    model_b: str
    model_type_a: str = "yolo"
    model_type_b: str = "yolo"
    img_dir: str
    conf: float = 0.25


@app.post("/api/analysis/model-compare")
async def run_model_compare(req: ModelCompareRequest):
    if _compare_state["running"]:
        return {"error": "Already running"}
    _compare_state.update(running=True, progress=0, total=0, msg="Starting...", results=[], images=[])

    def _run():
        from core.model_loader import load_model as _load
        from core.inference import run_inference
        try:
            mi_a = _load(req.model_a, model_type=req.model_type_a)
            mi_b = _load(req.model_b, model_type=req.model_type_b)
        except Exception as e:
            _compare_state.update(running=False, msg=f"Load error: {e}")
            return
        imgs = _glob_images(req.img_dir)
        if not imgs:
            _compare_state.update(running=False, msg="No images found")
            return
        _compare_state["total"] = len(imgs)
        names_a = mi_a.names or {}
        names_b = mi_b.names or {}
        for i, fp in enumerate(imgs):
            frame = cv2.imread(fp)
            if frame is None:
                _compare_state["progress"] = i + 1
                continue
            res_a = run_inference(mi_a, frame, req.conf)
            res_b = run_inference(mi_b, frame, req.conf)
            vis_a = _draw_detections(frame, res_a, names_a)
            vis_b = _draw_detections(frame, res_b, names_b)
            _compare_state["results"].append({
                "image_name": os.path.basename(fp),
                "img_a_b64": _encode_jpeg(vis_a),
                "img_b_b64": _encode_jpeg(vis_b),
                "count_a": len(res_a.boxes),
                "count_b": len(res_b.boxes),
                "ms_a": round(res_a.infer_ms, 2),
                "ms_b": round(res_b.infer_ms, 2),
            })
            _compare_state["progress"] = i + 1
            _compare_state["msg"] = f"{i+1}/{len(imgs)}"
        _compare_state.update(running=False, msg="Complete")

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True}


@app.get("/api/analysis/model-compare/status")
async def model_compare_status():
    return dict(_compare_state)


# ── 2. Error Analysis (FP/FN) API ──────────────────────
class ErrorAnalysisRequest(BaseModel):
    model_path: str
    model_type: str = "yolo"
    img_dir: str
    label_dir: str
    iou_threshold: float = 0.5
    conf: float = 0.25


@app.post("/api/analysis/error-analysis")
async def run_error_analysis(req: ErrorAnalysisRequest):
    if _error_analysis_state["running"]:
        return {"error": "Already running"}
    _error_analysis_state.update(running=True, progress=0, total=0, msg="Starting...", results={})

    def _run():
        from core.model_loader import load_model as _load
        from core.inference import run_inference
        try:
            mi = _load(req.model_path, model_type=req.model_type)
        except Exception as e:
            _error_analysis_state.update(running=False, msg=f"Load error: {e}")
            return
        imgs = _glob_images(req.img_dir)
        if not imgs:
            _error_analysis_state.update(running=False, msg="No images found")
            return
        _error_analysis_state["total"] = len(imgs)

        def _cat_size(area):
            if area < 32*32: return "small"
            if area < 96*96: return "medium"
            return "large"

        def _cat_pos(cy):
            if cy < 0.33: return "top"
            if cy < 0.67: return "center"
            return "bottom"

        fp_stats = {"count": 0, "small": 0, "medium": 0, "large": 0, "top": 0, "center": 0, "bottom": 0}
        fn_stats = {"count": 0, "small": 0, "medium": 0, "large": 0, "top": 0, "center": 0, "bottom": 0}

        for i, fp in enumerate(imgs):
            frame = cv2.imread(fp)
            if frame is None:
                _error_analysis_state["progress"] = i + 1
                continue
            h, w = frame.shape[:2]
            res = run_inference(mi, frame, req.conf)
            # Load GT
            stem = os.path.splitext(os.path.basename(fp))[0]
            txt = os.path.join(req.label_dir, stem + ".txt")
            gt_boxes = []
            if os.path.isfile(txt):
                with open(txt) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cx, cy, bw, bh = map(float, parts[1:5])
                            x1 = (cx - bw/2) * w; y1 = (cy - bh/2) * h
                            x2 = (cx + bw/2) * w; y2 = (cy + bh/2) * h
                            gt_boxes.append([x1, y1, x2, y2, cy])

            pred_boxes = list(zip(res.boxes, res.scores, res.class_ids))
            gt_matched = [False] * len(gt_boxes)
            pred_matched = [False] * len(pred_boxes)

            for pi, (pbox, _, _) in enumerate(pred_boxes):
                best_iou, best_gi = 0, -1
                for gi, gb in enumerate(gt_boxes):
                    if gt_matched[gi]:
                        continue
                    ix1 = max(pbox[0], gb[0]); iy1 = max(pbox[1], gb[1])
                    ix2 = min(pbox[2], gb[2]); iy2 = min(pbox[3], gb[3])
                    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                    a1 = (pbox[2]-pbox[0]) * (pbox[3]-pbox[1])
                    a2 = (gb[2]-gb[0]) * (gb[3]-gb[1])
                    iou = inter / (a1 + a2 - inter + 1e-9)
                    if iou > best_iou:
                        best_iou, best_gi = iou, gi
                if best_iou >= req.iou_threshold and best_gi >= 0:
                    gt_matched[best_gi] = True
                    pred_matched[pi] = True

            # FP: unmatched predictions
            for pi, (pbox, _, _) in enumerate(pred_boxes):
                if not pred_matched[pi]:
                    area = (pbox[2]-pbox[0]) * (pbox[3]-pbox[1])
                    cy = ((pbox[1]+pbox[3])/2) / h
                    fp_stats["count"] += 1
                    fp_stats[_cat_size(area)] += 1
                    fp_stats[_cat_pos(cy)] += 1
            # FN: unmatched GT
            for gi, gb in enumerate(gt_boxes):
                if not gt_matched[gi]:
                    area = (gb[2]-gb[0]) * (gb[3]-gb[1])
                    fn_stats["count"] += 1
                    fn_stats[_cat_size(area)] += 1
                    fn_stats[_cat_pos(gb[4])] += 1  # gb[4] is normalized cy

            _error_analysis_state["progress"] = i + 1
            _error_analysis_state["msg"] = f"{i+1}/{len(imgs)}"

        _error_analysis_state["results"] = {"fp": fp_stats, "fn": fn_stats}
        _error_analysis_state.update(running=False, msg="Complete")

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True}


@app.get("/api/analysis/error-analysis/status")
async def error_analysis_status():
    return dict(_error_analysis_state)


# ── 3. Confidence Optimizer API ─────────────────────────
class ConfOptimizerRequest(BaseModel):
    model_path: str
    model_type: str = "yolo"
    img_dir: str
    label_dir: str
    step: float = 0.05


@app.post("/api/analysis/conf-optimizer")
async def run_conf_optimizer(req: ConfOptimizerRequest):
    if _conf_opt_state["running"]:
        return {"error": "Already running"}
    _conf_opt_state.update(running=True, progress=0, total=0, msg="Starting...", results=[])

    def _run():
        from core.model_loader import load_model as _load
        from core.inference import run_inference
        try:
            mi = _load(req.model_path, model_type=req.model_type)
        except Exception as e:
            _conf_opt_state.update(running=False, msg=f"Load error: {e}")
            return
        imgs = _glob_images(req.img_dir)
        if not imgs:
            _conf_opt_state.update(running=False, msg="No images found")
            return
        names = mi.names or {}

        # Collect all detections at low conf and all GT
        all_preds = []  # (class_id, score, x1, y1, x2, y2, img_idx)
        all_gt = []     # (class_id, x1, y1, x2, y2, img_idx)
        _conf_opt_state["total"] = len(imgs)
        _conf_opt_state["msg"] = "Running inference..."

        for idx, fp in enumerate(imgs):
            frame = cv2.imread(fp)
            if frame is None:
                _conf_opt_state["progress"] = idx + 1
                continue
            h, w = frame.shape[:2]
            res = run_inference(mi, frame, 0.01)
            for box, score, cid in zip(res.boxes, res.scores, res.class_ids):
                all_preds.append((int(cid), float(score), *box, idx))
            # Load GT
            stem = os.path.splitext(os.path.basename(fp))[0]
            txt = os.path.join(req.label_dir, stem + ".txt")
            if os.path.isfile(txt):
                with open(txt) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:5])
                            x1 = (cx - bw/2) * w; y1 = (cy - bh/2) * h
                            x2 = (cx + bw/2) * w; y2 = (cy + bh/2) * h
                            all_gt.append((cid, x1, y1, x2, y2, idx))
            _conf_opt_state["progress"] = idx + 1

        # Group by class
        class_ids = set(g[0] for g in all_gt)
        thresholds = np.arange(0.05, 0.951, req.step)
        _conf_opt_state["msg"] = "Sweeping thresholds..."
        results = []

        for cid in sorted(class_ids):
            gt_cls = [(g[1], g[2], g[3], g[4], g[5]) for g in all_gt if g[0] == cid]
            pred_cls = [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in all_preds if p[0] == cid]
            best_f1, best_t, best_p, best_r = 0, 0.25, 0, 0

            for t in thresholds:
                filtered = [p for p in pred_cls if p[0] >= t]
                tp = 0
                gt_matched = set()
                for p in sorted(filtered, key=lambda x: -x[0]):
                    best_iou, best_gi = 0, -1
                    for gi, g in enumerate(gt_cls):
                        if gi in gt_matched or g[4] != p[5]:
                            continue
                        ix1 = max(p[1], g[0]); iy1 = max(p[2], g[1])
                        ix2 = min(p[3], g[2]); iy2 = min(p[4], g[3])
                        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                        a1 = (p[3]-p[1]) * (p[4]-p[2])
                        a2 = (g[2]-g[0]) * (g[3]-g[1])
                        iou = inter / (a1 + a2 - inter + 1e-9)
                        if iou > best_iou:
                            best_iou, best_gi = iou, gi
                    if best_iou >= 0.5 and best_gi >= 0:
                        tp += 1
                        gt_matched.add(best_gi)
                prec = tp / len(filtered) if filtered else 0
                rec = tp / len(gt_cls) if gt_cls else 0
                f1 = 2 * prec * rec / (prec + rec + 1e-9)
                if f1 > best_f1:
                    best_f1, best_t, best_p, best_r = f1, float(t), prec, rec

            results.append({
                "class_id": cid,
                "class_name": names.get(cid, str(cid)),
                "best_threshold": round(best_t, 3),
                "best_f1": round(best_f1, 4),
                "precision": round(best_p, 4),
                "recall": round(best_r, 4),
            })

        _conf_opt_state["results"] = results
        _conf_opt_state.update(running=False, msg="Complete")

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True}


@app.get("/api/analysis/conf-optimizer/status")
async def conf_optimizer_status():
    return dict(_conf_opt_state)


# ── 4. Embedding Viewer API ────────────────────────────
class EmbeddingViewerRequest(BaseModel):
    model_path: str
    img_dir: str
    method: str = "tsne"  # tsne / umap / pca


@app.post("/api/analysis/embedding-viewer")
async def run_embedding_viewer(req: EmbeddingViewerRequest):
    if _embedding_state["running"]:
        return {"error": "Already running"}
    _embedding_state.update(running=True, msg="Starting...", image=None)

    def _run():
        import onnxruntime as ort
        try:
            session = ort.InferenceSession(req.model_path)
        except Exception as e:
            _embedding_state.update(running=False, msg=f"Load error: {e}")
            return
        inp = session.get_inputs()[0]
        h = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 224
        w = int(inp.shape[3]) if isinstance(inp.shape[3], int) else 224

        embeddings, labels = [], []
        img_dir = Path(req.img_dir)
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        files = []
        for sub in sorted(img_dir.iterdir()):
            if sub.is_dir():
                for f in sorted(sub.iterdir()):
                    if f.suffix.lower() in exts:
                        files.append((f, sub.name))
            elif sub.suffix.lower() in exts:
                files.append((sub, "unlabeled"))

        if not files:
            _embedding_state.update(running=False, msg="No images found")
            return

        _embedding_state["msg"] = f"Extracting embeddings: 0/{len(files)}"
        for i, (fp, label) in enumerate(files):
            img = cv2.imread(str(fp))
            if img is None:
                continue
            img = cv2.resize(img, (w, h))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis], dtype=np.float32) / 255.0
            out = session.run(None, {inp.name: tensor})
            vec = out[0].flatten()
            embeddings.append(vec)
            labels.append(label)
            _embedding_state["msg"] = f"Extracting embeddings: {i+1}/{len(files)}"

        if len(embeddings) < 2:
            _embedding_state.update(running=False, msg="Need at least 2 images")
            return

        X = np.array(embeddings)
        _embedding_state["msg"] = f"Running {req.method.upper()}..."

        try:
            if req.method == "pca":
                from sklearn.decomposition import PCA
                coords = PCA(n_components=2).fit_transform(X)
            elif req.method == "umap":
                import umap
                n = min(15, len(X) - 1)
                coords = umap.UMAP(n_components=2, n_neighbors=n).fit_transform(X)
            else:
                from sklearn.manifold import TSNE
                perp = min(30, len(X) - 1)
                coords = TSNE(n_components=2, perplexity=perp).fit_transform(X)
        except Exception as e:
            _embedding_state.update(running=False, msg=f"Reduction error: {e}")
            return

        # Plot
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        unique_labels = sorted(set(labels))
        cmap = plt.cm.get_cmap('tab10', max(len(unique_labels), 1))
        for idx, lbl in enumerate(unique_labels):
            mask = [l == lbl for l in labels]
            pts = coords[mask]
            ax.scatter(pts[:, 0], pts[:, 1], c=[cmap(idx)], label=lbl, s=20, alpha=0.7)
        ax.legend(fontsize=8, markerscale=2)
        ax.set_title(f"Embedding Visualization ({req.method.upper()})")
        fig.tight_layout()

        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        plt.close(fig)
        buf.seek(0)
        _embedding_state["image"] = base64.b64encode(buf.read()).decode()
        _embedding_state.update(running=False, msg="Complete")

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True}


@app.get("/api/analysis/embedding-viewer/status")
async def embedding_viewer_status():
    return dict(_embedding_state)


# ── 5. Inference Analysis API ──────────────────────────
class InferenceAnalysisRequest(BaseModel):
    model_path: str
    model_type: str = "yolo"
    image_path: str
    conf: float = 0.25


@app.post("/api/analysis/inference-analysis")
async def run_inference_analysis(req: InferenceAnalysisRequest):
    try:
        from core.model_loader import load_model as _load
        from core.inference import run_inference, letterbox, preprocess
        import time as _time

        mi = _load(req.model_path, model_type=req.model_type)
        frame = cv2.imread(req.image_path)
        if frame is None:
            return {"error": "Cannot read image"}
        names = mi.names or {}

        # Pre-process
        t0 = _time.perf_counter()
        padded, ratio, pad = letterbox(frame, mi.input_size)
        tensor = preprocess(frame, mi.input_size)
        pre_ms = (_time.perf_counter() - t0) * 1000

        # Inference
        t1 = _time.perf_counter()
        result = run_inference(mi, frame, req.conf)
        infer_ms = result.infer_ms
        post_ms = (_time.perf_counter() - t1) * 1000 - infer_ms

        total_ms = pre_ms + infer_ms + post_ms

        # Letterbox visualization: highlight padding in red overlay
        lb_vis = padded.copy()
        h, w = padded.shape[:2]
        pw, ph = int(pad[0]), int(pad[1])
        overlay = lb_vis.copy()
        if ph > 0:
            cv2.rectangle(overlay, (0, 0), (w, ph), (0, 0, 200), -1)
            cv2.rectangle(overlay, (0, h - ph), (w, h), (0, 0, 200), -1)
        if pw > 0:
            cv2.rectangle(overlay, (0, 0), (pw, h), (0, 0, 200), -1)
            cv2.rectangle(overlay, (w - pw, 0), (w, h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.4, lb_vis, 0.6, 0, lb_vis)

        # Detection visualization
        det_vis = _draw_detections(frame, result, names)

        # Tensor stats
        t_stats = {
            "min": round(float(tensor.min()), 6),
            "max": round(float(tensor.max()), 6),
            "mean": round(float(tensor.mean()), 6),
            "std": round(float(tensor.std()), 6),
            "shape": list(tensor.shape),
        }

        detections = []
        for box, score, cid in zip(result.boxes, result.scores, result.class_ids):
            detections.append({
                "class_id": int(cid),
                "class_name": names.get(int(cid), str(int(cid))),
                "confidence": round(float(score), 4),
                "bbox": [round(float(x), 1) for x in box],
            })

        inp = mi.session.get_inputs()[0]
        out = mi.session.get_outputs()[0]

        return {
            "original_image": _encode_jpeg(frame),
            "letterbox_image": _encode_jpeg(lb_vis),
            "detection_image": _encode_jpeg(det_vis),
            "tensor_stats": t_stats,
            "detections": detections,
            "timing": {
                "pre_ms": round(pre_ms, 2),
                "infer_ms": round(infer_ms, 2),
                "post_ms": round(post_ms, 2),
                "total_ms": round(total_ms, 2),
            },
            "model_info": {
                "input_shape": list(inp.shape),
                "output_shape": list(out.shape),
                "num_classes": len(names),
            },
        }
    except Exception as e:
        return {"error": str(e)}


# ── 6. System Full Info ─────────────────────────────────
@app.get("/api/system/full-info")
async def system_full_info():
    import psutil
    gpu = _get_gpu_info()
    ort_ver = "N/A"
    try:
        import onnxruntime
        ort_ver = onnxruntime.__version__
    except ImportError:
        pass
    return {
        "os": f"{platform.system()} {platform.release()} {platform.version()}",
        "cpu": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "python": platform.python_version(),
        "onnxruntime": ort_ver,
        **gpu,
    }


# ── Launch ──────────────────────────────────────────────
def main():
    import uvicorn
    port = int(os.environ.get("PORT", 8765))
    print(f"\n  ssook running at http://localhost:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
