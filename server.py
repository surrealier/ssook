"""
ssook Web Server — FastAPI backend serving the web UI
and exposing core/ functionality as REST API.
"""
import os
import sys
import platform
import asyncio
import base64
import glob as glob_module
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
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


def _imread(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread replacement that handles unicode/Korean paths on Windows."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, flags)
    except Exception:
        return None


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

app = FastAPI(title="ssook", version="1.5.3")

# 백그라운드 작업용 스레드 풀 (최대 4개 동시 작업)
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ssook-bg")

# ── Client heartbeat & auto-shutdown ────────────────────
_last_heartbeat = time.time()
_HEARTBEAT_TIMEOUT = 30  # seconds without heartbeat before shutdown


@app.post("/api/heartbeat")
async def heartbeat():
    global _last_heartbeat
    _last_heartbeat = time.time()
    return {"ok": True}


def _heartbeat_watchdog():
    """Background thread: shutdown server if no heartbeat received."""
    global _last_heartbeat
    while True:
        time.sleep(5)
        if time.time() - _last_heartbeat > _HEARTBEAT_TIMEOUT:
            import logging
            logging.info("No client heartbeat — shutting down server")
            os._exit(0)


threading.Thread(target=_heartbeat_watchdog, daemon=True).start()

# ── Static files ────────────────────────────────────────
WEB_DIR = ROOT / "web"
app.mount("/css", StaticFiles(directory=WEB_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=WEB_DIR / "js"), name="js")
app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")


@app.get("/")
async def index():
    from fastapi.responses import HTMLResponse
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace("{{VERSION}}", app.version)
    return HTMLResponse(html)


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
            "samples_dir": str(ROOT / "assets" / "samples"),
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

        try:
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

            # 이미지 캐시: 첫 번째 모델 평가 시 로드, 이후 모델에서 재사용
            _img_cache = {}

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
                    if fp in _img_cache:
                        frame = _img_cache[fp]
                    else:
                        frame = _imread(fp)
                        if frame is not None:
                            _img_cache[fp] = frame
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

            _img_cache.clear()
            _eval_state.update(running=False, msg="Complete")
        except Exception as e:
            _eval_state.update(running=False, msg=f"Error: {e}")

    _executor.submit(_run)
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
    return await asyncio.get_event_loop().run_in_executor(_executor, _load_model_sync, req)


def _load_model_sync(req: ModelLoadRequest):
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
    """기본 모델을 백그라운드에서 로드 — 서버 시작을 블로킹하지 않음"""
    def _load_bg():
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
    _executor.submit(_load_bg)


# ── Inference API ───────────────────────────────────────
class InferRequest(BaseModel):
    model_path: str
    image_path: Optional[str] = None
    conf: float = 0.25
    clip_labels: Optional[str] = None          # comma-separated labels for CLIP
    clip_text_encoder: Optional[str] = None    # text encoder ONNX path
    vlm_prompt: Optional[str] = None           # VLM prompt


@app.post("/api/infer/image")
async def infer_image(req: InferRequest):
    """Run inference on a single image, return annotated JPEG + detections."""
    return await asyncio.get_event_loop().run_in_executor(_executor, _infer_image_sync, req)


def _infer_image_sync(req: InferRequest):
    global _loaded_model, _loaded_model_meta
    try:
        from core.model_loader import load_model as _load
        from core.inference import run_inference, run_classification, run_segmentation, run_embedding
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

        frame = _imread(req.image_path)
        if frame is None:
            return {"error": "Cannot read image"}

        names = _loaded_model.names or {}

        if _loaded_model.task_type == "embedding":
            # CLIP zero-shot classification or embedding visualization
            from core.clip_inference import CLIPModel, simple_tokenize
            import time as _time
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
                    cv2.putText(vis, f"{label}: {score*100:.1f}%", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y += 28
                _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return {
                    "image": base64.b64encode(buf).decode(), "detections": 0,
                    "clip_result": [{"label": l, "score": round(s, 4)} for l, s in ranked],
                    "infer_ms": round(infer_ms, 2), "classes": {},
                }
            else:
                emb = clip_model.encode_image(frame)
                infer_ms = (_time.perf_counter() - t0) * 1000
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return {
                    "image": base64.b64encode(buf).decode(), "detections": 0,
                    "embedding": f"dim={len(emb)}, norm={float(np.linalg.norm(emb)):.4f}",
                    "infer_ms": round(infer_ms, 2), "classes": {},
                }

        if _loaded_model.task_type == "pose":
            from core.inference import run_pose
            result = run_pose(_loaded_model, frame, cfg.conf_threshold)
            vis = frame.copy()
            _SKELETON = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
                         (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
            for i in range(len(result.boxes)):
                x1, y1, x2, y2 = map(int, result.boxes[i])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                kpts = result.keypoints[i]  # (17, 3)
                for j in range(17):
                    if kpts[j, 2] > 0.3:
                        cv2.circle(vis, (int(kpts[j,0]), int(kpts[j,1])), 3, (0, 0, 255), -1)
                for a, b in _SKELETON:
                    if kpts[a, 2] > 0.3 and kpts[b, 2] > 0.3:
                        cv2.line(vis, (int(kpts[a,0]), int(kpts[a,1])),
                                 (int(kpts[b,0]), int(kpts[b,1])), (255, 128, 0), 2)
            _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return {
                "image": base64.b64encode(buf).decode(),
                "detections": len(result.boxes),
                "pose": f"{len(result.boxes)} persons detected",
                "infer_ms": round(result.infer_ms, 2), "classes": {},
            }

        if _loaded_model.task_type == "instance_segmentation":
            from core.inference import run_instance_seg
            result = run_instance_seg(_loaded_model, frame, cfg.conf_threshold)
            vis = frame.copy()
            colors = _generate_palette(max(len(result.masks), 1))
            for i, mask in enumerate(result.masks):
                color = colors[i % len(colors)]
                overlay = vis.copy()
                overlay[mask > 0] = color
                vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
                x1, y1, x2, y2 = map(int, result.boxes[i])
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cid = int(result.class_ids[i])
                label = f"{names.get(cid, str(cid))} {result.scores[i]:.2f}"
                _draw_label(vis, label, x1, y1, color, 0.5, 1, True)
            _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return {
                "image": base64.b64encode(buf).decode(),
                "detections": len(result.boxes),
                "instance_seg": f"{len(result.boxes)} instances",
                "infer_ms": round(result.infer_ms, 2), "classes": {},
            }

        if _loaded_model.task_type == "vlm":
            vis = frame.copy()
            prompt = req.vlm_prompt or "Describe this image."
            cv2.putText(vis, f"VLM: {prompt[:60]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return {
                "image": base64.b64encode(buf).decode(), "detections": 0,
                "vlm_result": f"VLM inference not yet implemented (prompt: {prompt})",
                "infer_ms": 0, "classes": {},
            }

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

        if _loaded_model.task_type == "segmentation":
            result = run_segmentation(_loaded_model, frame)
            vis = _overlay_segmentation(frame, result.mask)
            _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            unique_cls = [int(c) for c in np.unique(result.mask) if c > 0]
            return {
                "image": base64.b64encode(buf).decode(),
                "detections": len(unique_cls),
                "segmentation": f"{result.num_classes} classes, {len(unique_cls)} present",
                "infer_ms": round(result.infer_ms, 2),
                "classes": {c: names.get(c, str(c)) for c in unique_cls},
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
_SESSION_TIMEOUT = 300  # 5분 무활동 시 세션 자동 정리


def _cleanup_stale_sessions():
    """타임아웃된 비디오 세션 정리"""
    now = time.time()
    stale = [sid for sid, s in _video_sessions.items()
             if not s.get("playing") and now - s.get("last_access", 0) > _SESSION_TIMEOUT]
    for sid in stale:
        sess = _video_sessions.pop(sid, None)
        if sess:
            try:
                sess["cap"].release()
            except Exception:
                pass
            sess["last_frame"] = None
            sess["last_result"] = None
            print(f"[Session] Cleaned up stale session: {sid}")


class VideoStartRequest(BaseModel):
    model_path: str
    video_path: str
    conf: float = 0.25
    stream_max_height: int = 0  # 0=원본, 720/480 등 지정 시 다운스케일
    tracker_type: str = "none"  # none / bytetrack / sort


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

        if _loaded_model.task_type == "embedding":
            return {"error": "CLIP/Embedder 모델은 뷰어에서 사용할 수 없습니다."}

        cap = cv2.VideoCapture(req.video_path)
        if not cap.isOpened():
            return {"error": "Cannot open video"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        sid = str(uuid.uuid4())[:8]
        tracker = None
        if req.tracker_type and req.tracker_type != "none":
            from core.tracking import create_tracker
            tracker = create_tracker(req.tracker_type)
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
            "last_access": time.time(),
            "stream_max_height": req.stream_max_height,
            "tracker": tracker,
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
        from core.inference import run_inference, run_classification, run_segmentation, run_embedding
        from core.app_config import AppConfig
        cap = sess["cap"]
        model = sess["model"]
        names = model.names or {}
        cfg = AppConfig()  # 루프 밖에서 한 번만 참조
        max_h = sess.get("stream_max_height", 0)

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

                if model.task_type == "classification":
                    result = run_classification(model, frame)
                    sess["last_detections"] = 0
                    sess["last_infer_ms"] = round(result.infer_ms, 2)
                    sess["last_frame"] = frame.copy()
                    sess["last_result"] = None
                    vis = frame
                    y = 30
                    for cid, conf in result.top_k[:5]:
                        label = f"{names.get(cid, str(cid))}: {conf:.3f}"
                        cv2.putText(vis, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        y += 30
                    _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 65])
                elif model.task_type == "segmentation":
                    result = run_segmentation(model, frame)
                    sess["last_detections"] = 0
                    sess["last_infer_ms"] = round(result.infer_ms, 2)
                    sess["last_frame"] = frame.copy()
                    sess["last_result"] = None
                    vis = _overlay_segmentation(frame, result.mask)
                    _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 65])
                else:
                    result = run_inference(model, frame, cfg.conf_threshold)
                    sess["last_detections"] = len(result.boxes)
                    sess["last_infer_ms"] = round(result.infer_ms, 2)
                    sess["last_frame"] = frame.copy()
                    sess["last_result"] = result

                    # Tracker integration
                    tracker = sess.get("tracker")
                    tracks = None
                    if tracker and len(result.boxes) > 0:
                        import numpy as _np
                        tracks = tracker.update(
                            _np.array(result.boxes), _np.array(result.scores), _np.array(result.class_ids)
                        )

                    thickness = cfg.box_thickness
                    label_size = cfg.label_size
                    total_cls = len(names)

                    if tracks:
                        for tr in tracks:
                            cid_int = int(tr.class_id)
                            style = cfg.get_class_style(cid_int)
                            if not style.enabled:
                                continue
                            x1, y1, x2, y2 = map(int, tr.box)
                            t_val = style.thickness or thickness
                            color = _get_color(style, cid_int, total_cls)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, t_val)
                            parts = [f"ID:{tr.id}"]
                            if cfg.show_labels:
                                parts.append(names.get(cid_int, str(cid_int)))
                            if cfg.show_confidence:
                                parts.append(f"{tr.score:.2f}")
                            _draw_label(frame, " ".join(parts), x1, y1, color, label_size, max(1, t_val - 1), cfg.show_label_bg)
                            # Draw trajectory
                            if len(tr.trajectory) > 1:
                                for j in range(1, len(tr.trajectory)):
                                    p1 = (int(tr.trajectory[j-1][0]), int(tr.trajectory[j-1][1]))
                                    p2 = (int(tr.trajectory[j][0]), int(tr.trajectory[j][1]))
                                    cv2.line(frame, p1, p2, color, max(1, t_val - 1))
                    else:
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
                    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                # 스트리밍 다운스케일 (노트북 최적화)
                if max_h > 0 and buf is not None:
                    out_frame = vis if model.task_type == "classification" else frame
                    oh, ow = out_frame.shape[:2]
                    if oh > max_h:
                        scale = max_h / oh
                        small = cv2.resize(out_frame, (int(ow * scale), max_h), interpolation=cv2.INTER_AREA)
                        _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 65])
                sess["last_display_jpeg"] = buf.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       sess["last_display_jpeg"] + b'\r\n')

                elapsed = time.time() - t0
                remaining = target_delay - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                elif remaining < -target_delay:
                    # 추론이 2프레임 이상 지연 시 프레임 스킵
                    extra_skip = int(-remaining / target_delay)
                    for _ in range(extra_skip):
                        r2, _ = cap.read()
                        if not r2:
                            break
        except Exception as exc:
            import traceback
            print(f"[MJPEG ERROR] {exc}")
            traceback.print_exc()
            sess["playing"] = False

    return StreamingResponse(generate(),
                             media_type='multipart/x-mixed-replace; boundary=frame')


@app.get("/api/viewer/status/{session_id}")
async def viewer_status(session_id: str):
    _cleanup_stale_sessions()
    sess = _video_sessions.get(session_id)
    if not sess:
        return {"error": "Invalid session"}
    sess["last_access"] = time.time()
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
    if not sess:
        return {"error": "No session"}
    os.makedirs("snapshots", exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("snapshots", f"snapshot_{ts}.jpg")
    # 디스플레이된 화면 그대로 저장
    jpeg_data = sess.get("last_display_jpeg")
    if jpeg_data:
        with open(path, "wb") as f:
            f.write(jpeg_data)
    elif sess.get("last_frame") is not None:
        cv2.imwrite(path, sess["last_frame"])
    else:
        return {"error": "No frame available"}
    return {"ok": True, "path": path}


@app.post("/api/viewer/save-crops/{session_id}")
async def viewer_save_crops(session_id: str):
    """Save cropped detection boxes from the last frame."""
    sess = _video_sessions.get(session_id)
    if not sess:
        return {"error": "No session"}
    frame = sess.get("last_frame")
    result = sess.get("last_result")
    if frame is None or result is None or len(result.boxes) == 0:
        return {"error": "No detections available"}
    names = (sess.get("model") and sess["model"].names) or {}
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("snapshots", f"crops_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    h, w = frame.shape[:2]
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


@app.post("/api/infer/save-crops")
async def infer_save_crops(req: InferRequest):
    """Run inference on a single image and save cropped boxes."""
    global _loaded_model, _loaded_model_meta
    try:
        from core.model_loader import load_model as _load
        from core.inference import run_inference
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
        frame = _imread(req.image_path)
        if frame is None:
            return {"error": "Cannot read image"}
        if _loaded_model.task_type != "detection":
            return {"error": "Crop save is only for detection models"}
        result = run_inference(_loaded_model, frame, cfg.conf_threshold)
        if len(result.boxes) == 0:
            return {"error": "No detections"}
        names = _loaded_model.names or {}
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("snapshots", f"crops_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        h, w = frame.shape[:2]
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


# ── Video Info ──────────────────────────────────────────
class VideoInfoRequest(BaseModel):
    path: str


@app.post("/api/video/info")
async def video_info(req: VideoInfoRequest):
    try:
        ext = os.path.splitext(req.path)[1].lower()
        # 이미지 파일인 경우 _imread로 처리
        if ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            frame = _imread(req.path)
            if frame is None:
                return {"error": "Cannot read image"}
            h, w = frame.shape[:2]
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return {
                "width": w, "height": h, "fps": 0,
                "total_frames": 1, "duration": "0:00",
                "first_frame": base64.b64encode(buf).decode(),
            }
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
_gpu_available = None  # None=미확인, True/False=캐싱

def _check_gpu_available():
    global _gpu_available
    if _gpu_available is None:
        try:
            import subprocess, sys as _sys
            flags = 0x08000000 if _sys.platform == "win32" else 0
            subprocess.check_output(["nvidia-smi", "--version"],
                                    text=True, timeout=2, creationflags=flags)
            _gpu_available = True
        except Exception:
            _gpu_available = False
    return _gpu_available

@app.get("/api/system/hw")
async def system_hw():
    import psutil
    proc = psutil.Process(os.getpid())
    info = {
        "cpu": round(proc.cpu_percent(interval=0), 1),
        "ram_mb": round(proc.memory_info().rss / 1024 / 1024),
    }
    if _check_gpu_available():
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
    else:
        info.update(gpu_name="N/A", gpu_util=0, gpu_mem_used=0, gpu_mem_total=0, gpu_temp=0)
    # 메모리 자동 정리: RSS가 시스템 RAM의 50% 초과 시 캐시 정리
    import psutil as _ps
    total_ram = _ps.virtual_memory().total
    if proc.memory_info().rss > total_ram * 0.5:
        _auto_cleanup_memory()
    return info


_MEM_CLEANUP_INTERVAL = 0  # 마지막 정리 시각

def _auto_cleanup_memory():
    """메모리 압박 시 자동 캐시 정리"""
    global _MEM_CLEANUP_INTERVAL
    now = time.time()
    if now - _MEM_CLEANUP_INTERVAL < 30:  # 30초 내 중복 정리 방지
        return
    _MEM_CLEANUP_INTERVAL = now
    # stale 비디오 세션 정리
    _cleanup_stale_sessions()
    # compare 결과 정리 (실행 중이 아닌 경우)
    if not _compare_state.get("running"):
        _compare_state["results"] = []
    # embedding 이미지 정리
    if not _embedding_state.get("running"):
        _embedding_state["image"] = None
    # 글로벌 팔레트 캐시 축소
    global _palette_cache
    _palette_cache = _palette_cache[:20] if len(_palette_cache) > 20 else _palette_cache
    import gc
    gc.collect()
    print("[Memory] Auto cleanup triggered")


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

# 모든 비동기 작업 상태 레지스트리 — 강제 중지용
_all_states = {
    "eval": _eval_state, "bench": _bench_state, "compare": _compare_state,
    "error_analysis": _error_analysis_state, "conf_opt": _conf_opt_state,
    "embedding": _embedding_state,
}

@app.post("/api/force-stop/{task_id}")
async def force_stop(task_id: str):
    """비동기 작업 강제 중지 — running 플래그를 False로 리셋"""
    if task_id == "all":
        for s in _all_states.values():
            s["running"] = False
        return {"ok": True, "msg": "All tasks stopped"}
    state = _all_states.get(task_id)
    if not state:
        return {"error": f"Unknown task: {task_id}"}
    state["running"] = False
    state["msg"] = "Stopped by user"
    return {"ok": True}


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
        finally:
            _bench_state["running"] = False
            if _bench_state["msg"] != "Error":
                _bench_state["msg"] = "Complete"

    _executor.submit(_run)
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
    if not _check_gpu_available():
        return {"gpu_name": "N/A", "gpu_driver": "N/A", "gpu_memory_gb": 0, "pci_bus": "N/A"}
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
    """벤치마크 결과를 CSV로 다운로드 (시스템 정보 포함, 정리된 포맷)"""
    import io, csv, psutil
    if not _bench_state["results"]:
        return {"error": "No results"}
    buf = io.StringIO()
    # BOM for Excel 한글 호환
    buf.write('\ufeff')
    w = csv.writer(buf)
    # ── System Info ──
    gpu = _get_gpu_info()
    ort_ver = "N/A"
    try:
        import onnxruntime; ort_ver = onnxruntime.__version__
    except ImportError:
        pass
    w.writerow(["=== System Information ==="])
    w.writerow(["OS", f"{platform.system()} {platform.release()}"])
    w.writerow(["CPU", platform.processor()])
    w.writerow(["RAM (GB)", round(psutil.virtual_memory().total / (1024**3), 1)])
    w.writerow(["GPU", gpu["gpu_name"]])
    w.writerow(["Python", platform.python_version()])
    w.writerow(["ONNX Runtime", ort_ver])
    w.writerow([])
    # ── Benchmark Results ──
    w.writerow(["=== Benchmark Results ==="])
    # 정리된 헤더
    header = ["Model", "Provider", "FPS", "Avg (ms)", "Pre (ms)", "Infer (ms)", "Post (ms)",
              "Min (ms)", "Max (ms)", "Std (ms)", "P50 (ms)", "P95 (ms)", "P99 (ms)",
              "CPU (%)", "RAM (MB)", "GPU (%)", "VRAM Used (MB)", "VRAM Total (MB)"]
    field_keys = ["name", "provider", "fps", "avg", "pre_ms", "infer_ms", "post_ms",
                  "min", "max", "std", "p50", "p95", "p99",
                  "cpu_pct", "ram_mb", "gpu_pct", "gpu_mem_used", "gpu_mem_total"]
    w.writerow(header)
    for r in _bench_state["results"]:
        if "error" in r:
            w.writerow([r.get("error", "Error")])
        else:
            w.writerow([r.get(k, "") for k in field_keys])
    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="text/csv; charset=utf-8-sig",
                    headers={"Content-Disposition": "attachment; filename=benchmark_results.csv"})


@app.get("/api/evaluation/export-csv")
async def evaluation_export_csv():
    """평가 결과를 CSV로 다운로드 (정리된 포맷)"""
    import io, csv
    if not _eval_state["results"]:
        return {"error": "No results"}
    buf = io.StringIO()
    buf.write('\ufeff')  # BOM for Excel 한글 호환
    w = csv.writer(buf)
    # ── Summary ──
    w.writerow(["=== Evaluation Summary ==="])
    w.writerow(["Model", "mAP@50 (%)", "mAP@50:95 (%)", "Precision (%)", "Recall (%)", "F1 (%)"])
    for r in _eval_state["results"]:
        w.writerow([r.get("name",""), r.get("map50",""), r.get("map5095",""),
                     r.get("precision",""), r.get("recall",""), r.get("f1","")])
    w.writerow([])
    # ── Per-class Detail ──
    for r in _eval_state["results"]:
        detail = r.get("detail", {})
        if not detail:
            continue
        w.writerow([f"=== {r.get('name','')} — Per-Class Detail ==="])
        w.writerow(["Class ID", "AP", "Precision", "Recall", "F1", "TP", "FP", "FN"])
        for cid, v in sorted(detail.items(), key=lambda x: int(x[0])):
            w.writerow([cid, v.get("ap",""), v.get("precision",""), v.get("recall",""),
                         v.get("f1",""), v.get("tp",""), v.get("fp",""), v.get("fn","")])
        w.writerow([])
    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="text/csv; charset=utf-8-sig",
                    headers={"Content-Disposition": "attachment; filename=evaluation_results.csv"})


# ── EP Status ───────────────────────────────────────────
@app.get("/api/system/ep")
async def system_ep():
    from core.ep_selector import get_ep_status
    return get_ep_status()

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
    if _check_gpu_available():
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
    else:
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


class BrowseRequest(BaseModel):
    path: Optional[str] = None
    exts: Optional[list[str]] = None
    mode: Optional[str] = "file"  # "file" | "dir"


@app.post("/api/fs/browse")
async def browse_fs(req: BrowseRequest):
    """Web-based filesystem browser: returns dirs + files for a given path."""
    if not req.path:
        # Default: server working directory (where exe/script was launched)
        cwd = str(ROOT)
        return {"current": cwd, "parent": str(Path(cwd).parent),
                "entries": _list_entries(cwd, req.exts, req.mode)}

    p = Path(req.path).resolve()
    if not p.is_dir():
        return {"error": "Not a directory"}
    return {"current": str(p), "parent": str(p.parent) if str(p) != str(p.parent) else "",
            "entries": _list_entries(str(p), req.exts, req.mode)}


def _list_entries(dir_path: str, exts: list | None, mode: str) -> list:
    entries = []
    try:
        for item in sorted(Path(dir_path).iterdir()):
            try:
                if item.name.startswith('.'):
                    continue
                if item.is_dir():
                    entries.append({"name": item.name, "path": str(item.resolve()), "type": "dir"})
                elif mode == "file" and item.is_file():
                    if exts and item.suffix.lower() not in exts:
                        continue
                    entries.append({"name": item.name, "path": str(item.resolve()), "type": "file"})
            except (PermissionError, OSError):
                continue
    except (PermissionError, OSError):
        pass
    return entries


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
                    frame = _imread(fp)
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
def _glob_images(img_dir, recursive=False):
    import glob
    files = []
    for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(glob.glob(os.path.join(img_dir, e)))
        if recursive:
            files.extend(glob.glob(os.path.join(img_dir, "**", e), recursive=True))
    files = list(dict.fromkeys(files))  # deduplicate preserving order
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


# Segmentation color palette (20 colors)
_SEG_PALETTE = [
    (128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128),
    (0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0),
    (192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128),
    (0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128),
]

def _overlay_segmentation(frame, mask, alpha=0.5):
    """Segmentation mask를 프레임에 오버레이"""
    h, w = frame.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = frame.copy()
    for cid in np.unique(mask_resized):
        if cid == 0:
            continue
        color = _SEG_PALETTE[cid % len(_SEG_PALETTE)]
        overlay[mask_resized == cid] = color
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


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

    # 이전 비교 임시 파일 정리
    import tempfile, shutil
    _cmp_dir = os.path.join(tempfile.gettempdir(), "ssook_compare")
    if os.path.isdir(_cmp_dir):
        shutil.rmtree(_cmp_dir, ignore_errors=True)
    os.makedirs(_cmp_dir, exist_ok=True)
    _compare_state["_tmp_dir"] = _cmp_dir

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
        try:
            for i, fp in enumerate(imgs):
                frame = _imread(fp)
                if frame is None:
                    _compare_state["progress"] = i + 1
                    continue
                res_a = run_inference(mi_a, frame, req.conf)
                res_b = run_inference(mi_b, frame, req.conf)
                vis_a = _draw_detections(frame, res_a, names_a)
                vis_b = _draw_detections(frame, res_b, names_b)
                # 이미지를 임시 파일로 저장 (메모리 대신 디스크)
                path_a = os.path.join(_cmp_dir, f"{i}_a.jpg")
                path_b = os.path.join(_cmp_dir, f"{i}_b.jpg")
                cv2.imwrite(path_a, vis_a, [cv2.IMWRITE_JPEG_QUALITY, 80])
                cv2.imwrite(path_b, vis_b, [cv2.IMWRITE_JPEG_QUALITY, 80])
                _compare_state["results"].append({
                    "image_name": os.path.basename(fp),
                    "_path_a": path_a,
                    "_path_b": path_b,
                    "count_a": len(res_a.boxes),
                    "count_b": len(res_b.boxes),
                    "ms_a": round(res_a.infer_ms, 2),
                    "ms_b": round(res_b.infer_ms, 2),
                })
                _compare_state["progress"] = i + 1
                _compare_state["msg"] = f"{i+1}/{len(imgs)}"
            _compare_state.update(running=False, msg="Complete")
        except Exception as e:
            _compare_state.update(running=False, msg=f"Error: {e}")

    _executor.submit(_run)
    return {"ok": True}


@app.get("/api/analysis/model-compare/status")
async def model_compare_status():
    """상태 반환 — 이미지는 개별 요청으로 로드"""
    state = dict(_compare_state)
    # 내부 경로 정보 제거, 메타데이터만 반환
    clean_results = []
    for r in state.get("results", []):
        clean = {k: v for k, v in r.items() if not k.startswith("_")}
        clean_results.append(clean)
    state["results"] = clean_results
    return state


@app.get("/api/analysis/model-compare/image/{index}/{side}")
async def model_compare_image(index: int, side: str):
    """비교 이미지를 개별 로드 (side: 'a' or 'b')"""
    results = _compare_state.get("results", [])
    if index < 0 or index >= len(results):
        return {"error": "Invalid index"}
    r = results[index]
    key = f"_path_{side}"
    path = r.get(key)
    if not path or not os.path.isfile(path):
        # fallback: 이전 방식 호환 (base64가 있으면 반환)
        b64_key = f"img_{side}_b64"
        if b64_key in r:
            return {"image": r[b64_key]}
        return {"error": "Image not found"}
    with open(path, "rb") as f:
        img_data = f.read()
    return {"image": base64.b64encode(img_data).decode()}


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

        try:
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
                frame = _imread(fp)
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
        except Exception as e:
            _error_analysis_state.update(running=False, msg=f"Error: {e}")

    _executor.submit(_run)
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

        try:
            # Collect all detections at low conf and all GT
            all_preds = []  # (class_id, score, x1, y1, x2, y2, img_idx)
            all_gt = []     # (class_id, x1, y1, x2, y2, img_idx)
            _conf_opt_state["total"] = len(imgs)
            _conf_opt_state["msg"] = "Running inference..."

            for idx, fp in enumerate(imgs):
                frame = _imread(fp)
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

                pr_curve = []  # store (threshold, precision, recall, f1) for PR curve
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
                    pr_curve.append({"t": round(float(t), 3), "p": round(prec, 4), "r": round(rec, 4), "f1": round(f1, 4)})

                results.append({
                    "class_id": cid,
                    "class_name": names.get(cid, str(cid)),
                    "best_threshold": round(best_t, 3),
                    "best_f1": round(best_f1, 4),
                    "precision": round(best_p, 4),
                    "recall": round(best_r, 4),
                    "pr_curve": pr_curve,
                })

            _conf_opt_state["results"] = results
            _conf_opt_state.update(running=False, msg="Complete")
        except Exception as e:
            _conf_opt_state.update(running=False, msg=f"Error: {e}")

    _executor.submit(_run)
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
            img = _imread(str(fp))
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
        # 임시 파일로 저장 (메모리 대신 디스크)
        import tempfile
        tmp_path = os.path.join(tempfile.gettempdir(), "ssook_embedding.png")
        with open(tmp_path, "wb") as f:
            f.write(buf.read())
        _embedding_state["_image_path"] = tmp_path
        _embedding_state["image"] = None  # 메모리에서 제거
        _embedding_state.update(running=False, msg="Complete")

    _executor.submit(_run)
    return {"ok": True}


@app.get("/api/analysis/embedding-viewer/status")
async def embedding_viewer_status():
    state = dict(_embedding_state)
    # 이미지가 임시 파일에 있으면 로드하여 반환
    img_path = state.pop("_image_path", None)
    if img_path and os.path.isfile(img_path) and state.get("image") is None:
        with open(img_path, "rb") as f:
            state["image"] = base64.b64encode(f.read()).decode()
    return state


    if img_path and os.path.isfile(img_path) and state.get("image") is None:
        with open(img_path, "rb") as f:
            state["image"] = base64.b64encode(f.read()).decode()
    return state


# ── CLIP Zero-Shot API ─────────────────────────────────
class CLIPRequest(BaseModel):
    image_encoder: str
    text_encoder: str
    img_dir: str
    labels: str  # comma-separated

_clip_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": []}
_all_states["clip"] = _clip_state

@app.post("/api/clip/run")
async def run_clip(req: CLIPRequest):
    if _clip_state["running"]:
        return {"error": "Already running"}
    _clip_state.update(running=True, progress=0, total=0, msg="Starting...", results=[])

    def _run():
        try:
            from core.clip_inference import CLIPModel, simple_tokenize
            model = CLIPModel(req.image_encoder, req.text_encoder)
            labels = [l.strip() for l in req.labels.split(",") if l.strip()]
            if not labels:
                _clip_state.update(running=False, msg="No labels provided")
                return
            # Pre-encode text
            text_embs = []
            for label in labels:
                tokens = simple_tokenize(label)
                text_embs.append(model.encode_text(tokens))
            imgs = _glob_images(req.img_dir)
            if not imgs:
                _clip_state.update(running=False, msg="No images found")
                return
            _clip_state["total"] = len(imgs)
            _clip_state["msg"] = "Running CLIP inference..."
            # Per-label correct count
            label_correct = {l: 0 for l in labels}
            label_total = {l: 0 for l in labels}
            detail_log = []  # per-image detail
            for idx, fp in enumerate(imgs):
                frame = _imread(fp)
                if frame is None:
                    _clip_state["progress"] = idx + 1
                    continue
                ranked = model.zero_shot_classify(frame, text_embs, labels)
                # 폴더명 = GT label
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
                _clip_state["progress"] = idx + 1
            # Results
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
            _clip_state["results"] = results
            _clip_state["detail"] = detail_log[:500]  # 최대 500개
            _clip_state.update(running=False, msg="Complete")
        except Exception as e:
            _clip_state.update(running=False, msg=f"Error: {e}")

    _executor.submit(_run)
    return {"ok": True}

@app.get("/api/clip/status")
async def clip_status():
    return dict(_clip_state)


# ── Embedder Evaluation API ────────────────────────────
class EmbedderRequest(BaseModel):
    model_path: str
    model_type: str = "yolo"
    img_dir: str
    top_k: int = 5

_embedder_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": [], "detail": []}
_all_states["embedder"] = _embedder_state

@app.post("/api/embedder/run")
async def run_embedder(req: EmbedderRequest):
    if _embedder_state["running"]:
        return {"error": "Already running"}
    _embedder_state.update(running=True, progress=0, total=0, msg="Starting...", results=[], detail=[])

    def _run():
        try:
            from core.model_loader import load_model as _load
            mi = _load(req.model_path, model_type=req.model_type)
            # 폴더 구조: img_dir/class_name/image.jpg (없으면 단일 클래스로 처리)
            class_dirs = [d for d in os.listdir(req.img_dir)
                          if os.path.isdir(os.path.join(req.img_dir, d))]
            # Collect embeddings per class
            class_embeddings = {}  # class_name -> list of embeddings
            all_files = []
            if class_dirs:
                for cls in sorted(class_dirs):
                    cls_path = os.path.join(req.img_dir, cls)
                    files = _glob_images(cls_path)
                    for f in files:
                        all_files.append((cls, f))
            else:
                # 폴더 구조 없음 — 루트의 이미지를 단일 클래스로
                files = _glob_images(req.img_dir)
                for f in files:
                    all_files.append(("default", f))
            if not all_files:
                _embedder_state.update(running=False, msg="No images found")
                return
            _embedder_state["total"] = len(all_files)
            _embedder_state["msg"] = "Extracting embeddings..."
            embeddings = []  # (class_name, embedding)
            for idx, (cls, fp) in enumerate(all_files):
                frame = _imread(fp)
                if frame is None:
                    _embedder_state["progress"] = idx + 1
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
                _embedder_state["progress"] = idx + 1
            # Compute retrieval metrics
            import numpy as np
            _embedder_state["msg"] = "Computing metrics..."
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
            _embedder_state["results"] = results
            _embedder_state["detail"] = detail_log
            _embedder_state.update(running=False, msg="Complete")
        except Exception as e:
            import traceback
            traceback.print_exc()
            _embedder_state.update(running=False, msg=f"Error: {e}")

    _executor.submit(_run)
    return {"ok": True}

@app.get("/api/embedder/status")
async def embedder_status():
    return dict(_embedder_state)

@app.post("/api/embedder/compare")
async def embedder_compare(req: dict):
    """Compare embeddings of multiple selected images using the loaded embedder model."""
    model_path = req.get("model_path", "")
    img_paths = req.get("img_paths", [])
    if not model_path or len(img_paths) < 2:
        return {"error": "Need model and at least 2 images"}
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(model_path)
        inp = session.get_inputs()[0]
        h = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 224
        w = int(inp.shape[3]) if isinstance(inp.shape[3], int) else 224
        embeddings = []
        names = []
        for fp in img_paths:
            img = _imread(fp)
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
        # Compute pairwise cosine similarity matrix
        matrix = []
        for i in range(len(embeddings)):
            row = []
            for j in range(len(embeddings)):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                row.append(round(sim, 4))
            matrix.append(row)
        return {"names": names, "matrix": matrix}
    except Exception as e:
        return {"error": str(e)}


# ── Segmentation Evaluation API ─────────────────────────
class SegmentationRequest(BaseModel):
    model_path: str
    img_dir: str
    label_dir: str = ""

_seg_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": [], "detail": []}
_all_states["seg"] = _seg_state

@app.post("/api/segmentation/run")
async def run_segmentation_eval(req: SegmentationRequest):
    if _seg_state["running"]:
        return {"error": "Already running"}
    _seg_state.update(running=True, progress=0, total=0, msg="Starting...", results=[], detail=[])

    def _run():
        try:
            from core.model_loader import load_model as _load
            mi = _load(req.model_path, model_type="yolo")
            imgs = _glob_images(req.img_dir)
            if not imgs:
                _seg_state.update(running=False, msg="No images found")
                return
            _seg_state["total"] = len(imgs)
            _seg_state["msg"] = "Running segmentation..."

            # Per-class IoU/Dice accumulation
            class_iou_sum = {}
            class_dice_sum = {}
            class_count = {}
            detail_log = []

            for idx, fp in enumerate(imgs):
                frame = _imread(fp)
                if frame is None:
                    _seg_state["progress"] = idx + 1
                    continue
                h, w = frame.shape[:2]

                # Run inference
                from core.inference import letterbox, _padded_to_tensor
                padded, ratio, pad = letterbox(frame, mi.input_size)
                tensor = _padded_to_tensor(padded, mi.input_size)
                if mi.batch_size > 1:
                    tensor = np.repeat(tensor, mi.batch_size, axis=0)
                outputs = mi.session.run(None, {mi.input_name: tensor})

                # 세그멘테이션 출력: (1, C, H, W) 또는 (1, H, W)
                seg_out = outputs[0][0] if len(outputs[0].shape) == 4 else outputs[0]
                if len(seg_out.shape) == 3:
                    pred_mask = np.argmax(seg_out, axis=0)  # (H, W)
                else:
                    pred_mask = (seg_out > 0.5).astype(np.int32)

                # Resize pred to original size
                pred_mask = cv2.resize(pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

                # Load GT mask (PNG with class IDs as pixel values)
                stem = os.path.splitext(os.path.basename(fp))[0]
                gt_path = None
                for ext in ['.png', '.bmp', '.tif']:
                    candidate = os.path.join(req.label_dir, stem + ext)
                    if os.path.isfile(candidate):
                        gt_path = candidate
                        break
                if gt_path is None:
                    _seg_state["progress"] = idx + 1
                    continue
                gt_mask = _imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is None:
                    _seg_state["progress"] = idx + 1
                    continue
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                # Per-class IoU/Dice
                all_classes = set(np.unique(gt_mask)) | set(np.unique(pred_mask))
                all_classes.discard(0)  # background
                for c in all_classes:
                    pred_c = (pred_mask == c)
                    gt_c = (gt_mask == c)
                    inter = np.logical_and(pred_c, gt_c).sum()
                    union = np.logical_or(pred_c, gt_c).sum()
                    iou = inter / (union + 1e-9)
                    dice = 2 * inter / (pred_c.sum() + gt_c.sum() + 1e-9)
                    class_iou_sum[c] = class_iou_sum.get(c, 0) + iou
                    class_dice_sum[c] = class_dice_sum.get(c, 0) + dice
                    class_count[c] = class_count.get(c, 0) + 1

                # Per-image detail
                img_iou = []
                for c in all_classes:
                    pred_c = (pred_mask == c)
                    gt_c = (gt_mask == c)
                    inter = np.logical_and(pred_c, gt_c).sum()
                    union = np.logical_or(pred_c, gt_c).sum()
                    img_iou.append(round(float(inter / (union + 1e-9)), 4))
                mean_iou = round(sum(img_iou) / len(img_iou), 4) if img_iou else 0
                entry = {"file": os.path.basename(fp), "iou": mean_iou, "classes": len(all_classes)}
                # Sample overlay for first 5 images
                if len(detail_log) < 5:
                    vis = _overlay_segmentation(frame, pred_mask)
                    entry["overlay"] = _encode_jpeg(vis, 60)
                detail_log.append(entry)

                _seg_state["progress"] = idx + 1
                _seg_state["msg"] = f"{idx+1}/{len(imgs)}"

            # Build results
            names = mi.names or {}
            results = []
            for c in sorted(class_count.keys()):
                n = class_count[c]
                results.append({
                    "class_name": names.get(c, str(c)),
                    "iou": round(class_iou_sum[c] / n, 4),
                    "dice": round(class_dice_sum[c] / n, 4),
                    "images": n,
                })
            if results:
                miou = round(sum(r["iou"] for r in results) / len(results), 4)
                mdice = round(sum(r["dice"] for r in results) / len(results), 4)
                results.append({"class_name": "Overall (mean)", "iou": miou, "dice": mdice, "images": len(imgs)})
            _seg_state["results"] = results
            _seg_state["detail"] = detail_log[:200]
            _seg_state.update(running=False, msg="Complete")
        except Exception as e:
            _seg_state.update(running=False, msg=f"Error: {e}")

    _executor.submit(_run)
    return {"ok": True}

@app.get("/api/segmentation/status")
async def segmentation_status():
    return dict(_seg_state)


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
        frame = _imread(req.image_path)
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


# ── Data: Explorer API ──────────────────────────────────
_explorer_state = {"running": False, "progress": 0, "total": 0, "msg": "", "data": None}
_all_states["explorer"] = _explorer_state

@app.post("/api/data/explorer")
async def data_explorer(req: dict):
    img_dir = req.get("img_dir", "")
    lbl_dir = req.get("label_dir", "")
    if not img_dir or not os.path.isdir(img_dir):
        return {"error": "Invalid image directory"}
    if _explorer_state["running"]:
        return {"error": "Already loading"}
    _explorer_state.update(running=True, progress=0, total=0, msg="Scanning...", data=None)

    def _run():
        try:
            imgs = _glob_images(img_dir)
            n = len(imgs)
            _explorer_state["total"] = n
            class_counts = {}
            img_class_counts = {}  # class -> set of image indices (for image-unit counting)
            file_info = []
            box_sizes = []  # (w, h) normalized
            aspect_ratios = []
            box_aspect_ratios = []
            for i, fp in enumerate(imgs[:5000]):
                _explorer_state["progress"] = i + 1
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
                    im = _imread(fp)
                    if im is not None:
                        h_, w_ = im.shape[:2]
                        aspect_ratios.append(round(w_ / max(h_, 1), 2))
                except:
                    pass
                file_info.append({"name": os.path.basename(fp), "path": fp, "boxes": len(boxes),
                                  "classes": list(set(boxes)), "box_details": box_details})
            img_class_count_dict = {k: len(v) for k, v in img_class_counts.items()}
            _explorer_state["data"] = {
                "total": n, "shown": len(file_info), "files": file_info,
                "class_counts": class_counts, "img_class_counts": img_class_count_dict,
                "box_sizes": box_sizes, "aspect_ratios": aspect_ratios,
                "box_aspect_ratios": box_aspect_ratios
            }
            _explorer_state.update(running=False, msg="Complete")
        except Exception as e:
            _explorer_state.update(running=False, msg=f"Error: {e}")

    _executor.submit(_run)
    return {"ok": True}

@app.get("/api/data/explorer/status")
async def explorer_status():
    s = dict(_explorer_state)
    if not s["running"] and s["data"]:
        data = s["data"]
        s.pop("data")
        s.update(data)
    elif s["running"]:
        s.pop("data", None)
    return s

@app.post("/api/data/explorer/preview")
async def explorer_preview(req: dict):
    """Return base64 JPEG of image with bounding boxes overlaid."""
    img_path = req.get("img_path", "")
    lbl_dir = req.get("label_dir", "")
    if not img_path or not os.path.isfile(img_path):
        return {"error": "Image not found"}
    img = _imread(img_path)
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
    return {"image": _encode_jpeg(img, 90), "width": w, "height": h, "box_count": len(boxes)}


# ── Data: Splitter API ─────────────────────────────────
class SplitterRequest(BaseModel):
    img_dir: str
    label_dir: str = ""
    output_dir: str
    train: float = 0.7
    val: float = 0.2
    test: float = 0.1
    strategy: str = "random"  # random, stratified, similarity

_splitter_state = {"running": False, "msg": "", "progress": 0, "total": 0, "results": {}}
_all_states["splitter"] = _splitter_state

@app.post("/api/data/splitter")
async def data_splitter(req: SplitterRequest):
    if _splitter_state["running"]:
        return {"error": "Already running"}
    _splitter_state.update(running=True, msg="Splitting...", progress=0, total=0, results={})

    def _run():
        try:
            import random, shutil
            imgs = _glob_images(req.img_dir)
            if not imgs:
                _splitter_state.update(running=False, msg="No images found")
                return
            n = len(imgs)
            _splitter_state["total"] = n

            # Normalize ratios — treat 0 as empty split
            ratios = {"train": max(req.train, 0), "val": max(req.val, 0), "test": max(req.test, 0)}
            total_ratio = sum(ratios.values())
            if total_ratio <= 0:
                _splitter_state.update(running=False, msg="Error: All ratios are 0")
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
                for group_imgs in class_groups.values():
                    random.shuffle(group_imgs)
                    gn = len(group_imgs)
                    n_train = int(gn * ratios["train"] / total_ratio)
                    n_val = int(gn * ratios["val"] / total_ratio)
                    splits["train"].extend(group_imgs[:n_train])
                    splits["val"].extend(group_imgs[n_train:n_train + n_val])
                    splits["test"].extend(group_imgs[n_train + n_val:])
            else:
                # Random split
                random.shuffle(imgs)
                n_train = int(n * ratios["train"] / total_ratio)
                n_val = int(n * ratios["val"] / total_ratio)
                splits = {"train": imgs[:n_train], "val": imgs[n_train:n_train + n_val], "test": imgs[n_train + n_val:]}

            # Copy files with progress
            total_files = sum(len(v) for v in splits.values())
            _splitter_state["total"] = total_files
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
                    _splitter_state["progress"] = done
            _splitter_state["results"] = {k: len(v) for k, v in splits.items()}
            _splitter_state.update(running=False, msg="Complete")
        except Exception as e:
            _splitter_state.update(running=False, msg=f"Error: {e}")

    _executor.submit(_run)
    return {"ok": True}

@app.get("/api/data/splitter/status")
async def splitter_status():
    return dict(_splitter_state)


# ── Data: Converter API ────────────────────────────────
class ConverterRequest(BaseModel):
    input_dir: str
    output_dir: str
    from_fmt: str = "YOLO"
    to_fmt: str = "COCO JSON"

_converter_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": {}}
_all_states["converter"] = _converter_state

@app.post("/api/data/converter")
async def data_converter(req: ConverterRequest):
    if _converter_state["running"]:
        return {"error": "Already running"}
    _converter_state.update(running=True, progress=0, total=0, msg="Converting...", results={})

    def _run():
        try:
            import json as _json
            os.makedirs(req.output_dir, exist_ok=True)
            label_files = sorted(glob_module.glob(os.path.join(req.input_dir, "*.txt")))
            img_files = _glob_images(req.input_dir)
            _converter_state["total"] = len(label_files) or len(img_files)

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
                        img = _imread(img_path)
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
                    _converter_state["progress"] = i + 1
                for c in sorted(cats_seen):
                    coco["categories"].append({"id": c, "name": str(c)})
                with open(os.path.join(req.output_dir, "annotations.json"), "w") as f:
                    _json.dump(coco, f, indent=2)
                _converter_state["results"] = {"images": len(coco["images"]), "annotations": ann_id - 1}

            elif "COCO" in req.from_fmt and req.to_fmt == "YOLO":
                json_files = glob_module.glob(os.path.join(req.input_dir, "*.json"))
                if not json_files:
                    _converter_state.update(running=False, msg="No JSON files found")
                    return
                with open(json_files[0]) as f:
                    coco = _json.load(f)
                img_map = {img["id"]: img for img in coco.get("images", [])}
                _converter_state["total"] = len(coco.get("annotations", []))
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
                    _converter_state["progress"] = idx + 1
                for iid, lines in per_image.items():
                    img_info = img_map.get(iid, {})
                    stem = os.path.splitext(img_info.get("file_name", str(iid)))[0]
                    with open(os.path.join(req.output_dir, stem + ".txt"), "w") as f:
                        f.write("\n".join(lines) + "\n")
                _converter_state["results"] = {"images": len(per_image), "labels": sum(len(v) for v in per_image.values())}

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
                        img = _imread(img_path)
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
                    _converter_state["progress"] = i + 1
                _converter_state["results"] = {"files": len(label_files), "objects": count}
            else:
                _converter_state["results"] = {"error": f"Unsupported: {req.from_fmt} → {req.to_fmt}"}

            _converter_state.update(running=False, msg="Complete")
        except Exception as e:
            _converter_state.update(running=False, msg=f"Error: {e}")

    _executor.submit(_run)
    return {"ok": True}

@app.get("/api/data/converter/status")
async def converter_status():
    return dict(_converter_state)


# ── Data: Remapper API ─────────────────────────────────
class RemapperRequest(BaseModel):
    label_dir: str
    output_dir: str
    mapping: dict = {}  # {"old_id": "new_id"}
    auto_reindex: bool = True
    recursive: bool = False

_remapper_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": {}}
_all_states["remapper"] = _remapper_state

@app.post("/api/data/remapper")
async def data_remapper(req: RemapperRequest):
    if _remapper_state["running"]:
        return {"error": "Already running"}
    _remapper_state.update(running=True, progress=0, total=0, msg="Remapping...", results={})

    def _run():
        try:
            os.makedirs(req.output_dir, exist_ok=True)
            label_files = sorted(glob_module.glob(os.path.join(req.label_dir, "*.txt")))
            if req.recursive:
                label_files += sorted(glob_module.glob(os.path.join(req.label_dir, "**", "*.txt"), recursive=True))
                label_files = list(dict.fromkeys(label_files))
            if not label_files:
                _remapper_state.update(running=False, msg="No label files found")
                return
            mapping = {int(k): int(v) for k, v in req.mapping.items()} if req.mapping else {}
            _remapper_state["total"] = len(label_files)
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
                _remapper_state["progress"] = idx + 1
            _remapper_state["results"] = {"files": len(label_files), "labels": count}
            _remapper_state.update(running=False, msg="Complete")
        except Exception as e:
            _remapper_state.update(running=False, msg=f"Error: {e}")

    _executor.submit(_run)
    return {"ok": True}

@app.get("/api/data/remapper/status")
async def remapper_status():
    return dict(_remapper_state)


# ── Data: Merger API ───────────────────────────────────
class MergerRequest(BaseModel):
    datasets: list[str]
    output_dir: str
    dhash_threshold: int = 10
    recursive: bool = False

_merger_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": {}}
_all_states["merger"] = _merger_state

@app.post("/api/data/merger")
async def data_merger(req: MergerRequest):
    if _merger_state["running"]:
        return {"error": "Already running"}
    _merger_state.update(running=True, progress=0, total=0, msg="Merging...", results={})

    def _run():
        try:
            import shutil, hashlib
            os.makedirs(os.path.join(req.output_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(req.output_dir, "labels"), exist_ok=True)
            all_imgs = []
            for d in req.datasets:
                all_imgs.extend(_glob_images(d, recursive=req.recursive))
            _merger_state["total"] = len(all_imgs)
            seen_hashes = set()
            copied = 0
            dupes = 0
            for i, fp in enumerate(all_imgs):
                # Simple hash-based dedup
                with open(fp, "rb") as f:
                    h = hashlib.md5(f.read(8192)).hexdigest()
                if h in seen_hashes:
                    dupes += 1
                    _merger_state["progress"] = i + 1
                    continue
                seen_hashes.add(h)
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
                _merger_state["progress"] = i + 1
            _merger_state["results"] = {"total": len(all_imgs), "copied": copied, "duplicates": dupes}
            _merger_state.update(running=False, msg="Complete")
        except Exception as e:
            _merger_state.update(running=False, msg=f"Error: {e}")

    _executor.submit(_run)
    return {"ok": True}

@app.get("/api/data/merger/status")
async def merger_status():
    return dict(_merger_state)


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

_sampler_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": {}}
_all_states["sampler"] = _sampler_state

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

@app.post("/api/data/sampler")
async def data_sampler(req: SamplerRequest):
    if _sampler_state["running"]:
        return {"error": "Already running"}
    _sampler_state.update(running=True, progress=0, total=0, msg="Scanning...", results={})

    def _run():
        try:
            import random, shutil
            import numpy as np
            random.seed(req.seed)
            np.random.seed(req.seed)
            imgs = _glob_images(req.img_dir, recursive=req.recursive)
            if not imgs:
                _sampler_state.update(running=False, msg="No images found")
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
            _sampler_state["total"] = len(selected)
            _sampler_state["msg"] = "Copying..."
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
                _sampler_state["progress"] = i + 1
            _sampler_state["results"] = {"total": len(imgs), "sampled": len(selected),
                                         "before": before_classes, "after": after_classes}
            _sampler_state.update(running=False, msg="Complete")
        except Exception as e:
            _sampler_state.update(running=False, msg=f"Error: {e}")

    _executor.submit(_run)
    return {"ok": True}

@app.get("/api/data/sampler/status")
async def sampler_status():
    return dict(_sampler_state)


# ── Quality: Anomaly Detector API ──────────────────────
_anomaly_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": []}
_all_states["anomaly"] = _anomaly_state

@app.post("/api/quality/anomaly")
async def quality_anomaly(req: dict):
    if _anomaly_state["running"]:
        return {"error": "Already running"}
    _anomaly_state.update(running=True, progress=0, total=0, msg="Scanning...", results=[])
    img_dir = req.get("img_dir", "")
    label_dir = req.get("label_dir", "")
    recursive = req.get("recursive", False)

    def _run():
        try:
            imgs = _glob_images(img_dir, recursive=recursive)
            _anomaly_state["total"] = len(imgs)
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
                _anomaly_state["progress"] = i + 1
            _anomaly_state["results"] = results[:1000]
            _anomaly_state.update(running=False, msg=f"Complete — {len(results)} issues found")
        except Exception as e:
            _anomaly_state.update(running=False, msg=f"Error: {e}")

    _executor.submit(_run)
    return {"ok": True}

@app.get("/api/quality/anomaly/status")
async def anomaly_status():
    return dict(_anomaly_state)


# ── Quality: Image Quality Checker API ─────────────────
_quality_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": []}
_all_states["quality"] = _quality_state

@app.post("/api/quality/image-quality")
async def quality_check(req: dict):
    if _quality_state["running"]:
        return {"error": "Already running"}
    _quality_state.update(running=True, progress=0, total=0, msg="Checking...", results=[])
    img_dir = req.get("img_dir", "")
    recursive = req.get("recursive", False)

    def _run():
        try:
            imgs = _glob_images(img_dir, recursive=recursive)
            _quality_state["total"] = len(imgs)
            results = []
            for i, fp in enumerate(imgs):
                frame = _imread(fp)
                if frame is None:
                    _quality_state["progress"] = i + 1
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
                _quality_state["progress"] = i + 1
            _quality_state["results"] = results[:1000]
            _quality_state.update(running=False, msg=f"Complete — {len(results)} images checked")
        except Exception as e:
            _quality_state.update(running=False, msg=f"Error: {e}")

    _executor.submit(_run)
    return {"ok": True}

@app.get("/api/quality/image-quality/status")
async def quality_status():
    return dict(_quality_state)


# ── Quality: Near-Duplicate Detector API ───────────────
_dup_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": []}
_all_states["duplicate"] = _dup_state

def _dhash(img, size=8):
    resized = cv2.resize(img, (size+1, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    diff = gray[:, 1:] > gray[:, :-1]
    return sum(2**i for i, v in enumerate(diff.flatten()) if v)

@app.post("/api/quality/duplicate")
async def quality_duplicate(req: dict):
    if _dup_state["running"]:
        return {"error": "Already running"}
    _dup_state.update(running=True, progress=0, total=0, msg="Hashing...", results=[])
    img_dir = req.get("img_dir", "")
    threshold = int(req.get("threshold", 10))
    recursive = req.get("recursive", False)

    def _run():
        try:
            imgs = _glob_images(img_dir, recursive=recursive)
            _dup_state["total"] = len(imgs)
            hashes = []
            for i, fp in enumerate(imgs):
                frame = _imread(fp)
                if frame is not None:
                    hashes.append((os.path.basename(fp), _dhash(frame)))
                _dup_state["progress"] = i + 1
            _dup_state["msg"] = "Comparing..."
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
            _dup_state["results"] = results
            _dup_state.update(running=False, msg=f"Complete — {len(results)} pairs found")
        except Exception as e:
            _dup_state.update(running=False, msg=f"Error: {e}")

    _executor.submit(_run)
    return {"ok": True}

@app.get("/api/quality/duplicate/status")
async def duplicate_status():
    return dict(_dup_state)


# ── Quality: Leaky Split Detector API ──────────────────
_leaky_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": []}
_all_states["leaky"] = _leaky_state

@app.post("/api/quality/leaky")
async def quality_leaky(req: dict):
    if _leaky_state["running"]:
        return {"error": "Already running"}
    _leaky_state.update(running=True, progress=0, total=0, msg="Scanning...", results=[])
    dirs = {k: req.get(k, "") for k in ["train_dir", "val_dir", "test_dir"]}
    threshold = int(req.get("threshold", 10))

    def _run():
        try:
            split_hashes = {}
            for name, d in dirs.items():
                if not d:
                    continue
                imgs = _glob_images(d)
                split_hashes[name] = {os.path.basename(fp): _dhash(_imread(fp)) for fp in imgs if _imread(fp) is not None}
            _leaky_state["total"] = sum(len(v) for v in split_hashes.values())
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
            _leaky_state["results"] = results
            _leaky_state.update(running=False, msg="Complete")
        except Exception as e:
            _leaky_state.update(running=False, msg=f"Error: {e}")

    _executor.submit(_run)
    return {"ok": True}

@app.get("/api/quality/leaky/status")
async def leaky_status():
    return dict(_leaky_state)


# ── Quality: Similarity Search API ─────────────────────
_sim_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": [], "index": None}
_all_states["similarity"] = _sim_state

@app.post("/api/quality/similarity")
async def quality_similarity(req: dict):
    if _sim_state["running"]:
        return {"error": "Already running"}
    _sim_state.update(running=True, progress=0, total=0, msg="Building index...", results=[])
    img_dir = req.get("img_dir", "")
    query = req.get("query", "")
    top_k = int(req.get("top_k", 10))

    def _run():
        try:
            imgs = _glob_images(img_dir)
            _sim_state["total"] = len(imgs)
            hashes = []
            for i, fp in enumerate(imgs):
                frame = _imread(fp)
                if frame is not None:
                    hashes.append((os.path.basename(fp), _dhash(frame, 16)))
                _sim_state["progress"] = i + 1
            if query and os.path.isfile(query):
                q_frame = _imread(query)
                q_hash = _dhash(q_frame, 16) if q_frame is not None else 0
                ranked = sorted(hashes, key=lambda x: bin(x[1] ^ q_hash).count('1'))
                _sim_state["results"] = [{"rank": i+1, "image": name, "distance": bin(h ^ q_hash).count('1')} for i, (name, h) in enumerate(ranked[:top_k])]
            else:
                _sim_state["results"] = [{"rank": i+1, "image": name, "distance": 0} for i, (name, _) in enumerate(hashes[:top_k])]
            _sim_state.update(running=False, msg="Complete")
        except Exception as e:
            _sim_state.update(running=False, msg=f"Error: {e}")

    _executor.submit(_run)
    return {"ok": True}

@app.get("/api/quality/similarity/status")
async def similarity_status():
    return dict(_sim_state)


# ── Batch: Augmentation Preview API ────────────────────
@app.post("/api/batch/augmentation")
async def batch_augmentation(req: dict):
    try:
        img_dir = req.get("img_dir", "")
        aug_type = req.get("aug_type", "Flip")
        imgs = _glob_images(img_dir)
        if not imgs:
            return {"error": "No images found"}
        import random
        fp = random.choice(imgs)
        frame = _imread(fp)
        if frame is None:
            return {"error": "Cannot read image"}
        original = _encode_jpeg(frame)
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
            samples = [_imread(random.choice(imgs)) for _ in range(4)]
            samples = [cv2.resize(s, (half_w, half_h)) if s is not None else np.zeros((half_h, half_w, 3), dtype=np.uint8) for s in samples]
            top = np.hstack([samples[0], samples[1]])
            bot = np.hstack([samples[2], samples[3]])
            aug = np.vstack([top, bot])
        else:
            aug = frame.copy()
        augmented = _encode_jpeg(aug)
        return {"original": original, "augmented": augmented, "file": os.path.basename(fp)}
    except Exception as e:
        return {"error": str(e)}


# ── Phase 1: Model Inspector API ───────────────────────
@app.post("/api/inspector/inspect")
async def api_inspect_model(req: dict):
    try:
        path = req.get("path", "")
        if not path or not os.path.isfile(path):
            return {"error": "Model file not found"}
        from core.model_inspector import inspect_model, inspection_to_dict
        info = inspect_model(path)
        return inspection_to_dict(info)
    except Exception as e:
        return {"error": str(e)}


# ── Phase 1: Model Profiler API ────────────────────────
@app.post("/api/profiler/run")
async def api_profile_model(req: dict):
    try:
        path = req.get("path", "")
        num_runs = req.get("num_runs", 20)
        if not path or not os.path.isfile(path):
            return {"error": "Model file not found"}
        from core.model_profiler import profile_model, profile_to_dict
        result = profile_model(path, num_runs=num_runs)
        return profile_to_dict(result)
    except Exception as e:
        return {"error": str(e)}


# ── Phase 1: Pose Estimation API ───────────────────────
@app.post("/api/infer/pose")
async def api_infer_pose(req: dict):
    try:
        model_path = req.get("model_path", "")
        image_path = req.get("image_path", "")
        conf = req.get("conf", 0.25)
        model_type = req.get("model_type", "pose_yolo")
        if not model_path or not image_path:
            return {"error": "model_path and image_path required"}
        frame = _imread(image_path)
        if frame is None:
            return {"error": "Cannot read image"}
        mi = _load(model_path, model_type=model_type)
        from core.inference import run_pose, COCO_SKELETON, COCO_KPT_NAMES
        result = run_pose(mi, frame, conf)
        # Draw on image
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
            "image": _encode_jpeg(vis),
            "num_persons": len(result.boxes),
            "infer_ms": round(result.infer_ms, 2),
            "detections": [
                {"box": result.boxes[i].tolist(),
                 "score": round(float(result.scores[i]), 3),
                 "keypoints": result.keypoints[i].tolist()}
                for i in range(len(result.boxes))
            ],
        }
    except Exception as e:
        return {"error": str(e)}


# ── Phase 1: Instance Segmentation API ─────────────────
@app.post("/api/infer/instance-seg")
async def api_infer_instance_seg(req: dict):
    try:
        model_path = req.get("model_path", "")
        image_path = req.get("image_path", "")
        conf = req.get("conf", 0.25)
        model_type = req.get("model_type", "instseg_yolo")
        if not model_path or not image_path:
            return {"error": "model_path and image_path required"}
        frame = _imread(image_path)
        if frame is None:
            return {"error": "Cannot read image"}
        mi = _load(model_path, model_type=model_type)
        from core.inference import run_instance_seg
        result = run_instance_seg(mi, frame, conf)
        # Draw masks on image
        vis = frame.copy()
        colors = _generate_palette(max(len(result.masks), 1))
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
            "image": _encode_jpeg(vis),
            "num_instances": len(result.boxes),
            "infer_ms": round(result.infer_ms, 2),
            "detections": [
                {"box": result.boxes[i].tolist(),
                 "score": round(float(result.scores[i]), 3),
                 "class_id": int(result.class_ids[i])}
                for i in range(len(result.boxes))
            ],
        }
    except Exception as e:
        return {"error": str(e)}


# ── Phase 1: Tracking API ──────────────────────────────
_trackers = {}

@app.post("/api/tracking/create")
async def api_tracking_create(req: dict):
    try:
        tracker_type = req.get("tracker_type", "bytetrack")
        from core.tracking import create_tracker
        tid = str(uuid.uuid4())[:8]
        _trackers[tid] = create_tracker(tracker_type)
        return {"tracker_id": tid, "type": tracker_type}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/tracking/reset")
async def api_tracking_reset(req: dict):
    tid = req.get("tracker_id", "")
    if tid in _trackers:
        _trackers[tid].reset()
        return {"status": "ok"}
    return {"error": "Tracker not found"}


# ── HuggingFace Hub API ────────────────────────────────

@app.post("/api/hf/search")
async def hf_search(req: dict):
    try:
        from core.hf_downloader import search_models
        results = search_models(req.get("query", ""), req.get("task", ""), req.get("limit", 20))
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/hf/files")
async def hf_files(req: dict):
    try:
        from core.hf_downloader import list_onnx_files
        files = list_onnx_files(req.get("repo_id", ""))
        return {"files": files}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/hf/download")
async def hf_download(req: dict):
    try:
        from core.hf_downloader import download_model
        path = await asyncio.get_event_loop().run_in_executor(
            _executor, download_model, req["repo_id"], req["filename"]
        )
        return {"path": path}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/hf/cached")
async def hf_cached():
    try:
        from core.hf_downloader import list_cached
        return {"models": list_cached()}
    except Exception as e:
        return {"error": str(e)}


# ── Launch ──────────────────────────────────────────────
def main():
    import uvicorn
    port = int(os.environ.get("PORT", 8765))
    print(f"\n  ssook running at http://localhost:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
