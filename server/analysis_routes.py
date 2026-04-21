"""/api/analysis/* 라우터."""
import base64
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from core.model_loader import load_model as _load_model
from core.inference import run_inference, letterbox, preprocess
from core.evaluation import evaluate_dataset
from server.state import compare_state, error_analysis_state, conf_opt_state, embedding_state, executor
from server.utils import imread, encode_jpeg, draw_detections, glob_images

router = APIRouter()

# ── 1. Model Compare API ───────────────────────────────
class ModelCompareRequest(BaseModel):
    model_a: str
    model_b: str
    model_type_a: str = "yolo"
    model_type_b: str = "yolo"
    img_dir: str
    conf: float = 0.25


@router.post("/api/analysis/model-compare")
async def run_model_compare(req: ModelCompareRequest):
    if compare_state["running"]:
        return {"error": "Already running"}
    compare_state.update(running=True, progress=0, total=0, msg="Starting...", results=[], images=[])

    # 이전 비교 임시 파일 정리
    import tempfile, shutil
    _cmp_dir = os.path.join(tempfile.gettempdir(), "ssook_compare")
    if os.path.isdir(_cmp_dir):
        shutil.rmtree(_cmp_dir, ignore_errors=True)
    os.makedirs(_cmp_dir, exist_ok=True)
    compare_state["_tmp_dir"] = _cmp_dir

    def _run():
        from core.model_loader import load_model as _load
        from core.inference import run_inference
        try:
            mi_a = _load(req.model_a, model_type=req.model_type_a)
            mi_b = _load(req.model_b, model_type=req.model_type_b)
        except Exception as e:
            compare_state.update(running=False, msg=f"Load error: {e}")
            return
        imgs = glob_images(req.img_dir)
        if not imgs:
            compare_state.update(running=False, msg="No images found")
            return
        compare_state["total"] = len(imgs)
        names_a = mi_a.names or {}
        names_b = mi_b.names or {}
        try:
            for i, fp in enumerate(imgs):
                frame = imread(fp)
                if frame is None:
                    compare_state["progress"] = i + 1
                    continue
                res_a = run_inference(mi_a, frame, req.conf)
                res_b = run_inference(mi_b, frame, req.conf)
                vis_a = draw_detections(frame, res_a, names_a)
                vis_b = draw_detections(frame, res_b, names_b)
                # 이미지를 임시 파일로 저장 (메모리 대신 디스크)
                path_a = os.path.join(_cmp_dir, f"{i}_a.jpg")
                path_b = os.path.join(_cmp_dir, f"{i}_b.jpg")
                cv2.imwrite(path_a, vis_a, [cv2.IMWRITE_JPEG_QUALITY, 80])
                cv2.imwrite(path_b, vis_b, [cv2.IMWRITE_JPEG_QUALITY, 80])
                compare_state["results"].append({
                    "image_name": os.path.basename(fp),
                    "_path_a": path_a,
                    "_path_b": path_b,
                    "count_a": len(res_a.boxes),
                    "count_b": len(res_b.boxes),
                    "ms_a": round(res_a.infer_ms, 2),
                    "ms_b": round(res_b.infer_ms, 2),
                })
                compare_state["progress"] = i + 1
                compare_state["msg"] = f"{i+1}/{len(imgs)}"
            compare_state.update(running=False, msg="Complete")
        except Exception as e:
            compare_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}


@router.get("/api/analysis/model-compare/status")
async def model_compare_status():
    """상태 반환 — 이미지는 개별 요청으로 로드"""
    state = dict(compare_state)
    # 내부 경로 정보 제거, 메타데이터만 반환
    clean_results = []
    for r in state.get("results", []):
        clean = {k: v for k, v in r.items() if not k.startswith("_")}
        clean_results.append(clean)
    state["results"] = clean_results
    return state


@router.get("/api/analysis/model-compare/image/{index}/{side}")
async def model_compare_image(index: int, side: str):
    """비교 이미지를 개별 로드 (side: 'a' or 'b')"""
    results = compare_state.get("results", [])
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


@router.post("/api/analysis/error-analysis")
async def run_error_analysis(req: ErrorAnalysisRequest):
    if error_analysis_state["running"]:
        return {"error": "Already running"}
    error_analysis_state.update(running=True, progress=0, total=0, msg="Starting...", results={})

    def _run():
        from core.model_loader import load_model as _load
        from core.inference import run_inference
        try:
            mi = _load(req.model_path, model_type=req.model_type)
        except Exception as e:
            error_analysis_state.update(running=False, msg=f"Load error: {e}")
            return
        imgs = glob_images(req.img_dir)
        if not imgs:
            error_analysis_state.update(running=False, msg="No images found")
            return
        error_analysis_state["total"] = len(imgs)

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
                frame = imread(fp)
                if frame is None:
                    error_analysis_state["progress"] = i + 1
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

                error_analysis_state["progress"] = i + 1
                error_analysis_state["msg"] = f"{i+1}/{len(imgs)}"

            error_analysis_state["results"] = {"fp": fp_stats, "fn": fn_stats}
            error_analysis_state.update(running=False, msg="Complete")
        except Exception as e:
            error_analysis_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}


@router.get("/api/analysis/error-analysis/status")
async def error_analysis_status():
    return dict(error_analysis_state)


# ── 3. Confidence Optimizer API ─────────────────────────
class ConfOptimizerRequest(BaseModel):
    model_path: str
    model_type: str = "yolo"
    img_dir: str
    label_dir: str
    step: float = 0.05


@router.post("/api/analysis/conf-optimizer")
async def run_conf_optimizer(req: ConfOptimizerRequest):
    if conf_opt_state["running"]:
        return {"error": "Already running"}
    conf_opt_state.update(running=True, progress=0, total=0, msg="Starting...", results=[])

    def _run():
        from core.model_loader import load_model as _load
        from core.inference import run_inference
        try:
            mi = _load(req.model_path, model_type=req.model_type)
        except Exception as e:
            conf_opt_state.update(running=False, msg=f"Load error: {e}")
            return
        imgs = glob_images(req.img_dir)
        if not imgs:
            conf_opt_state.update(running=False, msg="No images found")
            return
        names = mi.names or {}

        try:
            # Collect all detections at low conf and all GT
            all_preds = []  # (class_id, score, x1, y1, x2, y2, img_idx)
            all_gt = []     # (class_id, x1, y1, x2, y2, img_idx)
            conf_opt_state["total"] = len(imgs)
            conf_opt_state["msg"] = "Running inference..."

            for idx, fp in enumerate(imgs):
                frame = imread(fp)
                if frame is None:
                    conf_opt_state["progress"] = idx + 1
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
                conf_opt_state["progress"] = idx + 1

            # Group by class
            class_ids = set(g[0] for g in all_gt)
            thresholds = np.arange(0.05, 0.951, req.step)
            conf_opt_state["msg"] = "Sweeping thresholds..."
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

            conf_opt_state["results"] = results
            conf_opt_state.update(running=False, msg="Complete")
        except Exception as e:
            conf_opt_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}


@router.get("/api/analysis/conf-optimizer/status")
async def conf_optimizer_status():
    return dict(conf_opt_state)


# ── 4. Embedding Viewer API ────────────────────────────
class EmbeddingViewerRequest(BaseModel):
    model_path: str
    img_dir: str
    method: str = "tsne"  # tsne / umap / pca


@router.post("/api/analysis/embedding-viewer")
async def run_embedding_viewer(req: EmbeddingViewerRequest):
    if embedding_state["running"]:
        return {"error": "Already running"}
    embedding_state.update(running=True, msg="Starting...", image=None)

    def _run():
        import onnxruntime as ort
        try:
            session = ort.InferenceSession(req.model_path)
        except Exception as e:
            embedding_state.update(running=False, msg=f"Load error: {e}")
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
            embedding_state.update(running=False, msg="No images found")
            return

        embedding_state["msg"] = f"Extracting embeddings: 0/{len(files)}"
        for i, (fp, label) in enumerate(files):
            img = imread(str(fp))
            if img is None:
                continue
            img = cv2.resize(img, (w, h))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis], dtype=np.float32) / 255.0
            out = session.run(None, {inp.name: tensor})
            vec = out[0].flatten()
            embeddings.append(vec)
            labels.append(label)
            embedding_state["msg"] = f"Extracting embeddings: {i+1}/{len(files)}"

        if len(embeddings) < 2:
            embedding_state.update(running=False, msg="Need at least 2 images")
            return

        X = np.array(embeddings)
        embedding_state["msg"] = f"Running {req.method.upper()}..."

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
            embedding_state.update(running=False, msg=f"Reduction error: {e}")
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
        embedding_state["_image_path"] = tmp_path
        embedding_state["image"] = None  # 메모리에서 제거
        embedding_state.update(running=False, msg="Complete")

    executor.submit(_run)
    return {"ok": True}


@router.get("/api/analysis/embedding-viewer/status")
async def embedding_viewer_status():
    state = dict(embedding_state)
    # 이미지가 임시 파일에 있으면 로드하여 반환
    img_path = state.pop("_image_path", None)
    if img_path and os.path.isfile(img_path) and state.get("image") is None:
        with open(img_path, "rb") as f:
            state["image"] = base64.b64encode(f.read()).decode()
    return state


# ── 5. Inference Analysis API ──────────────────────────
class InferenceAnalysisRequest(BaseModel):
    model_path: str
    model_type: str = "yolo"
    image_path: str
    conf: float = 0.25


@router.post("/api/analysis/inference-analysis")
async def run_inference_analysis(req: InferenceAnalysisRequest):
    try:
        from core.model_loader import load_model as _load
        from core.inference import run_inference, letterbox, preprocess
        import time as _time

        mi = _load(req.model_path, model_type=req.model_type)
        frame = imread(req.image_path)
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
        det_vis = draw_detections(frame, result, names)

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
            "original_image": encode_jpeg(frame),
            "letterbox_image": encode_jpeg(lb_vis),
            "detection_image": encode_jpeg(det_vis),
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

