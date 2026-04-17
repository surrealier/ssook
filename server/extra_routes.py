"""/api/clip/*, /api/embedder/*, /api/segmentation/*, /api/batch/*, /api/quantize/*, /api/inspector/*, /api/profiler/*, /api/tracking/*, /api/hf/*, /api/force-stop/* 라우터."""
import os
import time

import cv2
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from core.model_loader import load_model as _load_model
from core.inference import run_inference, letterbox, preprocess, _padded_to_tensor, run_segmentation
from server.state import clip_state, embedder_state, seg_state, quant_state, all_states, executor
from server.utils import imread, encode_jpeg, glob_images, overlay_segmentation

router = APIRouter()

# ── CLIP Zero-Shot API ─────────────────────────────────
class CLIPRequest(BaseModel):
    image_encoder: str
    text_encoder: str
    img_dir: str
    labels: str  # comma-separated

clip_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": []}
all_states["clip"] = clip_state

@router.post("/api/clip/run")
async def run_clip(req: CLIPRequest):
    if clip_state["running"]:
        return {"error": "Already running"}
    clip_state.update(running=True, progress=0, total=0, msg="Starting...", results=[])

    def _run():
        try:
            from core.clip_inference import CLIPModel, simple_tokenize
            model = CLIPModel(req.image_encoder, req.text_encoder)
            labels = [l.strip() for l in req.labels.split(",") if l.strip()]
            if not labels:
                clip_state.update(running=False, msg="No labels provided")
                return
            # Pre-encode text
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
            # Per-label correct count
            label_correct = {l: 0 for l in labels}
            label_total = {l: 0 for l in labels}
            detail_log = []  # per-image detail
            for idx, fp in enumerate(imgs):
                frame = imread(fp)
                if frame is None:
                    clip_state["progress"] = idx + 1
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
                clip_state["progress"] = idx + 1
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
            clip_state["results"] = results
            clip_state["detail"] = detail_log[:500]  # 최대 500개
            clip_state.update(running=False, msg="Complete")
        except Exception as e:
            clip_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/clip/status")
async def clip_status():
    return dict(clip_state)


# ── Embedder Evaluation API ────────────────────────────
class EmbedderRequest(BaseModel):
    model_path: str
    model_type: str = "yolo"
    img_dir: str
    top_k: int = 5

embedder_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": [], "detail": []}
all_states["embedder"] = embedder_state

@router.post("/api/embedder/run")
async def run_embedder(req: EmbedderRequest):
    if embedder_state["running"]:
        return {"error": "Already running"}
    embedder_state.update(running=True, progress=0, total=0, msg="Starting...", results=[], detail=[])

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
            import traceback
            traceback.print_exc()
            embedder_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}

@router.get("/api/embedder/status")
async def embedder_status():
    return dict(embedder_state)

@router.post("/api/embedder/compare")
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


# ── Batch: Augmentation Preview API ────────────────────
@router.post("/api/batch/augmentation")
async def batch_augmentation(req: dict):
    try:
        img_dir = req.get("img_dir", "")
        aug_type = req.get("aug_type", "Flip")
        imgs = glob_images(img_dir)
        if not imgs:
            return {"error": "No images found"}
        import random
        fp = random.choice(imgs)
        frame = imread(fp)
        if frame is None:
            return {"error": "Cannot read image"}
        original = encode_jpeg(frame)
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
    except Exception as e:
        return {"error": str(e)}


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

quant_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": {}}
all_states["quantize"] = quant_state

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
    return dict(quant_state)


# ── Phase 1: Model Inspector API ───────────────────────
@router.post("/api/inspector/inspect")
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
@router.post("/api/profiler/run")
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
@router.post("/api/infer/pose")
async def api_infer_pose(req: dict):
    try:
        model_path = req.get("model_path", "")
        image_path = req.get("image_path", "")
        conf = req.get("conf", 0.25)
        model_type = req.get("model_type", "pose_yolo")
        if not model_path or not image_path:
            return {"error": "model_path and image_path required"}
        frame = imread(image_path)
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
    except Exception as e:
        return {"error": str(e)}


# ── Phase 1: Instance Segmentation API ─────────────────
@router.post("/api/infer/instance-seg")
async def api_infer_instance_seg(req: dict):
    try:
        model_path = req.get("model_path", "")
        image_path = req.get("image_path", "")
        conf = req.get("conf", 0.25)
        model_type = req.get("model_type", "instseg_yolo")
        if not model_path or not image_path:
            return {"error": "model_path and image_path required"}
        frame = imread(image_path)
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
    except Exception as e:
        return {"error": str(e)}


# ── Phase 1: Tracking API ──────────────────────────────
_trackers = {}

@router.post("/api/tracking/create")
async def api_tracking_create(req: dict):
    try:
        tracker_type = req.get("tracker_type", "bytetrack")
        from core.tracking import create_tracker
        tid = str(uuid.uuid4())[:8]
        _trackers[tid] = create_tracker(tracker_type)
        return {"tracker_id": tid, "type": tracker_type}
    except Exception as e:
        return {"error": str(e)}


@router.post("/api/tracking/reset")
async def api_tracking_reset(req: dict):
    tid = req.get("tracker_id", "")
    if tid in _trackers:
        _trackers[tid].reset()
        return {"status": "ok"}
    return {"error": "Tracker not found"}


# ── HuggingFace Hub API ────────────────────────────────

@router.post("/api/hf/search")
async def hf_search(req: dict):
    try:
        from core.hf_downloader import search_models
        results = search_models(req.get("query", ""), req.get("task", ""), req.get("limit", 20))
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}


@router.post("/api/hf/files")
async def hf_files(req: dict):
    try:
        from core.hf_downloader import list_onnx_files
        files = list_onnx_files(req.get("repo_id", ""))
        return {"files": files}
    except Exception as e:
        return {"error": str(e)}


@router.post("/api/hf/download")
async def hf_download(req: dict):
    try:
        from core.hf_downloader import download_model
        path = await asyncio.get_event_loop().run_inexecutor(
            executor, download_model, req["repo_id"], req["filename"]
        )
        return {"path": path}
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/hf/cached")
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
@router.post("/api/force-stop/{task_id}")
async def force_stop(task_id: str):
    """비동기 작업 강제 중지 — running 플래그를 False로 리셋"""
    if task_id == "all":
        for s in all_states.values():
            s["running"] = False
        return {"ok": True, "msg": "All tasks stopped"}
    state = all_states.get(task_id)
    if not state:
        return {"error": f"Unknown task: {task_id}"}
    state["running"] = False
    state["msg"] = "Stopped by user"
    return {"ok": True}


