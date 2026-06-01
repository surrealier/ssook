"""/api/evaluation/* 라우터."""
import glob
import os
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.model_loader import load_model as _load_model
from core.inference import run_inference, run_classification, run_segmentation, run_embedding
from core.evaluation import (
    evaluate_dataset, evaluate_map50_95, evaluate_classification,
    evaluate_segmentation, evaluate_embedder,
    _compute_iou_matrix, _match_greedy, _yolo_to_xyxy,
)
from server.state import eval_state, all_states, executor
from server.utils import imread, glob_images
from server.path_safety import (
    safe_image_dir,
    safe_label_dir,
    safe_model_file,
    UnsafePathError,
)

router = APIRouter()

_EVAL_HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval_history")


def _build_confusion_matrix(gt_data, pred_data, iou_thres=0.5):
    """Build class confusion matrix from GT and prediction dicts.

    Returns (matrix, classes) where matrix is (n+1)x(n+1) with the last
    index representing background (FP/FN). Rows=GT, cols=predicted.

    Matching is class-agnostic on purpose: a prediction may match a GT of a
    *different* class so the misclassification lands in an off-diagonal cell.
    Uses the shared vectorized IoU + greedy helper (EVAL-07) so we don't
    recompute IoU with a scalar triple-loop alongside evaluate_dataset.
    """
    all_cls: set = set()
    for boxes in list(gt_data.values()) + list(pred_data.values()):
        for b in boxes:
            all_cls.add(b[0])
    classes = sorted(all_cls)
    if not classes:
        return [], []
    n = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}
    bg = n
    size = n + 1
    matrix = [[0] * size for _ in range(size)]

    stems = set(list(gt_data.keys()) + list(pred_data.keys()))
    for stem in stems:
        gts = gt_data.get(stem, [])
        preds = pred_data.get(stem, [])
        if not gts and not preds:
            continue
        gt_boxes = [_yolo_to_xyxy(b[1], b[2], b[3], b[4]) for b in gts]
        gt_cls = [b[0] for b in gts]
        pred_boxes = [_yolo_to_xyxy(b[1], b[2], b[3], b[4]) for b in preds]
        pred_cls = [b[0] for b in preds]
        scores = [b[5] if len(b) > 5 else 0.0 for b in preds]
        order = sorted(range(len(preds)), key=lambda i: -scores[i])

        iou_mat = _compute_iou_matrix(pred_boxes, gt_boxes)  # (P, G)
        gt_matched = np.zeros(len(gts), dtype=bool)
        pred_matched = [False] * len(preds)

        for pi in order:
            best_j = _match_greedy(iou_mat[pi], gt_matched, iou_thres)
            if best_j >= 0:
                pred_matched[pi] = True
                gi = cls_idx.get(gt_cls[best_j], bg)
                pi2 = cls_idx.get(pred_cls[pi], bg)
                matrix[gi][pi2] += 1

        for j in range(len(gts)):
            if not gt_matched[j]:
                matrix[cls_idx.get(gt_cls[j], bg)][bg] += 1
        for pi, matched in enumerate(pred_matched):
            if not matched:
                matrix[bg][cls_idx.get(pred_cls[pi], bg)] += 1

    return matrix, classes


# ── Evaluation async API (#6, #7) ──────────────────────


class EvalAsyncRequest(BaseModel):
    models: list                       # [{path, model_type, class_mapping?}, ...]
    img_dir: str
    label_dir: str = ""
    conf: float = 0.25
    class_mapping: Optional[dict] = None  # {gt_id: name, ...}
    per_model_mappings: Optional[dict] = None  # {model_name: {model_cls_id: gt_cls_id}}
    mapped_only: bool = True
    task: str = "detection"            # detection | classification | segmentation | clip | embedder
    cm_iou: float = 0.5                # confusion-matrix IoU threshold (EVAL-08)
    num_classes: int = 80              # for segmentation
    gt_mask_dir: str = ""              # for segmentation
    text_encoder_path: str = ""        # for clip
    prompts: list = []                 # for clip
    query_dir: str = ""                # for embedder
    top_k: int = 5                     # for embedder


def _safe_model_paths(models: list) -> list:
    """Validate each model entry's path (EVAL-04). Mutates entry dicts in
    place with the resolved path so the worker loads only vetted files."""
    safe_entries: list = []
    for entry in models:
        raw = entry if isinstance(entry, str) else entry.get("path", "")
        resolved = safe_model_file(raw)
        if isinstance(entry, str):
            safe_entries.append(resolved)
        else:
            new_entry = dict(entry)
            new_entry["path"] = resolved
            safe_entries.append(new_entry)
    return safe_entries


@router.post("/api/evaluation/run-async")
async def run_evaluation_async(req: EvalAsyncRequest):
    """비동기 평가 실행 (#6 pbar + #7 모델타입/클래스 지정)"""
    if eval_state["running"]:
        return {"error": "Evaluation already running"}

    # 경로 안전성 검증을 워커 제출 전(boundary)에 수행 — 사용자에게 즉시 명확한 에러 (EVAL-04).
    try:
        req.img_dir = safe_image_dir(req.img_dir)
        if req.label_dir:
            req.label_dir = safe_label_dir(req.label_dir)
        if req.gt_mask_dir:
            req.gt_mask_dir = safe_label_dir(req.gt_mask_dir)
        if req.query_dir:
            req.query_dir = safe_image_dir(req.query_dir)
        req.models = _safe_model_paths(req.models)
    except UnsafePathError as exc:
        return {"error": f"Unsafe path: {exc}"}

    eval_state.update(running=True, progress=0, total=1, msg="Starting...", model_name="", results=[])

    def _auto_save_results():
        """Auto-save eval results on completion."""
        import json, datetime
        if not eval_state["results"]:
            return
        os.makedirs(_EVAL_HISTORY_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(_EVAL_HISTORY_DIR, f"eval_{ts}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"results": eval_state["results"], "msg": eval_state["msg"]}, f, ensure_ascii=False)
        except Exception:
            pass

    def _run():
        import glob
        from core.model_loader import load_model as _load_model
        from core.inference import run_inference
        from core.evaluation import evaluate_dataset, evaluate_map50_95

        try:
            # Task routing
            if req.task == "classification":
                return _run_classification_eval()
            elif req.task == "segmentation":
                return _run_segmentation_eval()
            elif req.task == "clip":
                return _run_clip_eval()
            elif req.task == "embedder":
                return _run_embedder_eval()

            eval_state.update(progress=0, msg="Loading images...")

            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            img_files = []
            for e in exts:
                img_files.extend(glob.glob(os.path.join(req.img_dir, e)))
            img_files.sort()
            if not img_files:
                eval_state.update(running=False, msg="No images found")
                return

            # GT 로드 — 손상된 라벨 라인은 건너뛰고 카운트만 (EVAL-05).
            gt_data = {}
            gt_skipped = 0
            for fp in img_files:
                stem = os.path.splitext(os.path.basename(fp))[0]
                txt = os.path.join(req.label_dir, stem + ".txt")
                boxes = []
                if os.path.isfile(txt):
                    with open(txt) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            try:
                                boxes.append((int(parts[0]), *map(float, parts[1:5])))
                            except ValueError:
                                # 헤더 라인·비숫자 토큰 — 한 줄 오류로 전체 run을 중단하지 않음.
                                gt_skipped += 1
                gt_data[stem] = boxes

            total_work = len(img_files)
            eval_state["total"] = total_work
            done = 0

            # 모델을 먼저 로드하고, 이미지를 1회만 디코드해 모든 모델에 적용 (loop inversion, EVAL-06).
            # 이전 고정-500 캐시는 500장 초과 시 모델마다 재디코드 + 멀티-GB 점유 문제가 있었음.
            loaded = []  # [(name, mi, mapping, mapped_only)]
            for entry in req.models:
                model_path = entry if isinstance(entry, str) else entry.get("path", "")
                model_type = "yolo" if isinstance(entry, str) else entry.get("model_type", "yolo")
                name = os.path.basename(model_path)
                try:
                    mi = _load_model(model_path, model_type=model_type)
                    mi._frame_buffer = []
                except Exception as exc:
                    eval_state["results"].append({"name": name, "error": str(exc)})
                    continue
                pm = (req.per_model_mappings or {}).get(name, {})
                mapping = {int(k): int(v) for k, v in pm.items()} if pm else {}
                loaded.append((name, mi, mapping, req.mapped_only))

            if not loaded:
                eval_state.update(running=False, msg="No models loaded")
                return

            eval_state["model_name"] = ", ".join(name for name, _, _, _ in loaded)
            pred_per_model = {name: {} for name, _, _, _ in loaded}

            for fp in img_files:
                # 협조적 취소: Stop 후 running=False면 즉시 종료 (EVAL-02).
                if not eval_state["running"]:
                    eval_state.update(running=False, msg="Stopped")
                    return
                frame = imread(fp)
                done += 1
                eval_state["progress"] = done
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                stem = os.path.splitext(os.path.basename(fp))[0]
                for name, mi, mapping, mapped_only in loaded:
                    res = run_inference(mi, frame, req.conf)
                    boxes = []
                    for box, score, cid in zip(res.boxes, res.scores, res.class_ids):
                        cid = int(cid)
                        if mapping:
                            if cid in mapping:
                                cid = mapping[cid]
                            elif mapped_only:
                                continue
                        x1, y1, x2, y2 = box
                        cx = ((x1+x2)/2)/w; cy = ((y1+y2)/2)/h
                        bw = (x2-x1)/w; bh = (y2-y1)/h
                        boxes.append((cid, cx, cy, bw, bh, float(score)))
                    pred_per_model[name][stem] = boxes
                eval_state["msg"] = f"{done}/{total_work}" + (f" ({gt_skipped} GT lines skipped)" if gt_skipped else "")

            for name, _, mapping, mapped_only in loaded:
                # GT 필터링 (매핑된 클래스만)
                if mapping and mapped_only:
                    allowed_gt = set(mapping.values())
                    gt_eval = {s: [b for b in boxes if b[0] in allowed_gt]
                               for s, boxes in gt_data.items()}
                else:
                    gt_eval = gt_data
                pred_data = pred_per_model[name]

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
                conf_matrix, conf_classes = _build_confusion_matrix(gt_eval, pred_data, req.cm_iou)
                eval_state["results"].append({
                    "name": name,
                    "map50": round(ov.get("ap", 0) * 100, 4),
                    "map5095": round(map5095 * 100, 4),
                    "precision": round(ov.get("precision", 0) * 100, 4),
                    "recall": round(ov.get("recall", 0) * 100, 4),
                    "f1": round(ov.get("f1", 0) * 100, 4),
                    "detail": detail,
                    "confusion_matrix": conf_matrix,
                    "confusion_classes": conf_classes,
                })

            complete_msg = "Complete" + (f" ({gt_skipped} GT lines skipped)" if gt_skipped else "")
            eval_state.update(running=False, msg=complete_msg)
            _auto_save_results()
        except Exception as e:
            eval_state.update(running=False, msg=f"Error: {e}")

    def _run_classification_eval():
        import glob
        from core.model_loader import load_model as _load_model
        from core.inference import run_classification, ClassificationResult
        from core.evaluation import evaluate_classification
        try:
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            img_files = []
            for e in exts:
                img_files.extend(glob.glob(os.path.join(req.img_dir, e)))
            img_files.sort()
            if not img_files:
                eval_state.update(running=False, msg="No images found"); return

            # GT: label_dir에 stem.txt (한 줄에 class_id)
            gt_data = {}
            for fp in img_files:
                stem = os.path.splitext(os.path.basename(fp))[0]
                txt = os.path.join(req.label_dir, stem + ".txt")
                if os.path.isfile(txt):
                    with open(txt) as f:
                        line = f.readline().strip()
                        if line.isdigit():
                            gt_data[stem] = int(line)

            total_work = len(img_files) * len(req.models)
            eval_state["total"] = total_work
            done = 0

            for entry in req.models:
                model_path = entry if isinstance(entry, str) else entry.get("path", "")
                model_type = "yolo" if isinstance(entry, str) else entry.get("model_type", "yolo")
                name = os.path.basename(model_path)
                eval_state["model_name"] = name
                try:
                    mi = _load_model(model_path, model_type=model_type)
                except Exception as exc:
                    eval_state["results"].append({"name": name, "error": str(exc)})
                    done += len(img_files); eval_state["progress"] = done; continue

                pred_data = {}
                for fp in img_files:
                    if not eval_state["running"]:
                        eval_state.update(running=False, msg="Stopped"); return
                    frame = imread(fp)
                    if frame is None: done += 1; continue
                    stem = os.path.splitext(os.path.basename(fp))[0]
                    cls_res = run_classification(mi, frame)
                    if cls_res and cls_res.class_id is not None:
                        pred_data[stem] = (cls_res.class_id, cls_res.confidence)
                    done += 1; eval_state.update(progress=done, msg=f"{name}: {done}/{total_work}")

                metrics = evaluate_classification(gt_data, pred_data)
                ov = metrics.get("__overall__", {})
                detail = {str(k): v for k, v in metrics.items() if k != "__overall__"}
                eval_state["results"].append({
                    "name": name, "task": "classification",
                    "accuracy": round(ov.get("accuracy", 0) * 100, 4),
                    "precision": round(ov.get("precision", 0) * 100, 4),
                    "recall": round(ov.get("recall", 0) * 100, 4),
                    "f1": round(ov.get("f1", 0) * 100, 4),
                    "detail": detail,
                })
            eval_state.update(running=False, msg="Complete")
        except Exception as e:
            eval_state.update(running=False, msg=f"Error: {e}")

    def _run_segmentation_eval():
        import glob
        from core.model_loader import load_model as _load_model
        from core.evaluation import evaluate_segmentation
        try:
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            img_files = []
            for e in exts:
                img_files.extend(glob.glob(os.path.join(req.img_dir, e)))
            img_files.sort()
            if not img_files:
                eval_state.update(running=False, msg="No images found"); return

            total_work = len(img_files) * len(req.models)
            eval_state["total"] = total_work
            done = 0
            nc = req.num_classes

            for entry in req.models:
                model_path = entry if isinstance(entry, str) else entry.get("path", "")
                model_type = "yolo" if isinstance(entry, str) else entry.get("model_type", "yolo")
                name = os.path.basename(model_path)
                eval_state["model_name"] = name
                try:
                    mi = _load_model(model_path, model_type=model_type)
                except Exception as exc:
                    eval_state["results"].append({"name": name, "error": str(exc)})
                    done += len(img_files); eval_state["progress"] = done; continue

                from core.inference import run_segmentation
                agg_pred, agg_gt = [], []
                mask_skipped = 0
                mask_dir = req.gt_mask_dir or req.label_dir
                for fp in img_files:
                    if not eval_state["running"]:
                        eval_state.update(running=False, msg="Stopped"); return
                    frame = imread(fp)
                    if frame is None: done += 1; continue
                    stem = os.path.splitext(os.path.basename(fp))[0]
                    # GT mask: 확장자 후보 + 원본 이미지 ext 중 첫 존재 선택 (SPEC-08).
                    src_ext = os.path.splitext(fp)[1]
                    gt_path = None
                    for cand_ext in (".png", ".bmp", ".tif", ".tiff", ".jpg", src_ext):
                        p = os.path.join(mask_dir, stem + cand_ext)
                        if os.path.isfile(p):
                            gt_path = p
                            break
                    if gt_path is None:
                        mask_skipped += 1; done += 1; continue
                    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    if gt_mask is None:
                        mask_skipped += 1; done += 1; continue
                    # Predict
                    seg_res = run_segmentation(mi, frame)
                    if seg_res and seg_res.mask is not None:
                        pred_mask = seg_res.mask
                        if pred_mask.shape != gt_mask.shape:
                            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                        agg_pred.append(pred_mask)
                        agg_gt.append(gt_mask)
                    done += 1
                    skip_note = f" ({mask_skipped} masks skipped)" if mask_skipped else ""
                    eval_state.update(progress=done, msg=f"{name}: {done}/{total_work}{skip_note}")

                if agg_pred:
                    # Per-image evaluation, then average
                    all_ious = {}  # class_id -> [iou, ...]
                    all_dices = {}
                    for pm, gm in zip(agg_pred, agg_gt):
                        m = evaluate_segmentation(pm, gm, nc)
                        for k, v in m.items():
                            if k == "__overall__":
                                continue
                            all_ious.setdefault(k, []).append(v["iou"])
                            all_dices.setdefault(k, []).append(v["dice"])
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
                    eval_state["results"].append({
                        "name": name, "task": "segmentation",
                        "mIoU": round(miou * 100, 4),
                        "mDice": round(mdice * 100, 4),
                        "detail": detail,
                        "num_images": len(agg_pred),
                        "masks_skipped": mask_skipped,
                    })
                else:
                    eval_state["results"].append({"name": name, "task": "segmentation",
                                                  "error": "No valid predictions",
                                                  "masks_skipped": mask_skipped})
            eval_state.update(running=False, msg="Complete")
        except Exception as e:
            eval_state.update(running=False, msg=f"Error: {e}")

    def _run_clip_eval():
        import glob
        from core.clip_inference import CLIPModel
        try:
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            img_files = []
            for e in exts:
                img_files.extend(glob.glob(os.path.join(req.img_dir, e)))
            img_files.sort()
            if not img_files:
                eval_state.update(running=False, msg="No images found"); return
            prompts = req.prompts or []
            if not prompts:
                eval_state.update(running=False, msg="Error: no prompts provided"); return
            gt_data = {}
            for fp in img_files:
                stem = os.path.splitext(os.path.basename(fp))[0]
                txt = os.path.join(req.label_dir, stem + ".txt")
                if os.path.isfile(txt):
                    with open(txt) as f:
                        line = f.readline().strip()
                        if line.isdigit(): gt_data[stem] = int(line)
            total_work = len(img_files) * len(req.models)
            eval_state.update(total=total_work)
            done = 0
            for entry in req.models:
                model_path = entry if isinstance(entry, str) else entry.get("path", "")
                name = os.path.basename(model_path)
                eval_state["model_name"] = name
                try:
                    clip = CLIPModel(model_path, req.text_encoder_path)
                except Exception as exc:
                    eval_state["results"].append({"name": name, "error": str(exc)})
                    done += len(img_files); eval_state["progress"] = done; continue
                top1 = 0; top5 = 0; total = 0
                for fp in img_files:
                    if not eval_state["running"]:
                        eval_state.update(running=False, msg="Stopped"); return
                    stem = os.path.splitext(os.path.basename(fp))[0]
                    if stem not in gt_data: done += 1; continue
                    frame = imread(fp)
                    if frame is None: done += 1; continue
                    scores = clip.zero_shot_classify(frame, prompts)
                    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
                    gt_cls = gt_data[stem]
                    if ranked[0] == gt_cls: top1 += 1
                    if gt_cls in ranked[:5]: top5 += 1
                    total += 1; done += 1
                    eval_state.update(progress=done, msg=f"{name}: {done}/{total_work}")
                eval_state["results"].append({
                    "name": name, "task": "clip",
                    "top1_acc": round(top1 / max(total, 1) * 100, 4),
                    "top5_acc": round(top5 / max(total, 1) * 100, 4),
                })
            eval_state.update(running=False, msg="Complete")
        except Exception as e:
            eval_state.update(running=False, msg=f"Error: {e}")

    def _run_embedder_eval():
        import glob
        from core.model_loader import load_model as _load_model
        from core.inference import run_embedding
        try:
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            gallery_files, query_files = [], []
            for e in exts:
                gallery_files.extend(glob.glob(os.path.join(req.img_dir, e)))
            gallery_files.sort()
            q_dir = req.query_dir or req.img_dir
            for e in exts:
                query_files.extend(glob.glob(os.path.join(q_dir, e)))
            query_files.sort()
            if not gallery_files or not query_files:
                eval_state.update(running=False, msg="No images found"); return
            def _labels(files):
                out = {}
                for fp in files:
                    stem = os.path.splitext(os.path.basename(fp))[0]
                    txt = os.path.join(req.label_dir, stem + ".txt")
                    if os.path.isfile(txt):
                        with open(txt) as f:
                            line = f.readline().strip()
                            if line.isdigit(): out[stem] = int(line)
                return out
            gt_g = _labels(gallery_files); gt_q = _labels(query_files)
            total_work = (len(gallery_files) + len(query_files)) * len(req.models)
            eval_state.update(total=total_work); done = 0
            for entry in req.models:
                model_path = entry if isinstance(entry, str) else entry.get("path", "")
                model_type = "yolo" if isinstance(entry, str) else entry.get("model_type", "yolo")
                name = os.path.basename(model_path)
                eval_state["model_name"] = name
                try:
                    mi = _load_model(model_path, model_type=model_type)
                except Exception as exc:
                    eval_state["results"].append({"name": name, "error": str(exc)})
                    done += len(gallery_files) + len(query_files); eval_state["progress"] = done; continue
                g_embs, g_labels, q_embs, q_labels = [], [], [], []
                for fp in gallery_files:
                    if not eval_state["running"]:
                        eval_state.update(running=False, msg="Stopped"); return
                    stem = os.path.splitext(os.path.basename(fp))[0]
                    frame = imread(fp)
                    if frame is not None:
                        res = run_embedding(mi, frame)
                        if res and res.embedding is not None:
                            g_embs.append(res.embedding); g_labels.append(gt_g.get(stem, -1))
                    done += 1; eval_state.update(progress=done, msg=f"{name}: gallery")
                for fp in query_files:
                    if not eval_state["running"]:
                        eval_state.update(running=False, msg="Stopped"); return
                    stem = os.path.splitext(os.path.basename(fp))[0]
                    frame = imread(fp)
                    if frame is not None:
                        res = run_embedding(mi, frame)
                        if res and res.embedding is not None:
                            q_embs.append(res.embedding); q_labels.append(gt_q.get(stem, -1))
                    done += 1; eval_state.update(progress=done, msg=f"{name}: query")
                if q_embs and g_embs:
                    from core.evaluation import evaluate_embedder
                    m = evaluate_embedder(q_embs, g_embs, q_labels, g_labels, req.top_k)
                    eval_state["results"].append({
                        "name": name, "task": "embedder",
                        "retrieval_at_1": round(m.get("retrieval_at_1", 0) * 100, 4),
                        "retrieval_at_k": round(m.get("retrieval_at_k", 0) * 100, 4),
                        "mean_cosine_sim": round(m.get("mean_cosine_sim", 0), 4),
                    })
                else:
                    eval_state["results"].append({"name": name, "task": "embedder", "error": "No valid embeddings"})
            eval_state.update(running=False, msg="Complete")
        except Exception as e:
            eval_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}


@router.get("/api/evaluation/status")
async def evaluation_status():
    return {
        "running": eval_state["running"],
        "progress": eval_state["progress"],
        "total": eval_state["total"],
        "msg": eval_state["msg"],
        "model_name": eval_state["model_name"],
        "results": eval_state["results"],
    }

@router.get("/api/eval/stop")
async def evaluation_stop():
    all_states["eval"]["running"] = False
    return {"ok": True}


@router.get("/api/evaluation/export-csv")
async def evaluation_export_csv():
    """평가 결과를 CSV로 다운로드 (정리된 포맷)"""
    import io, csv
    if not eval_state["results"]:
        return {"error": "No results"}
    buf = io.StringIO()
    buf.write('\ufeff')  # BOM for Excel 한글 호환
    w = csv.writer(buf)
    # ── Summary ──
    w.writerow(["=== Evaluation Summary ==="])
    w.writerow(["Model", "mAP@50 (%)", "mAP@50:95 (%)", "Precision (%)", "Recall (%)", "F1 (%)"])
    for r in eval_state["results"]:
        w.writerow([r.get("name",""), r.get("map50",""), r.get("map5095",""),
                     r.get("precision",""), r.get("recall",""), r.get("f1","")])
    w.writerow([])
    # ── Per-class Detail ──
    for r in eval_state["results"]:
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


# ── Evaluation API ──────────────────────────────────────
class EvalRequest(BaseModel):
    models: list[str]
    img_dir: str
    label_dir: str
    conf: float = 0.25


@router.post("/api/evaluation/run")
async def run_evaluation(req: EvalRequest):
    """Run multi-model evaluation against GT labels."""
    try:
        from core.model_loader import load_model as _load_model
        from core.inference import run_inference
        from core.evaluation import evaluate_dataset, evaluate_map50_95
        import glob, cv2

        # 경로 안전성 검증 (EVAL-04).
        try:
            img_dir = safe_image_dir(req.img_dir)
            label_dir = safe_label_dir(req.label_dir)
            model_paths = [safe_model_file(m) for m in req.models]
        except UnsafePathError as exc:
            return {"error": f"Unsafe path: {exc}"}

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        img_files = []
        for e in exts:
            img_files.extend(glob.glob(os.path.join(img_dir, e)))
        img_files.sort()

        # Load GT — 손상된 라벨 라인은 건너뜀 (EVAL-05).
        gt_data = {}
        for fp in img_files:
            stem = os.path.splitext(os.path.basename(fp))[0]
            txt = os.path.join(label_dir, stem + ".txt")
            boxes = []
            if os.path.isfile(txt):
                with open(txt) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        try:
                            boxes.append((int(parts[0]), *map(float, parts[1:5])))
                        except ValueError:
                            continue
            gt_data[stem] = boxes

        results = []
        for model_path in model_paths:
            name = os.path.basename(model_path)
            try:
                mi = _load_model(model_path)
                pred_data = {}
                mi._frame_buffer = []
                for fp in img_files:
                    frame = imread(fp)
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



# ── Eval result persistence ─────────────────────────────


@router.get("/api/eval/history")
async def eval_history():
    """List saved evaluation results."""
    if not os.path.isdir(_EVAL_HISTORY_DIR):
        return {"files": []}
    files = sorted([f for f in os.listdir(_EVAL_HISTORY_DIR) if f.endswith(".json")], reverse=True)
    return {"files": files}


@router.get("/api/eval/load/{filename}")
async def eval_load(filename: str):
    """Load a saved evaluation result.

    Path-param is untrusted (EVAL-04): reject any separator/`..`, force a
    `.json` extension, and confirm the realpath stays inside the history dir
    so a crafted name can't read arbitrary files (Windows backslash escapes).
    """
    import json
    # A bare filename never contains a path separator or parent ref.
    if os.path.basename(filename) != filename or ".." in filename:
        return {"error": "Invalid filename"}
    if not filename.endswith(".json"):
        return {"error": "Invalid filename"}
    history_root = os.path.realpath(_EVAL_HISTORY_DIR)
    path = os.path.realpath(os.path.join(history_root, filename))
    if os.path.commonpath([history_root, path]) != history_root:
        return {"error": "Invalid filename"}
    if not os.path.isfile(path):
        return {"error": "File not found"}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@router.post("/api/eval/save")
async def eval_save():
    """Save current evaluation results to file."""
    import json, datetime
    if not eval_state["results"]:
        return {"error": "No results to save"}
    os.makedirs(_EVAL_HISTORY_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{ts}.json"
    path = os.path.join(_EVAL_HISTORY_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"results": eval_state["results"], "msg": eval_state["msg"]}, f, ensure_ascii=False, indent=2)
    return {"ok": True, "filename": filename}
