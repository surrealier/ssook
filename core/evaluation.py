"""Evaluation metrics — extracted from ui/evaluation_tab.py for headless use."""
import numpy as np


def _compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def _compute_iou_matrix(pred_boxes, gt_boxes):
    """벡터화된 IoU 행렬 계산: (M, 4) x (N, 4) → (M, N)"""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    pred = np.asarray(pred_boxes, dtype=np.float32)
    gt = np.asarray(gt_boxes, dtype=np.float32)
    ix1 = np.maximum(pred[:, 0:1], gt[:, 0:1].T)
    iy1 = np.maximum(pred[:, 1:2], gt[:, 1:2].T)
    ix2 = np.minimum(pred[:, 2:3], gt[:, 2:3].T)
    iy2 = np.minimum(pred[:, 3:4], gt[:, 3:4].T)
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    a1 = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    a2 = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = a1[:, None] + a2[None, :] - inter
    return inter / (union + 1e-9)


def _yolo_to_xyxy(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _compute_ap(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # 벡터화된 101-point interpolation
    thresholds = np.linspace(0, 1, 101)
    indices = np.searchsorted(mrec, thresholds, side='left')
    indices = np.clip(indices, 0, len(mpre) - 1)
    return float(np.mean(mpre[indices]))


def evaluate_dataset(gt_data, pred_data, iou_thres=0.5):
    all_classes = set()
    for boxes in list(gt_data.values()) + list(pred_data.values()):
        for b in boxes:
            all_classes.add(b[0])
    if not all_classes:
        return {}

    results = {}
    total_tp = total_fp = total_fn = 0

    for cid in sorted(all_classes):
        preds = []
        gt_per_img = {}
        for stem in set(list(gt_data.keys()) + list(pred_data.keys())):
            gt_boxes = [_yolo_to_xyxy(b[1], b[2], b[3], b[4]) for b in gt_data.get(stem, []) if b[0] == cid]
            gt_per_img[stem] = {"boxes": gt_boxes, "matched": [False] * len(gt_boxes)}
            for b in pred_data.get(stem, []):
                if b[0] == cid:
                    score = b[5] if len(b) > 5 else 0.0
                    preds.append((score, stem, _yolo_to_xyxy(b[1], b[2], b[3], b[4])))
        preds.sort(key=lambda x: -x[0])

        n_gt = sum(len(v["boxes"]) for v in gt_per_img.values())
        if n_gt == 0 and len(preds) == 0:
            continue

        tp_list = []
        for _score, stem, pbox in preds:
            best_iou = 0
            best_j = -1
            gt_entry = gt_per_img[stem]
            gt_boxes_list = gt_entry["boxes"]
            if gt_boxes_list:
                iou_row = _compute_iou_matrix([pbox], gt_boxes_list)[0]
                for j in range(len(gt_boxes_list)):
                    if gt_entry["matched"][j]:
                        iou_row[j] = 0.0
                best_j = int(np.argmax(iou_row))
                best_iou = float(iou_row[best_j])
            if best_iou >= iou_thres and best_j >= 0 and not gt_entry["matched"][best_j]:
                gt_entry["matched"][best_j] = True
                tp_list.append(1)
            else:
                tp_list.append(0)

        tp_arr = np.array(tp_list)
        tp_cum = np.cumsum(tp_arr)
        fp_cum = np.cumsum(1 - tp_arr)
        recalls = tp_cum / max(n_gt, 1)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

        ap = _compute_ap(recalls, precisions) if len(recalls) > 0 else 0.0
        tp_total = int(tp_arr.sum())
        fp_total = len(preds) - tp_total
        fn_total = n_gt - tp_total
        total_tp += tp_total
        total_fp += fp_total
        total_fn += fn_total

        prec = tp_total / (tp_total + fp_total + 1e-9)
        rec = tp_total / (n_gt + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        results[cid] = {"ap": ap, "precision": prec, "recall": rec, "f1": f1}

    prec_all = total_tp / (total_tp + total_fp + 1e-9)
    rec_all = total_tp / (total_tp + total_fn + 1e-9)
    f1_all = 2 * prec_all * rec_all / (prec_all + rec_all + 1e-9)
    mean_ap = float(np.mean([v["ap"] for v in results.values()])) if results else 0.0
    results["__overall__"] = {"ap": mean_ap, "precision": prec_all, "recall": rec_all, "f1": f1_all}
    return results


def evaluate_map50_95(gt_data, pred_data):
    """mAP@50:95를 한 번의 IoU 계산으로 처리 (10회 반복 호출 제거)"""
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    all_classes = set()
    for boxes in list(gt_data.values()) + list(pred_data.values()):
        for b in boxes:
            all_classes.add(b[0])
    if not all_classes:
        return 0.0

    # threshold별 per-class AP 수집
    ap_per_thresh = {t: [] for t in iou_thresholds}

    for cid in sorted(all_classes):
        preds = []
        gt_per_img = {}
        for stem in set(list(gt_data.keys()) + list(pred_data.keys())):
            gt_boxes = [_yolo_to_xyxy(b[1], b[2], b[3], b[4])
                        for b in gt_data.get(stem, []) if b[0] == cid]
            gt_per_img[stem] = gt_boxes
            for b in pred_data.get(stem, []):
                if b[0] == cid:
                    score = b[5] if len(b) > 5 else 0.0
                    preds.append((score, stem, _yolo_to_xyxy(b[1], b[2], b[3], b[4])))
        preds.sort(key=lambda x: -x[0])

        n_gt = sum(len(v) for v in gt_per_img.values())
        if n_gt == 0 and len(preds) == 0:
            continue

        # 각 pred에 대해 best IoU를 한 번만 계산
        pred_ious = []  # (best_iou, best_gt_idx_in_stem, stem)
        for _score, stem, pbox in preds:
            gt_boxes_list = gt_per_img[stem]
            if gt_boxes_list:
                iou_row = _compute_iou_matrix([pbox], gt_boxes_list)[0]
                best_j = int(np.argmax(iou_row))
                best_iou = float(iou_row[best_j])
            else:
                best_j = -1
                best_iou = 0.0
            pred_ious.append((best_iou, best_j, stem))

        # 각 threshold에 대해 매칭 (greedy, IoU 내림차순 이미 정렬됨)
        for iou_t in iou_thresholds:
            matched = {stem: set() for stem in gt_per_img}
            tp_list = []
            for (best_iou, best_j, stem) in pred_ious:
                if best_iou >= iou_t and best_j >= 0 and best_j not in matched[stem]:
                    matched[stem].add(best_j)
                    tp_list.append(1)
                else:
                    tp_list.append(0)

            if not tp_list:
                ap_per_thresh[iou_t].append(0.0)
                continue

            tp_arr = np.array(tp_list)
            tp_cum = np.cumsum(tp_arr)
            fp_cum = np.cumsum(1 - tp_arr)
            recalls = tp_cum / max(n_gt, 1)
            precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
            ap_per_thresh[iou_t].append(
                _compute_ap(recalls, precisions) if len(recalls) > 0 else 0.0)

    # 각 threshold의 mAP 평균
    thresh_maps = []
    for iou_t in iou_thresholds:
        aps = ap_per_thresh[iou_t]
        thresh_maps.append(float(np.mean(aps)) if aps else 0.0)
    return float(np.mean(thresh_maps)) if thresh_maps else 0.0
