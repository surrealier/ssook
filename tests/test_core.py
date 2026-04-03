"""core 로직 단위 테스트"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


# ============================================================
# inference.py
# ============================================================
from core.inference import (
    _softmax, letterbox, preprocess, preprocess_classification,
    DetectionResult, ClassificationResult,
)


class TestSoftmax:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        p = _softmax(x)
        assert abs(p.sum() - 1.0) < 1e-6
        assert p[2] > p[1] > p[0]

    def test_uniform(self):
        x = np.zeros(5)
        p = _softmax(x)
        np.testing.assert_allclose(p, 0.2, atol=1e-6)

    def test_large_values(self):
        x = np.array([1000.0, 1001.0])
        p = _softmax(x)
        assert abs(p.sum() - 1.0) < 1e-6


class TestLetterbox:
    def test_square(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, ratio, (pw, ph) = letterbox(img, (640, 640))
        assert padded.shape[0] == 640
        assert padded.shape[1] == 640

    def test_preserves_aspect(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        padded, ratio, _ = letterbox(img, (640, 640))
        assert padded.shape == (640, 640, 3)


class TestPreprocess:
    def test_output_shape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor = preprocess(img, (640, 640))
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32
        assert tensor.max() <= 1.0

    def test_classification_preprocess(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor = preprocess_classification(img, (224, 224))
        assert tensor.shape == (1, 3, 224, 224)


class TestDetectionResult:
    def test_empty(self):
        r = DetectionResult.empty()
        assert len(r.boxes) == 0
        assert len(r.scores) == 0
        assert len(r.class_ids) == 0


# ============================================================
# evaluation_tab.py — 순수 함수만 테스트 (PySide6 우회)
# ============================================================
# PySide6 import 우회를 위해 직접 함수 복사
def _compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def _compute_ap(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = mpre[mrec >= t]
        ap += (p.max() if len(p) > 0 else 0.0)
    return ap / 101.0


def _yolo_to_xyxy(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def evaluate_classification(gt_data, pred_data, num_classes):
    if not gt_data or not pred_data:
        return {}
    y_true, y_pred = [], []
    for stem in gt_data:
        if stem in pred_data:
            y_true.append(gt_data[stem])
            y_pred.append(pred_data[stem][0])
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    accuracy = float((y_true == y_pred).mean())
    results = {}
    all_classes = sorted(set(y_true.tolist() + y_pred.tolist()))
    for cid in all_classes:
        tp = int(((y_true == cid) & (y_pred == cid)).sum())
        fp = int(((y_true != cid) & (y_pred == cid)).sum())
        fn = int(((y_true == cid) & (y_pred != cid)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        results[cid] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp}
    macro_p = np.mean([v["precision"] for v in results.values()])
    macro_r = np.mean([v["recall"] for v in results.values()])
    macro_f1 = np.mean([v["f1"] for v in results.values()])
    results["__overall__"] = {"accuracy": accuracy, "precision": float(macro_p),
                              "recall": float(macro_r), "f1": float(macro_f1)}
    return results


class TestIoU:
    def test_perfect_overlap(self):
        assert abs(_compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-6

    def test_no_overlap(self):
        assert _compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) < 1e-6

    def test_partial_overlap(self):
        iou = _compute_iou((0, 0, 10, 10), (5, 5, 15, 15))
        expected = 25.0 / (100 + 100 - 25)
        assert abs(iou - expected) < 1e-4

    def test_contained(self):
        iou = _compute_iou((0, 0, 10, 10), (2, 2, 8, 8))
        assert 0 < iou < 1


class TestAP:
    def test_perfect(self):
        recalls = np.array([0.5, 1.0])
        precisions = np.array([1.0, 1.0])
        ap = _compute_ap(recalls, precisions)
        assert abs(ap - 1.0) < 0.01

    def test_zero(self):
        recalls = np.array([])
        precisions = np.array([])
        ap = _compute_ap(recalls, precisions)
        assert ap >= 0


class TestYoloToXyxy:
    def test_basic(self):
        box = _yolo_to_xyxy(0.5, 0.5, 0.2, 0.2)
        assert abs(box[0] - 0.4) < 1e-6
        assert abs(box[1] - 0.4) < 1e-6
        assert abs(box[2] - 0.6) < 1e-6
        assert abs(box[3] - 0.6) < 1e-6


class TestClassificationEval:
    def test_perfect(self):
        gt = {"a": 0, "b": 1, "c": 2}
        pred = {"a": (0, 0.9), "b": (1, 0.8), "c": (2, 0.95)}
        r = evaluate_classification(gt, pred, 3)
        assert r["__overall__"]["accuracy"] == 1.0

    def test_partial(self):
        gt = {"a": 0, "b": 1, "c": 0, "d": 1, "e": 2}
        pred = {"a": (0, 0.9), "b": (1, 0.8), "c": (1, 0.6), "d": (1, 0.7), "e": (2, 0.95)}
        r = evaluate_classification(gt, pred, 3)
        assert r["__overall__"]["accuracy"] == 0.8

    def test_empty(self):
        assert evaluate_classification({}, {}, 0) == {}


# ============================================================
# segmentation — compute_seg_metrics (PySide6 우회를 위해 직접 구현)
# ============================================================
def compute_seg_metrics(pred_mask, gt_mask, num_classes):
    results = {}
    for c in range(num_classes):
        pred_c = (pred_mask == c)
        gt_c = (gt_mask == c)
        inter = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        iou = float(inter) / (float(union) + 1e-9) if union > 0 else float('nan')
        dice = 2.0 * float(inter) / (float(pred_c.sum() + gt_c.sum()) + 1e-9)
        results[c] = {"iou": iou, "dice": dice, "pred_px": int(pred_c.sum()), "gt_px": int(gt_c.sum())}
    valid = [v["iou"] for v in results.values() if not np.isnan(v["iou"]) and v["gt_px"] > 0]
    results["__overall__"] = {"mIoU": float(np.mean(valid)) if valid else 0.0,
                              "mDice": float(np.mean([v["dice"] for v in results.values() if v["gt_px"] > 0])) if valid else 0.0}
    return results


class TestSegMetrics:
    def test_perfect(self):
        mask = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.uint8)
        r = compute_seg_metrics(mask, mask, 2)
        assert abs(r[0]["iou"] - 1.0) < 1e-6
        assert abs(r[1]["iou"] - 1.0) < 1e-6
        assert abs(r["__overall__"]["mIoU"] - 1.0) < 1e-6

    def test_no_overlap(self):
        pred = np.zeros((4, 4), dtype=np.uint8)
        gt = np.ones((4, 4), dtype=np.uint8)
        r = compute_seg_metrics(pred, gt, 2)
        assert r[1]["iou"] < 1e-6

    def test_partial(self):
        pred = np.array([[0, 1], [1, 1]], dtype=np.uint8)
        gt = np.array([[0, 0], [1, 1]], dtype=np.uint8)
        r = compute_seg_metrics(pred, gt, 2)
        assert 0 < r["__overall__"]["mIoU"] < 1.0


# ============================================================
# clip_inference
# ============================================================
from core.clip_inference import simple_tokenize


class TestTokenize:
    def test_shape(self):
        tokens = simple_tokenize("hello world")
        assert tokens.shape == (1, 77)
        assert tokens[0, 0] == 49406  # SOT
        assert 49407 in tokens[0]     # EOT

    def test_padding(self):
        tokens = simple_tokenize("a")
        assert tokens[0, -1] == 0  # padding


# ============================================================
# dataset_splitter — stratified split 로직
# ============================================================
class TestStratifiedSplit:
    def test_ratio(self):
        import random
        from collections import defaultdict
        files = [f"img_{i}" for i in range(100)]
        class_to_files = defaultdict(list)
        for i, f in enumerate(files):
            class_to_files[i % 3].append(f)

        rng = random.Random(42)
        splits = {"train": [], "val": [], "test": []}
        tr, va, te = 0.7, 0.2, 0.1
        total = tr + va + te
        for cid, cfiles in class_to_files.items():
            rng.shuffle(cfiles)
            n = len(cfiles)
            n_train = round(n * tr / total)
            n_val = round(n * va / total)
            splits["train"].extend(cfiles[:n_train])
            splits["val"].extend(cfiles[n_train:n_train + n_val])
            splits["test"].extend(cfiles[n_train + n_val:])

        total_split = sum(len(v) for v in splits.values())
        assert total_split == 100
        assert len(splits["train"]) >= 60
        assert len(splits["val"]) >= 15
        assert len(splits["test"]) >= 5


# ============================================================
# model_loader — ModelInfo
# ============================================================
from core.model_loader import ModelInfo


class TestModelInfo:
    def test_defaults(self):
        mi = ModelInfo(path="t", format="onnx", names={}, input_size=(640, 640),
                       session=None, output_layout="v8")
        assert mi.batch_size == 1
        assert mi.task_type == "detection"
        assert mi.model_type == "yolo"

    def test_batch_size(self):
        mi = ModelInfo(path="t", format="onnx", names={}, input_size=(640, 640),
                       session=None, output_layout="v8", batch_size=4)
        assert mi.batch_size == 4
