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
# evaluation — core/evaluation.py 직접 import
# ============================================================
from core.evaluation import (
    _compute_iou, _compute_ap, _yolo_to_xyxy,
    evaluate_classification as _eval_cls_core,
    evaluate_segmentation as _eval_seg_core,
)


# evaluate_classification 래퍼 (테스트 시그니처 호환: num_classes 인자)
def evaluate_classification(gt_data, pred_data, num_classes=0):
    return _eval_cls_core(gt_data, pred_data)


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
# segmentation — core/evaluation.py 직접 import
# ============================================================
def compute_seg_metrics(pred_mask, gt_mask, num_classes):
    return _eval_seg_core(pred_mask, gt_mask, num_classes)


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


# ============================================================
# inference — postprocess_v8, postprocess_v5 단위 테스트
# ============================================================
from core.inference import postprocess_v8, postprocess_v5, _nms


class TestPostprocessV8:
    """YOLOv8 출력: (1, 4+N, 8400) 형태."""

    def _make_output(self, n_classes, n_anchors, detections):
        """detections: list of (cx, cy, w, h, class_id, score)"""
        out = np.zeros((1, 4 + n_classes, n_anchors), dtype=np.float32)
        for i, (cx, cy, w, h, cid, score) in enumerate(detections):
            out[0, 0, i] = cx
            out[0, 1, i] = cy
            out[0, 2, i] = w
            out[0, 3, i] = h
            out[0, 4 + cid, i] = score
        return out

    def test_empty_no_detections(self):
        out = np.zeros((1, 84, 8400), dtype=np.float32)
        r = postprocess_v8(out, 0.5, 1.0, (0, 0), (640, 640))
        assert len(r.boxes) == 0

    def test_single_detection(self):
        # 1 class, 1 anchor with high score at center
        out = self._make_output(1, 100, [(320, 320, 100, 100, 0, 0.9)])
        r = postprocess_v8(out, 0.5, 1.0, (0, 0), (640, 640))
        assert len(r.boxes) == 1
        assert r.scores[0] > 0.8
        assert r.class_ids[0] == 0

    def test_confidence_filter(self):
        out = self._make_output(2, 100, [
            (320, 320, 100, 100, 0, 0.9),
            (100, 100, 50, 50, 1, 0.1),  # below threshold
        ])
        r = postprocess_v8(out, 0.5, 1.0, (0, 0), (640, 640))
        assert len(r.boxes) == 1

    def test_coordinate_unscale(self):
        # ratio=0.5, pad=(0,0) → boxes should be 2x in original space
        out = self._make_output(1, 100, [(160, 160, 50, 50, 0, 0.9)])
        r = postprocess_v8(out, 0.5, 0.5, (0, 0), (640, 640))
        assert len(r.boxes) == 1
        # center at 160 with ratio 0.5 → original center at 320
        cx = (r.boxes[0][0] + r.boxes[0][2]) / 2
        assert abs(cx - 320) < 5


class TestPostprocessV5:
    """YOLOv5 출력: (1, 25200, 5+N) 형태."""

    def _make_output(self, n_classes, n_anchors, detections):
        """detections: list of (cx, cy, w, h, objectness, class_id, class_score)"""
        out = np.zeros((1, n_anchors, 5 + n_classes), dtype=np.float32)
        for i, (cx, cy, w, h, obj, cid, score) in enumerate(detections):
            out[0, i, 0] = cx
            out[0, i, 1] = cy
            out[0, i, 2] = w
            out[0, i, 3] = h
            out[0, i, 4] = obj
            out[0, i, 5 + cid] = score
        return out

    def test_empty(self):
        out = np.zeros((1, 100, 85), dtype=np.float32)
        r = postprocess_v5(out, 0.5, 1.0, (0, 0), (640, 640))
        assert len(r.boxes) == 0

    def test_single_detection(self):
        out = self._make_output(80, 100, [(320, 320, 100, 100, 0.95, 0, 0.9)])
        r = postprocess_v5(out, 0.5, 1.0, (0, 0), (640, 640))
        assert len(r.boxes) == 1
        assert r.class_ids[0] == 0

    def test_objectness_filter(self):
        out = self._make_output(2, 100, [
            (320, 320, 100, 100, 0.9, 0, 0.9),   # obj*cls = 0.81
            (100, 100, 50, 50, 0.1, 1, 0.9),      # obj*cls = 0.09 → filtered
        ])
        r = postprocess_v5(out, 0.5, 1.0, (0, 0), (640, 640))
        assert len(r.boxes) == 1


class TestNMS:
    def test_empty(self):
        assert _nms(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int32), 0.5) == []

    def test_no_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        cids = np.array([0, 0], dtype=np.int32)
        keep = _nms(boxes, scores, cids, 0.5)
        assert len(keep) == 2

    def test_full_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        cids = np.array([0, 0], dtype=np.int32)
        keep = _nms(boxes, scores, cids, 0.5)
        assert len(keep) == 1

    def test_different_classes(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        cids = np.array([0, 1], dtype=np.int32)
        keep = _nms(boxes, scores, cids, 0.5)
        assert len(keep) == 2  # different classes → both kept
