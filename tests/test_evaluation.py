"""Evaluation function unit tests."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from core.evaluation import (
    evaluate_segmentation, evaluate_classification, evaluate_embedder,
    evaluate_dataset, evaluate_map50_95, _match_greedy,
)


class TestMatchGreedy:
    def test_picks_best_unmatched(self):
        iou = np.array([0.9, 0.6])
        mask = np.zeros(2, dtype=bool)
        assert _match_greedy(iou, mask, 0.5) == 0
        assert mask[0] and not mask[1]
        # second prediction with same row must now claim col 1, not re-book col 0
        assert _match_greedy(iou, mask, 0.5) == 1
        assert mask[1]

    def test_below_threshold_returns_minus_one(self):
        iou = np.array([0.4])
        mask = np.zeros(1, dtype=bool)
        assert _match_greedy(iou, mask, 0.5) == -1
        assert not mask[0]

    def test_empty_row(self):
        assert _match_greedy(np.empty(0), np.empty(0, dtype=bool), 0.5) == -1


class TestEvaluateDetection:
    def test_clustered_two_box_perfect(self):
        # EVAL-01: two GT, two perfectly matched preds -> map both 1.0 (no double-booking)
        gt = {"img": [(0, 0.2, 0.2, 0.1, 0.1), (0, 0.8, 0.8, 0.1, 0.1)]}
        pred = {"img": [(0, 0.2, 0.2, 0.1, 0.1, 0.9), (0, 0.8, 0.8, 0.1, 0.1, 0.8)]}
        r50 = evaluate_dataset(gt, pred, 0.5)
        assert r50["__overall__"]["ap"] == pytest.approx(1.0, abs=1e-6)
        assert r50["__overall__"]["recall"] == pytest.approx(1.0, abs=1e-6)
        assert evaluate_map50_95(gt, pred) == pytest.approx(1.0, abs=1e-6)

    def test_single_perfect_match(self):
        gt = {"i": [(0, 0.5, 0.5, 0.2, 0.2)]}
        pred = {"i": [(0, 0.5, 0.5, 0.2, 0.2, 0.9)]}
        assert evaluate_map50_95(gt, pred) == pytest.approx(1.0, abs=1e-6)

    def test_map5095_le_map50_invariant(self):
        gt = {"a": [(0, 0.5, 0.5, 0.2, 0.2), (0, 0.1, 0.1, 0.1, 0.1)]}
        pred = {"a": [(0, 0.52, 0.52, 0.2, 0.2, 0.9), (0, 0.3, 0.3, 0.15, 0.15, 0.5)]}
        m50 = evaluate_dataset(gt, pred, 0.5)["__overall__"]["ap"]
        m5095 = evaluate_map50_95(gt, pred)
        assert m5095 <= m50 + 1e-9

    def test_no_double_booking_two_preds_one_gt(self):
        # One GT, two overlapping preds -> exactly one TP (the higher score), one FP
        gt = {"i": [(0, 0.5, 0.5, 0.2, 0.2)]}
        pred = {"i": [(0, 0.5, 0.5, 0.2, 0.2, 0.9), (0, 0.51, 0.51, 0.2, 0.2, 0.4)]}
        r = evaluate_dataset(gt, pred, 0.5)["__overall__"]
        assert r["recall"] == pytest.approx(1.0, abs=1e-6)
        # precision = 1 TP / 2 preds = 0.5
        assert r["precision"] == pytest.approx(0.5, abs=1e-3)

    def test_per_class_tp_fp_fn_surfaced(self):
        # tp/fp/fn must now be in the per-class result so the route can render them.
        gt = {"i": [(0, 0.5, 0.5, 0.2, 0.2)]}
        pred = {"i": [(0, 0.5, 0.5, 0.2, 0.2, 0.9), (0, 0.1, 0.1, 0.05, 0.05, 0.3)]}
        r = evaluate_dataset(gt, pred, 0.5)
        assert r[0]["tp"] == 1 and r[0]["fp"] == 1 and r[0]["fn"] == 0


class TestMacroMicroAveraging:
    """macro vs micro pooling of __overall__ P/R/F1 (per-class + AP unchanged)."""

    # Imbalanced two-class set: a huge perfect class drowns out a tiny failing class
    # under micro pooling, but macro weights them equally.
    GT = {
        # class 0: 4 easy GT boxes, all detected perfectly (4 TP)
        "a": [(0, 0.2, 0.2, 0.1, 0.1), (0, 0.8, 0.8, 0.1, 0.1)],
        "b": [(0, 0.2, 0.8, 0.1, 0.1), (0, 0.8, 0.2, 0.1, 0.1)],
        # class 1: a single GT box that the model entirely misses (0 TP, 1 FN)
        "c": [(1, 0.5, 0.5, 0.2, 0.2)],
    }
    PRED = {
        "a": [(0, 0.2, 0.2, 0.1, 0.1, 0.9), (0, 0.8, 0.8, 0.1, 0.1, 0.9)],
        "b": [(0, 0.2, 0.8, 0.1, 0.1, 0.9), (0, 0.8, 0.2, 0.1, 0.1, 0.9)],
        "c": [],  # class 1 not predicted at all
    }

    def test_micro_recall_dominated_by_large_class(self):
        r = evaluate_dataset(self.GT, self.PRED, 0.5, averaging="micro")["__overall__"]
        # micro recall = total TP / total GT = 4 / 5 = 0.8 — large class dominates
        assert r["recall"] == pytest.approx(4 / 5, abs=1e-6)
        assert r["averaging"] == "micro"

    def test_macro_recall_equal_weight(self):
        r = evaluate_dataset(self.GT, self.PRED, 0.5, averaging="macro")["__overall__"]
        # macro recall = mean(class0_recall=1.0, class1_recall=0.0) = 0.5
        assert r["recall"] == pytest.approx(0.5, abs=1e-6)
        assert r["averaging"] == "macro"

    def test_macro_and_micro_differ_on_imbalanced_set(self):
        micro = evaluate_dataset(self.GT, self.PRED, 0.5, averaging="micro")["__overall__"]
        macro = evaluate_dataset(self.GT, self.PRED, 0.5, averaging="macro")["__overall__"]
        assert micro["recall"] != pytest.approx(macro["recall"], abs=1e-3)

    def test_default_is_micro_legacy_behavior(self):
        # Omitting averaging must reproduce the pre-option micro pooling exactly.
        default = evaluate_dataset(self.GT, self.PRED, 0.5)["__overall__"]
        micro = evaluate_dataset(self.GT, self.PRED, 0.5, averaging="micro")["__overall__"]
        assert default["recall"] == pytest.approx(micro["recall"], abs=1e-9)
        assert default["precision"] == pytest.approx(micro["precision"], abs=1e-9)
        assert default["f1"] == pytest.approx(micro["f1"], abs=1e-9)

    def test_per_class_and_ap_unchanged_by_averaging(self):
        # Only __overall__ P/R/F1 change; per-class metrics + mAP are identical.
        micro = evaluate_dataset(self.GT, self.PRED, 0.5, averaging="micro")
        macro = evaluate_dataset(self.GT, self.PRED, 0.5, averaging="macro")
        for cid in (0, 1):
            for k in ("ap", "precision", "recall", "f1", "tp", "fp", "fn"):
                assert micro[cid][k] == pytest.approx(macro[cid][k], abs=1e-9)
        assert micro["__overall__"]["ap"] == pytest.approx(macro["__overall__"]["ap"], abs=1e-9)


class TestEvaluateSegmentation:
    def test_perfect_match(self):
        mask = np.array([[0, 1], [1, 2]], dtype=np.uint8)
        r = evaluate_segmentation(mask, mask, 3)
        assert r["__overall__"]["mIoU"] == pytest.approx(1.0, abs=1e-6)
        assert r["__overall__"]["mDice"] == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap(self):
        pred = np.array([[0, 0], [0, 0]], dtype=np.uint8)
        gt = np.array([[1, 1], [1, 1]], dtype=np.uint8)
        r = evaluate_segmentation(pred, gt, 2)
        assert r[1]["iou"] == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap(self):
        pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        gt = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        r = evaluate_segmentation(pred, gt, 2)
        # class 1: inter=1, union=3, iou=1/3
        assert r[1]["iou"] == pytest.approx(1/3, abs=1e-4)

    def test_empty_masks(self):
        pred = np.zeros((4, 4), dtype=np.uint8)
        gt = np.zeros((4, 4), dtype=np.uint8)
        r = evaluate_segmentation(pred, gt, 2)
        assert "__overall__" in r


class TestEvaluateClassification:
    def test_perfect(self):
        gt = {"a": 0, "b": 1, "c": 0}
        pred = {"a": (0, 0.9), "b": (1, 0.8), "c": (0, 0.7)}
        r = evaluate_classification(gt, pred)
        assert r["__overall__"]["accuracy"] == pytest.approx(1.0)

    def test_all_wrong(self):
        gt = {"a": 0, "b": 1}
        pred = {"a": (1, 0.9), "b": (0, 0.8)}
        r = evaluate_classification(gt, pred)
        assert r["__overall__"]["accuracy"] == pytest.approx(0.0)

    def test_empty(self):
        r = evaluate_classification({}, {})
        assert r == {}

    def test_partial_overlap(self):
        gt = {"a": 0, "b": 1, "c": 2}
        pred = {"a": (0, 0.9), "b": (1, 0.8)}  # c missing
        r = evaluate_classification(gt, pred)
        assert r["__overall__"]["accuracy"] == pytest.approx(1.0)  # only matched stems


class TestEvaluateEmbedder:
    def test_basic(self):
        q = [np.array([1, 0, 0], dtype=np.float32)]
        g = [np.array([1, 0, 0], dtype=np.float32), np.array([0, 1, 0], dtype=np.float32)]
        ql = [0]
        gl = [0, 1]
        r = evaluate_embedder(q, g, ql, gl, top_k=2)
        assert r["retrieval_at_1"] == pytest.approx(1.0)

    def test_empty(self):
        r = evaluate_embedder([], [], [], [])
        assert r == {}

    def test_random(self):
        np.random.seed(42)
        q = [np.random.randn(128).astype(np.float32) for _ in range(5)]
        g = [np.random.randn(128).astype(np.float32) for _ in range(20)]
        ql = list(range(5))
        gl = list(range(20))
        r = evaluate_embedder(q, g, ql, gl, top_k=5)
        assert "retrieval_at_1" in r
        assert "mean_cosine_sim" in r


# ── Route-level tests (HEAVY: import server.eval_routes -> cv2/onnxruntime). ──
# Memory-safe protocol: DO NOT run these in the per-agent pass; they are run
# centrally. They cover the GT-missing warning surfacing + batch eligibility.
eval_routes_importable = True
try:  # pragma: no cover - environment dependent
    import server.eval_routes as _er  # noqa: F401
except Exception:
    eval_routes_importable = False


@pytest.mark.skipif(not eval_routes_importable,
                    reason="server.eval_routes needs cv2/onnxruntime (run centrally)")
class TestEvalRouteHelpers:
    def test_can_batch_excludes_specialized_types(self):
        # Helper guarding which model_types may use run_inference_batch: only the
        # generic letterbox->v8/v5 path is batch-equivalent to per-frame inference.
        from types import SimpleNamespace
        from server.eval_routes import _can_batch
        for mt in ("yolo", "darknet", "yolox", "yolov10", "efficientdet"):
            assert _can_batch(SimpleNamespace(model_type=mt)), mt
        for mt in ("detr", "custom", "yolo_nas",
                   "pose_yolo", "instseg_yolo", "seq_yolo", "seq_rfdetr"):
            assert not _can_batch(SimpleNamespace(model_type=mt)), mt

    def test_gt_missing_all(self):
        from server.eval_routes import _gt_missing_warnings
        w = _gt_missing_warnings(missing_gt=3, total=3, label_dir="C:/bad/labels")
        assert len(w) == 1 and "no GT labels found" in w[0] and "C:/bad/labels" in w[0]

    def test_gt_missing_some(self):
        from server.eval_routes import _gt_missing_warnings
        w = _gt_missing_warnings(missing_gt=1, total=3, label_dir="x")
        assert len(w) == 1 and "1 of 3 images" in w[0]

    def test_gt_missing_none(self):
        from server.eval_routes import _gt_missing_warnings
        assert _gt_missing_warnings(missing_gt=0, total=3, label_dir="x") == []

    def test_gt_missing_empty_dataset_no_warning(self):
        # No images at all -> no spurious "no GT labels" warning.
        from server.eval_routes import _gt_missing_warnings
        assert _gt_missing_warnings(missing_gt=0, total=0, label_dir="") == []
