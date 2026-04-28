"""Evaluation function unit tests."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from core.evaluation import evaluate_segmentation, evaluate_classification, evaluate_embedder


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
