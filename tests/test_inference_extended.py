"""Extended inference tests — pose, instance seg, segmentation results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from core.inference import (
    PoseResult, InstanceSegResult, SegmentationResult,
    DetectionResult, postprocess_pose_v8,
)


class TestPoseResult:
    def test_empty(self):
        r = PoseResult.empty()
        assert r.boxes.shape == (0, 4)
        assert r.scores.shape == (0,)
        assert r.keypoints.shape == (0, 17, 3)
        assert r.infer_ms == 0.0


class TestInstanceSegResult:
    def test_empty(self):
        r = InstanceSegResult.empty()
        assert r.boxes.shape == (0, 4)
        assert r.scores.shape == (0,)
        assert r.class_ids.shape == (0,)
        assert r.masks == []
        assert r.infer_ms == 0.0


class TestSegmentationResult:
    def test_basic(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        r = SegmentationResult(mask=mask, num_classes=2, infer_ms=5.0)
        assert r.mask.shape == (10, 10)
        assert r.num_classes == 2
        assert r.infer_ms == 5.0


class TestPostprocessPoseV8:
    def test_no_detections(self):
        # (1, 56, 8400) with all zeros → no detections above conf
        output = np.zeros((1, 56, 8400), dtype=np.float32)
        r = postprocess_pose_v8(output, conf=0.25, ratio=1.0, pad=(0, 0), orig_shape=(640, 640, 3))
        assert len(r.boxes) == 0

    def test_single_detection(self):
        # Create synthetic output: 4 box + 1 conf + 51 kpts = 56
        output = np.zeros((1, 56, 100), dtype=np.float32)
        # Set one detection with high confidence
        output[0, 0, 0] = 320  # cx
        output[0, 1, 0] = 320  # cy
        output[0, 2, 0] = 100  # w
        output[0, 3, 0] = 200  # h
        output[0, 4, 0] = 0.9  # conf
        # Set some keypoints
        for k in range(17):
            output[0, 5 + k*3, 0] = 300 + k  # x
            output[0, 6 + k*3, 0] = 300 + k  # y
            output[0, 7 + k*3, 0] = 0.8       # visibility
        r = postprocess_pose_v8(output, conf=0.25, ratio=1.0, pad=(0, 0), orig_shape=(640, 640, 3))
        assert len(r.boxes) == 1
        assert len(r.keypoints) == 1
        assert r.keypoints[0].shape == (17, 3)
        assert r.scores[0] > 0.5


class TestDetectionResultConversion:
    def test_pose_to_detection(self):
        """Verify PoseResult can be converted to DetectionResult."""
        pose = PoseResult(
            boxes=np.array([[10, 20, 100, 200]], dtype=np.float32),
            scores=np.array([0.9], dtype=np.float32),
            keypoints=np.random.rand(1, 17, 3).astype(np.float32),
            infer_ms=5.0,
        )
        det = DetectionResult(
            boxes=pose.boxes, scores=pose.scores,
            class_ids=np.zeros(len(pose.scores), dtype=np.int32),
            infer_ms=pose.infer_ms,
        )
        assert det.boxes.shape == (1, 4)
        assert det.class_ids[0] == 0

    def test_instseg_to_detection(self):
        """Verify InstanceSegResult can be converted to DetectionResult."""
        iseg = InstanceSegResult(
            boxes=np.array([[10, 20, 100, 200]], dtype=np.float32),
            scores=np.array([0.8], dtype=np.float32),
            class_ids=np.array([2], dtype=np.int32),
            masks=[np.zeros((100, 100), dtype=np.uint8)],
            infer_ms=3.0,
        )
        det = DetectionResult(
            boxes=iseg.boxes, scores=iseg.scores,
            class_ids=iseg.class_ids, infer_ms=iseg.infer_ms,
        )
        assert det.class_ids[0] == 2
