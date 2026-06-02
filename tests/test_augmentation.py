"""Pure-logic tests for core/augmentation bbox co-transform math.

These exercise ONLY the analytic bbox functions, which import numpy + math but
NOT cv2 — so they run in a cv2-less env. The image-side (cv2.warpAffine etc.)
is covered separately by the endpoint tests.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from core.augmentation import (
    flip_boxes_h, flip_boxes_v, rotate_boxes, scale_boxes, translate_boxes,
    clip_boxes, parse_yolo_label, format_yolo_label,
    _affine_matrix_rotate, _xywhn_to_corners,
)


class TestFlip:
    def test_horizontal_mirrors_cx(self):
        # A box centered at cx=0.25 must map to cx=0.75 after horizontal flip.
        boxes = np.array([[0.25, 0.4, 0.2, 0.3]], dtype=np.float32)
        out = flip_boxes_h(boxes)
        assert out[0, 0] == pytest.approx(0.75)
        # cy / w / h unchanged.
        assert out[0, 1] == pytest.approx(0.4)
        assert out[0, 2] == pytest.approx(0.2)
        assert out[0, 3] == pytest.approx(0.3)

    def test_horizontal_is_involution(self):
        boxes = np.array([[0.1, 0.5, 0.2, 0.2], [0.9, 0.3, 0.1, 0.1]], dtype=np.float32)
        assert np.allclose(flip_boxes_h(flip_boxes_h(boxes)), boxes)

    def test_vertical_mirrors_cy(self):
        boxes = np.array([[0.4, 0.2, 0.2, 0.3]], dtype=np.float32)
        out = flip_boxes_v(boxes)
        assert out[0, 1] == pytest.approx(0.8)
        assert out[0, 0] == pytest.approx(0.4)

    def test_empty_input(self):
        out = flip_boxes_h(np.zeros((0, 4), dtype=np.float32))
        assert out.shape == (0, 4)


class TestRotate:
    def test_centered_box_keeps_center(self):
        # A box centered at (0.5, 0.5) keeps its center under any rotation.
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        out = rotate_boxes(boxes, 90.0, 640, 640)
        assert out[0, 0] == pytest.approx(0.5, abs=1e-4)
        assert out[0, 1] == pytest.approx(0.5, abs=1e-4)

    def test_90deg_swaps_square_dims(self):
        # A square box rotated 90deg about center stays the same size (square).
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        out = rotate_boxes(boxes, 90.0, 640, 640)
        assert out[0, 2] == pytest.approx(0.2, abs=1e-3)
        assert out[0, 3] == pytest.approx(0.2, abs=1e-3)

    def test_90deg_swaps_rect_dims_on_square_canvas(self):
        # On a square canvas, a 90deg rotation of an axis-aligned rectangle swaps
        # its width/height (the enclosing axis-aligned box).
        boxes = np.array([[0.5, 0.5, 0.4, 0.2]], dtype=np.float32)
        out = rotate_boxes(boxes, 90.0, 640, 640)
        assert out[0, 2] == pytest.approx(0.2, abs=1e-3)
        assert out[0, 3] == pytest.approx(0.4, abs=1e-3)

    def test_zero_angle_is_identity(self):
        boxes = np.array([[0.3, 0.7, 0.2, 0.1]], dtype=np.float32)
        out = rotate_boxes(boxes, 0.0, 640, 480)
        assert np.allclose(out, boxes, atol=1e-5)

    def test_offcenter_box_moves(self):
        # 180deg rotation maps (cx,cy) -> (1-cx, 1-cy) on any canvas.
        boxes = np.array([[0.25, 0.25, 0.2, 0.2]], dtype=np.float32)
        out = rotate_boxes(boxes, 180.0, 640, 640)
        assert out[0, 0] == pytest.approx(0.75, abs=1e-3)
        assert out[0, 1] == pytest.approx(0.75, abs=1e-3)

    def test_affine_matrix_matches_cv2_convention(self):
        # cv2.getRotationMatrix2D(center, 0, 1) is identity-translation: the
        # center maps to itself.
        m = _affine_matrix_rotate(0.0, 100, 100)
        center = np.array([50.0, 50.0, 1.0])
        mapped = m @ center
        assert mapped[0] == pytest.approx(50.0)
        assert mapped[1] == pytest.approx(50.0)


class TestScale:
    def test_scale_up_grows_box(self):
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        out = scale_boxes(boxes, 2.0, 2.0)
        assert out[0, 2] == pytest.approx(0.4)
        assert out[0, 3] == pytest.approx(0.4)
        # Centered box stays centered.
        assert out[0, 0] == pytest.approx(0.5)
        assert out[0, 1] == pytest.approx(0.5)

    def test_scale_moves_offcenter(self):
        boxes = np.array([[0.25, 0.5, 0.1, 0.1]], dtype=np.float32)
        out = scale_boxes(boxes, 2.0, 1.0)
        # cx: 0.5 + (0.25-0.5)*2 = 0.0
        assert out[0, 0] == pytest.approx(0.0)
        assert out[0, 1] == pytest.approx(0.5)


class TestTranslate:
    def test_shifts_center(self):
        boxes = np.array([[0.4, 0.4, 0.2, 0.2]], dtype=np.float32)
        out = translate_boxes(boxes, 0.1, -0.2)
        assert out[0, 0] == pytest.approx(0.5)
        assert out[0, 1] == pytest.approx(0.2)
        # Size unchanged.
        assert out[0, 2] == pytest.approx(0.2)


class TestClip:
    def test_drops_box_with_center_out_of_frame(self):
        boxes = np.array([[1.2, 0.5, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        kept, mask = clip_boxes(boxes)
        assert kept.shape[0] == 1
        assert mask.tolist() == [False, True]

    def test_keeps_in_frame_box(self):
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        kept, mask = clip_boxes(boxes)
        assert kept.shape[0] == 1
        assert mask.tolist() == [True]

    def test_drops_degenerate_box(self):
        boxes = np.array([[0.5, 0.5, 0.0, 0.2]], dtype=np.float32)
        kept, mask = clip_boxes(boxes)
        assert kept.shape[0] == 0


class TestYoloLabelIO:
    def test_parse_basic(self):
        text = "0 0.5 0.5 0.2 0.3\n2 0.1 0.1 0.05 0.05\n"
        cids, boxes = parse_yolo_label(text)
        assert cids.tolist() == [0, 2]
        assert boxes.shape == (2, 4)
        assert boxes[0].tolist() == pytest.approx([0.5, 0.5, 0.2, 0.3])

    def test_parse_skips_malformed_lines(self):
        # A non-numeric/header line is skipped, not fatal.
        text = "foo bar baz qux quux\n1 0.5 0.5 0.2 0.2\n"
        cids, boxes = parse_yolo_label(text)
        assert cids.tolist() == [1]
        assert boxes.shape == (1, 4)

    def test_parse_skips_short_lines(self):
        text = "0 0.5 0.5\n1 0.5 0.5 0.2 0.2\n"
        cids, _ = parse_yolo_label(text)
        assert cids.tolist() == [1]

    def test_parse_empty(self):
        cids, boxes = parse_yolo_label("")
        assert cids.shape == (0,)
        assert boxes.shape == (0, 4)

    def test_format_roundtrip(self):
        cids = np.array([0, 3], dtype=np.int32)
        boxes = np.array([[0.5, 0.5, 0.2, 0.3], [0.25, 0.75, 0.1, 0.1]], dtype=np.float32)
        text = format_yolo_label(cids, boxes)
        cids2, boxes2 = parse_yolo_label(text)
        assert cids2.tolist() == cids.tolist()
        assert np.allclose(boxes2, boxes, atol=1e-5)


class TestCornersHelper:
    def test_xywhn_to_corners_shape_and_values(self):
        boxes = np.array([[0.5, 0.5, 0.4, 0.2]], dtype=np.float32)
        corners = _xywhn_to_corners(boxes, 100, 100)
        assert corners.shape == (1, 4, 2)
        # TL = (30,40), BR = (70,60) for a 0.4x0.2 box centered at (50,50).
        assert corners[0, 0].tolist() == pytest.approx([30.0, 40.0])
        assert corners[0, 2].tolist() == pytest.approx([70.0, 60.0])
