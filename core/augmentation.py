"""Augmentation with label co-transform.

The augmentation tab augments the IMAGE; this module adds the missing half —
transforming the YOLO bounding boxes alongside the pixels so a user can validate
an augmentation pipeline (image + labels together) before training.

Design split (deliberate, for memory-safe testability):
  * bbox-math functions (`flip_boxes_h`, `rotate_boxes`, `scale_boxes`,
    `translate_boxes`) operate purely on normalized YOLO boxes — no cv2, no
    image — so they are unit-testable analytically without loading opencv.
  * `augment_image_and_boxes` is the cv2-using entry point that applies the
    pixel transform and the matching box transform together.

YOLO box convention used throughout: a box is (cx, cy, w, h) in NORMALIZED
[0,1] image coordinates (the YOLO label format, sans class id). Functions take
and return numpy arrays of shape (N, 4) so the caller keeps the class column
separate.
"""
from __future__ import annotations

import math

import numpy as np


# ── Pure bbox math (no cv2 — analytically testable) ─────────────────


def _empty_boxes() -> np.ndarray:
    return np.zeros((0, 4), dtype=np.float32)


def flip_boxes_h(boxes_xywhn: np.ndarray) -> np.ndarray:
    """Horizontal (left-right) flip of normalized YOLO boxes.

    Only the x-center mirrors (cx -> 1 - cx); width/height/cy are unchanged.
    """
    boxes = np.asarray(boxes_xywhn, dtype=np.float32)
    if boxes.size == 0:
        return _empty_boxes()
    out = boxes.copy()
    out[:, 0] = 1.0 - boxes[:, 0]
    return out


def flip_boxes_v(boxes_xywhn: np.ndarray) -> np.ndarray:
    """Vertical (up-down) flip of normalized YOLO boxes (cy -> 1 - cy)."""
    boxes = np.asarray(boxes_xywhn, dtype=np.float32)
    if boxes.size == 0:
        return _empty_boxes()
    out = boxes.copy()
    out[:, 1] = 1.0 - boxes[:, 1]
    return out


def _xywhn_to_corners(boxes_xywhn: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """(N,4) normalized cxcywh → (N,4,2) absolute-pixel corner points (TL,TR,BR,BL)."""
    cx = boxes_xywhn[:, 0] * img_w
    cy = boxes_xywhn[:, 1] * img_h
    bw = boxes_xywhn[:, 2] * img_w
    bh = boxes_xywhn[:, 3] * img_h
    x1, y1 = cx - bw / 2.0, cy - bh / 2.0
    x2, y2 = cx + bw / 2.0, cy + bh / 2.0
    corners = np.stack([
        np.stack([x1, y1], axis=1),
        np.stack([x2, y1], axis=1),
        np.stack([x2, y2], axis=1),
        np.stack([x1, y2], axis=1),
    ], axis=1)  # (N, 4, 2)
    return corners


def _corners_to_xywhn(corners: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """(N,4,2) transformed corner points → axis-aligned normalized cxcywh, clipped.

    After rotation a box is no longer axis-aligned; YOLO labels are axis-aligned,
    so we take the enclosing axis-aligned bounding box of the rotated corners
    (the standard YOLO-augmentation convention).
    """
    xs = corners[:, :, 0]
    ys = corners[:, :, 1]
    x_min = xs.min(axis=1)
    x_max = xs.max(axis=1)
    y_min = ys.min(axis=1)
    y_max = ys.max(axis=1)
    # Clip to image bounds so boxes that rotate partly out of frame stay valid.
    x_min = np.clip(x_min, 0.0, img_w)
    x_max = np.clip(x_max, 0.0, img_w)
    y_min = np.clip(y_min, 0.0, img_h)
    y_max = np.clip(y_max, 0.0, img_h)
    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h
    return np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)


def _affine_matrix_rotate(angle_deg: float, img_w: int, img_h: int,
                          scale: float = 1.0) -> np.ndarray:
    """2x3 affine rotation matrix about the image center — matches cv2.getRotationMatrix2D.

    Built analytically (no cv2) so box rotation is testable without opencv and
    stays bit-identical to the pixel transform used by the route.
    """
    cx, cy = img_w / 2.0, img_h / 2.0
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a) * scale, math.sin(a) * scale
    # cv2 convention: positive angle = counter-clockwise.
    m = np.array([
        [cos_a, sin_a, (1 - cos_a) * cx - sin_a * cy],
        [-sin_a, cos_a, sin_a * cx + (1 - cos_a) * cy],
    ], dtype=np.float64)
    return m


def _apply_affine_to_corners(corners: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to (N,4,2) corner points."""
    pts = corners.reshape(-1, 2)  # (N*4, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homo = np.concatenate([pts, ones], axis=1)  # (N*4, 3)
    transformed = homo @ m.T  # (N*4, 2)
    return transformed.reshape(corners.shape).astype(np.float64)


def rotate_boxes(boxes_xywhn: np.ndarray, angle_deg: float,
                 img_w: int, img_h: int, scale: float = 1.0) -> np.ndarray:
    """Rotate normalized YOLO boxes by angle_deg about the image center.

    Returns axis-aligned enclosing boxes (YOLO labels are axis-aligned). Image
    dimensions are unchanged by cv2.warpAffine, so normalization uses the same
    (img_w, img_h) on both sides.
    """
    boxes = np.asarray(boxes_xywhn, dtype=np.float32)
    if boxes.size == 0:
        return _empty_boxes()
    corners = _xywhn_to_corners(boxes, img_w, img_h)
    m = _affine_matrix_rotate(angle_deg, img_w, img_h, scale)
    rotated = _apply_affine_to_corners(corners, m)
    return _corners_to_xywhn(rotated, img_w, img_h)


def scale_boxes(boxes_xywhn: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Scale about the image center by (fx, fy) in normalized space.

    A pixel-space zoom that keeps the image canvas size constant maps a
    normalized coordinate p -> 0.5 + (p - 0.5) * f about the 0.5 center.
    """
    boxes = np.asarray(boxes_xywhn, dtype=np.float32)
    if boxes.size == 0:
        return _empty_boxes()
    out = boxes.copy()
    out[:, 0] = 0.5 + (boxes[:, 0] - 0.5) * fx
    out[:, 1] = 0.5 + (boxes[:, 1] - 0.5) * fy
    out[:, 2] = boxes[:, 2] * fx
    out[:, 3] = boxes[:, 3] * fy
    return out


def translate_boxes(boxes_xywhn: np.ndarray, dx_n: float, dy_n: float) -> np.ndarray:
    """Translate normalized YOLO boxes by (dx_n, dy_n) in normalized units."""
    boxes = np.asarray(boxes_xywhn, dtype=np.float32)
    if boxes.size == 0:
        return _empty_boxes()
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] + dx_n
    out[:, 1] = boxes[:, 1] + dy_n
    return out


def clip_boxes(boxes_xywhn: np.ndarray, min_size: float = 1e-3) -> "tuple[np.ndarray, np.ndarray]":
    """Drop boxes whose center left the frame; clip the rest to [0,1].

    Returns (kept_boxes, kept_mask) so the caller can drop the matching class
    ids. A box whose center is outside the image is removed (it is no longer a
    meaningful label); a box clipped below `min_size` in either dim is dropped.
    """
    boxes = np.asarray(boxes_xywhn, dtype=np.float32)
    if boxes.size == 0:
        return _empty_boxes(), np.zeros(0, dtype=bool)
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    in_frame = (cx >= 0.0) & (cx <= 1.0) & (cy >= 0.0) & (cy <= 1.0)
    big_enough = (bw >= min_size) & (bh >= min_size)
    mask = in_frame & big_enough
    kept = boxes[mask].copy()
    if kept.size:
        kept[:, 0] = np.clip(kept[:, 0], 0.0, 1.0)
        kept[:, 1] = np.clip(kept[:, 1], 0.0, 1.0)
        kept[:, 2] = np.clip(kept[:, 2], 0.0, 1.0)
        kept[:, 3] = np.clip(kept[:, 3], 0.0, 1.0)
    return kept, mask


# ── YOLO label IO (pure text, no cv2) ───────────────────────────────


def parse_yolo_label(text: str) -> "tuple[np.ndarray, np.ndarray]":
    """Parse YOLO label text → (class_ids (N,), boxes_xywhn (N,4)).

    Malformed lines (non-numeric, too few tokens) are skipped rather than
    aborting — a single bad line should not discard the whole label.
    """
    class_ids: list[int] = []
    boxes: list[list[float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, bw, bh = (float(parts[1]), float(parts[2]),
                              float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        class_ids.append(cid)
        boxes.append([cx, cy, bw, bh])
    if not boxes:
        return np.zeros(0, dtype=np.int32), _empty_boxes()
    return (np.asarray(class_ids, dtype=np.int32),
            np.asarray(boxes, dtype=np.float32))


def format_yolo_label(class_ids: np.ndarray, boxes_xywhn: np.ndarray) -> str:
    """(class_ids, boxes_xywhn) → YOLO label text (one box per line)."""
    lines = []
    for cid, box in zip(class_ids, boxes_xywhn):
        lines.append(f"{int(cid)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
    return "\n".join(lines)


# ── Image + box co-transform (cv2 entry point) ──────────────────────

# Supported single-image ops that co-transform labels. Mosaic is intentionally
# excluded: it composites four images, so a single label set has no meaning.
LABEL_TRANSFORM_OPS = ("Flip", "Rotate", "Brightness")


def augment_image_and_boxes(
    image: np.ndarray,
    boxes_xywhn: np.ndarray,
    class_ids: np.ndarray,
    aug_type: str,
    angle: float = 15.0,
    alpha: float = 1.3,
    beta: float = 30.0,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Apply `aug_type` to both the image and its YOLO boxes together.

    Returns (aug_image, aug_boxes_xywhn, aug_class_ids). Boxes whose center
    leaves the frame after a geometric op are dropped (with their class ids).
    Brightness leaves boxes unchanged. Unknown ops return the image untouched
    with boxes/labels passed through.

    cv2 is imported lazily so the pure bbox-math functions above stay importable
    in environments without opencv.
    """
    import cv2

    boxes = np.asarray(boxes_xywhn, dtype=np.float32)
    cids = np.asarray(class_ids, dtype=np.int32)
    h, w = image.shape[:2]

    if aug_type == "Flip":
        aug_img = cv2.flip(image, 1)
        new_boxes = flip_boxes_h(boxes)
    elif aug_type == "Rotate":
        m = _affine_matrix_rotate(angle, w, h).astype(np.float32)
        aug_img = cv2.warpAffine(image, m, (w, h))
        new_boxes = rotate_boxes(boxes, angle, w, h)
    elif aug_type == "Brightness":
        aug_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        # Brightness is photometric — boxes are unchanged.
        return aug_img, boxes.copy(), cids.copy()
    else:
        return image.copy(), boxes.copy(), cids.copy()

    kept_boxes, mask = clip_boxes(new_boxes)
    kept_cids = cids[mask] if cids.size else cids
    return aug_img, kept_boxes, kept_cids


def draw_boxes_on_image(image: np.ndarray, boxes_xywhn: np.ndarray,
                        color: "tuple[int, int, int]" = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
    """Draw normalized YOLO boxes on a copy of the image (for preview)."""
    import cv2

    vis = image.copy()
    h, w = image.shape[:2]
    for box in np.asarray(boxes_xywhn, dtype=np.float32):
        cx, cy, bw, bh = box
        x1 = int(round((cx - bw / 2.0) * w))
        y1 = int(round((cy - bh / 2.0) * h))
        x2 = int(round((cx + bw / 2.0) * w))
        y2 = int(round((cy + bh / 2.0) * h))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    return vis
