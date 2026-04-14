"""Object Tracking: ByteTrack & SORT implementations for ssook."""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Track:
    id: int
    box: np.ndarray          # xyxy
    score: float
    class_id: int
    age: int = 0             # frames since creation
    hits: int = 1            # total matched frames
    time_since_update: int = 0
    trajectory: list = field(default_factory=list)  # list of center points


def _iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes (N,4) and (M,4) → (N,M)"""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _linear_assignment(cost_matrix: np.ndarray, threshold: float):
    """Greedy assignment: returns (matches, unmatched_a, unmatched_b)"""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    matches, ua, ub = [], set(range(cost_matrix.shape[0])), set(range(cost_matrix.shape[1]))
    # greedy: pick best IoU pairs
    flat = cost_matrix.flatten()
    order = np.argsort(-flat)
    for idx in order:
        r, c = divmod(int(idx), cost_matrix.shape[1])
        if r not in ua or c not in ub:
            continue
        if cost_matrix[r, c] < threshold:
            break
        matches.append((r, c))
        ua.discard(r)
        ub.discard(c)
    return matches, list(ua), list(ub)


class SORTTracker:
    """Simple Online and Realtime Tracking (SORT) — IoU-based."""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame_count = 0

    def reset(self):
        self.tracks.clear()
        self._next_id = 1
        self.frame_count = 0

    def update(self, boxes: np.ndarray, scores: np.ndarray,
               class_ids: np.ndarray) -> list[Track]:
        self.frame_count += 1
        if len(boxes) == 0:
            for t in self.tracks:
                t.time_since_update += 1
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return [t for t in self.tracks if t.hits >= self.min_hits]

        # Match existing tracks with detections
        if self.tracks:
            track_boxes = np.array([t.box for t in self.tracks])
            iou_mat = _iou_batch(track_boxes, boxes)
            matches, unmatched_tracks, unmatched_dets = _linear_assignment(
                iou_mat, self.iou_threshold)
        else:
            matches, unmatched_tracks, unmatched_dets = [], [], list(range(len(boxes)))

        # Update matched tracks
        for t_idx, d_idx in matches:
            t = self.tracks[t_idx]
            t.box = boxes[d_idx]
            t.score = float(scores[d_idx])
            t.class_id = int(class_ids[d_idx])
            t.hits += 1
            t.time_since_update = 0
            cx, cy = (t.box[0]+t.box[2])/2, (t.box[1]+t.box[3])/2
            t.trajectory.append((float(cx), float(cy)))
            if len(t.trajectory) > 60:
                t.trajectory = t.trajectory[-60:]

        # Age unmatched tracks
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].time_since_update += 1

        # Create new tracks
        for d_idx in unmatched_dets:
            box = boxes[d_idx]
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            t = Track(id=self._next_id, box=box, score=float(scores[d_idx]),
                      class_id=int(class_ids[d_idx]),
                      trajectory=[(float(cx), float(cy))])
            self._next_id += 1
            self.tracks.append(t)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks if t.hits >= self.min_hits or self.frame_count <= self.min_hits]


class ByteTracker:
    """ByteTrack: two-stage association (high + low confidence)."""

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 high_thresh: float = 0.5, low_thresh: float = 0.1,
                 iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_threshold = iou_threshold
        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame_count = 0

    def reset(self):
        self.tracks.clear()
        self._next_id = 1
        self.frame_count = 0

    def update(self, boxes: np.ndarray, scores: np.ndarray,
               class_ids: np.ndarray) -> list[Track]:
        self.frame_count += 1
        if len(boxes) == 0:
            for t in self.tracks:
                t.time_since_update += 1
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return [t for t in self.tracks if t.hits >= self.min_hits]

        # Split detections into high/low confidence
        high_mask = scores >= self.high_thresh
        low_mask = (scores >= self.low_thresh) & (~high_mask)
        high_boxes, high_scores, high_cids = boxes[high_mask], scores[high_mask], class_ids[high_mask]
        low_boxes, low_scores, low_cids = boxes[low_mask], scores[low_mask], class_ids[low_mask]

        # First association: high-confidence detections
        unmatched_track_indices = list(range(len(self.tracks)))
        if self.tracks and len(high_boxes) > 0:
            track_boxes = np.array([self.tracks[i].box for i in unmatched_track_indices])
            iou_mat = _iou_batch(track_boxes, high_boxes)
            matches, um_tracks, um_dets_high = _linear_assignment(iou_mat, self.iou_threshold)
        else:
            matches, um_tracks = [], list(range(len(self.tracks)))
            um_dets_high = list(range(len(high_boxes)))

        for t_idx, d_idx in matches:
            t = self.tracks[unmatched_track_indices[t_idx]]
            t.box = high_boxes[d_idx]
            t.score = float(high_scores[d_idx])
            t.class_id = int(high_cids[d_idx])
            t.hits += 1
            t.time_since_update = 0
            cx, cy = (t.box[0]+t.box[2])/2, (t.box[1]+t.box[3])/2
            t.trajectory.append((float(cx), float(cy)))
            if len(t.trajectory) > 60:
                t.trajectory = t.trajectory[-60:]

        # Second association: low-confidence with remaining tracks
        remaining_tracks = [unmatched_track_indices[i] for i in um_tracks]
        if remaining_tracks and len(low_boxes) > 0:
            track_boxes = np.array([self.tracks[i].box for i in remaining_tracks])
            iou_mat = _iou_batch(track_boxes, low_boxes)
            matches2, um_tracks2, _ = _linear_assignment(iou_mat, self.iou_threshold)
            for t_idx, d_idx in matches2:
                t = self.tracks[remaining_tracks[t_idx]]
                t.box = low_boxes[d_idx]
                t.score = float(low_scores[d_idx])
                t.class_id = int(low_cids[d_idx])
                t.hits += 1
                t.time_since_update = 0
            for t_idx in um_tracks2:
                self.tracks[remaining_tracks[t_idx]].time_since_update += 1
        else:
            for t_idx in remaining_tracks:
                self.tracks[t_idx].time_since_update += 1

        # Create new tracks from unmatched high-confidence detections
        for d_idx in um_dets_high:
            box = high_boxes[d_idx]
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            t = Track(id=self._next_id, box=box, score=float(high_scores[d_idx]),
                      class_id=int(high_cids[d_idx]),
                      trajectory=[(float(cx), float(cy))])
            self._next_id += 1
            self.tracks.append(t)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks if t.hits >= self.min_hits or self.frame_count <= self.min_hits]


def create_tracker(tracker_type: str = "bytetrack", **kwargs):
    """Factory function to create a tracker instance."""
    if tracker_type == "sort":
        return SORTTracker(**kwargs)
    return ByteTracker(**kwargs)
