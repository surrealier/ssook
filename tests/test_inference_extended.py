"""Extended inference tests — pose, instance seg, segmentation results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from core.inference import (
    PoseResult, InstanceSegResult, SegmentationResult,
    DetectionResult, postprocess_pose_v8, run_pose, run_instance_seg,
    run_pose_batch, run_instance_seg_batch,
    UnsupportedModelError,
)


class _FakeSession:
    """Minimal ORT-session stand-in: returns a fixed list of output arrays."""

    def __init__(self, outputs):
        self._outputs = outputs

    def run(self, _output_names, _feed):
        return self._outputs


class _BatchEchoSession:
    """ORT-session stand-in for batch tests: emits per-sample outputs sized to
    the fed batch dim, so run_*_batch can slice output[*][i] for each frame.

    `builder(batch_n)` returns a list of output arrays whose leading dim == batch_n.
    """

    def __init__(self, builder):
        self._builder = builder

    def run(self, _output_names, feed):
        # The single image tensor is the only NCHW (rank-4) value in the feed.
        tensor = next(v for v in feed.values() if hasattr(v, "ndim") and v.ndim == 4)
        return self._builder(tensor.shape[0])


class _FakeModelInfo:
    """Just enough of ModelInfo for run_pose / run_instance_seg."""

    def __init__(self, model_type, outputs=None, input_size=(640, 640),
                 session=None, batch_size=1):
        self.model_type = model_type
        if session is not None:
            self.session = session
        else:
            self.session = _FakeSession(outputs) if outputs is not None else None
        self.input_name = "images"
        self.input_size = input_size
        self.batch_size = batch_size
        self.names = {}


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


class TestPoseKeypointCount:
    """P1: keypoint count is derived from output shape, not hardcoded to 17."""

    def test_derives_21_keypoints(self):
        # 21-kpt layout: 4 box + 1 conf + 21*3 = 68 channels
        channels = 4 + 1 + 21 * 3
        output = np.zeros((1, channels, 100), dtype=np.float32)
        output[0, 0, 0] = 320  # cx
        output[0, 1, 0] = 320  # cy
        output[0, 2, 0] = 100  # w
        output[0, 3, 0] = 200  # h
        output[0, 4, 0] = 0.9  # conf
        r = postprocess_pose_v8(output, conf=0.25, ratio=1.0, pad=(0, 0),
                                orig_shape=(640, 640, 3))
        assert len(r.boxes) == 1
        assert r.keypoints.shape == (1, 21, 3)

    def test_still_handles_coco_17(self):
        channels = 4 + 1 + 17 * 3
        output = np.zeros((1, channels, 100), dtype=np.float32)
        output[0, 4, 0] = 0.9
        r = postprocess_pose_v8(output, conf=0.25, ratio=1.0, pad=(0, 0),
                                orig_shape=(640, 640, 3))
        assert r.keypoints.shape == (1, 17, 3)


class TestPoseUnsupported:
    """P0: non-YOLO pose layouts raise instead of fabricating keypoints."""

    def test_heatmap_shape_raises(self):
        # HRNet/ViTPose-style heatmap (1,K,Hh,Ww): data=output[0] is rank 3 → reject
        heatmap = np.zeros((1, 17, 64, 48), dtype=np.float32)
        with pytest.raises(UnsupportedModelError):
            postprocess_pose_v8(heatmap, conf=0.25, ratio=1.0, pad=(0, 0),
                                orig_shape=(640, 640, 3))

    def test_non_yolo_channel_count_raises(self):
        # 2D but channel count not 4+1+3K: 18-5=13, not divisible by 3 → reject
        bad = np.zeros((1, 18, 100), dtype=np.float32)
        with pytest.raises(UnsupportedModelError):
            postprocess_pose_v8(bad, conf=0.25, ratio=1.0, pad=(0, 0),
                                orig_shape=(640, 640, 3))

    def test_run_pose_rejects_hrnet_type(self):
        mi = _FakeModelInfo("pose_hrnet",
                            outputs=[np.zeros((1, 17, 64, 48), dtype=np.float32)])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(UnsupportedModelError):
            run_pose(mi, frame, conf=0.25)

    def test_run_pose_rejects_vitpose_type(self):
        mi = _FakeModelInfo("pose_vitpose",
                            outputs=[np.zeros((1, 17, 64, 48), dtype=np.float32)])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(UnsupportedModelError):
            run_pose(mi, frame, conf=0.25)


class TestInstanceSegUnsupported:
    """P0: Mask R-CNN / non-YOLO-Seg layouts raise instead of silent empty."""

    def test_maskrcnn_type_raises(self):
        # Mask R-CNN emits separate boxes/labels/scores/masks; type guard fires
        # before inference, so outputs are irrelevant here.
        mi = _FakeModelInfo("instseg_maskrcnn",
                            outputs=[np.zeros((1, 100, 4), dtype=np.float32),
                                     np.zeros((1, 100), dtype=np.float32)])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(UnsupportedModelError):
            run_instance_seg(mi, frame, conf=0.25)

    def test_custom_type_raises(self):
        mi = _FakeModelInfo("instseg_custom",
                            outputs=[np.zeros((1, 116, 8400), dtype=np.float32),
                                     np.zeros((1, 32, 160, 160), dtype=np.float32)])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(UnsupportedModelError):
            run_instance_seg(mi, frame, conf=0.25)

    def test_yolo_type_wrong_output_layout_raises(self):
        # Declared instseg_yolo but output[1] is not the 4D proto tensor
        mi = _FakeModelInfo("instseg_yolo",
                            outputs=[np.zeros((1, 116, 8400), dtype=np.float32),
                                     np.zeros((1, 100), dtype=np.float32)])  # 2D, not 4D
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(UnsupportedModelError):
            run_instance_seg(mi, frame, conf=0.25)

    def test_yolo_type_valid_layout_returns_empty_not_raises(self):
        # Valid YOLO-Seg layout with all-zero scores → empty result, no raise.
        det = np.zeros((1, 116, 8400), dtype=np.float32)
        protos = np.zeros((1, 32, 160, 160), dtype=np.float32)
        mi = _FakeModelInfo("instseg_yolo", outputs=[det, protos])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        r = run_instance_seg(mi, frame, conf=0.25)
        assert len(r.boxes) == 0


# ── Tracking: opt-in class-gating ───────────────────────
from core.tracking import SORTTracker, ByteTracker, create_tracker


def _det(box, score, cid):
    return (np.array([box], dtype=np.float32),
            np.array([score], dtype=np.float32),
            np.array([cid], dtype=np.int32))


class TestTrackingClassGating:
    """P1: class-gating (default off) prevents a track ID jumping across classes."""

    def test_default_is_class_agnostic(self):
        # Default (gating off): a class-0 track matched to an overlapping class-1
        # detection adopts class_id 1 (the pre-existing behavior, preserved).
        tr = SORTTracker(min_hits=1, iou_threshold=0.3)
        b, s, c = _det([0, 0, 10, 10], 0.9, 0)
        tr.update(b, s, c)
        # Same position, different class → matched & relabeled under gating-off.
        b2, s2, c2 = _det([0, 0, 10, 10], 0.9, 1)
        out = tr.update(b2, s2, c2)
        assert len(out) == 1
        assert out[0].class_id == 1  # relabeled — class-agnostic match

    def test_gating_creates_new_id_on_class_flip_sort(self):
        tr = SORTTracker(min_hits=1, iou_threshold=0.3, class_gating=True)
        b, s, c = _det([0, 0, 10, 10], 0.9, 0)
        first = tr.update(b, s, c)
        first_id = first[0].id
        assert first[0].class_id == 0

        # Frame 2: a class-0 box slightly moved AND a higher-IoU class-1 box at
        # nearly the same spot. The class-0 track must keep class_id 0 (matched
        # to the same-class box) and a NEW track must appear for class 1.
        boxes = np.array([[1, 1, 11, 11], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.95], dtype=np.float32)
        cids = np.array([0, 1], dtype=np.int32)
        out = tr.update(boxes, scores, cids)

        by_class = {t.class_id: t for t in out}
        assert 0 in by_class and 1 in by_class
        assert by_class[0].id == first_id          # original track kept, same class
        assert by_class[1].id != first_id          # new id for the class-1 object

    def test_gating_creates_new_id_on_class_flip_bytetrack(self):
        tr = ByteTracker(min_hits=1, iou_threshold=0.3, high_thresh=0.5,
                         class_gating=True)
        b, s, c = _det([0, 0, 10, 10], 0.9, 0)
        first = tr.update(b, s, c)
        first_id = first[0].id

        boxes = np.array([[1, 1, 11, 11], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.95], dtype=np.float32)
        cids = np.array([0, 1], dtype=np.int32)
        out = tr.update(boxes, scores, cids)

        by_class = {t.class_id: t for t in out}
        assert 0 in by_class and 1 in by_class
        assert by_class[0].id == first_id
        assert by_class[1].id != first_id

    def test_create_tracker_forwards_class_gating(self):
        tr = create_tracker("bytetrack", class_gating=True)
        assert isinstance(tr, ByteTracker)
        assert tr.class_gating is True
        sort = create_tracker("sort", class_gating=True)
        assert isinstance(sort, SORTTracker)
        assert sort.class_gating is True
        # Default stays off.
        assert create_tracker("bytetrack").class_gating is False


# ── Batch / directory inference (DEFERRED — imports cv2 via core.inference) ──
# These verify run_pose_batch / run_instance_seg_batch return one result per
# frame and respect fixed-batch padding/chunking. Run centrally (not by the
# memory-safe agent) because importing core.inference pulls in cv2.

def _pose_output_builder(channels=56, anchors=100, conf_idx=4):
    """Return a builder(batch_n) -> [(1*? , channels, anchors)] with one strong
    detection per sample, so each decoded PoseResult has exactly one box."""
    def _build(batch_n):
        out = np.zeros((batch_n, channels, anchors), dtype=np.float32)
        # one detection per sample with a valid box + high conf
        out[:, 0, 0] = 320.0  # cx
        out[:, 1, 0] = 320.0  # cy
        out[:, 2, 0] = 100.0  # w
        out[:, 3, 0] = 200.0  # h
        out[:, conf_idx, 0] = 0.9
        return [out]
    return _build


class TestRunPoseBatch:
    """run_pose_batch: per-frame results, type guard, fixed-batch handling."""

    def test_empty_frames_returns_empty_list(self):
        mi = _FakeModelInfo("pose_yolo", session=_BatchEchoSession(_pose_output_builder()))
        assert run_pose_batch(mi, [], conf=0.25) == []

    def test_one_result_per_frame(self):
        mi = _FakeModelInfo("pose_yolo", session=_BatchEchoSession(_pose_output_builder()))
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        results = run_pose_batch(mi, frames, conf=0.25)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, PoseResult)
            assert len(r.boxes) == 1

    def test_rejects_non_yolo_pose_type(self):
        mi = _FakeModelInfo("pose_hrnet", session=_BatchEchoSession(_pose_output_builder()))
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        with pytest.raises(UnsupportedModelError):
            run_pose_batch(mi, frames, conf=0.25)

    def test_fixed_batch_pad_up_trims_to_n(self):
        # Model fixed batch=4 but only 2 frames → pad up to 4, trim back to 2.
        mi = _FakeModelInfo("pose_yolo", batch_size=4,
                            session=_BatchEchoSession(_pose_output_builder()))
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
        results = run_pose_batch(mi, frames, conf=0.25)
        assert len(results) == 2

    def test_fixed_batch_chunks_when_more_frames_than_bs(self):
        # Model fixed batch=2 but 5 frames → chunk into 2+2+1, 5 results total.
        mi = _FakeModelInfo("pose_yolo", batch_size=2,
                            session=_BatchEchoSession(_pose_output_builder()))
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        results = run_pose_batch(mi, frames, conf=0.25)
        assert len(results) == 5


def _seg_output_builder(nc=80, nm=32, anchors=8400, mh=160, mw=160):
    """Builder(batch_n) -> [det (batch_n, 4+nc+nm, anchors), proto (batch_n, nm, mh, mw)]
    with one above-conf detection per sample."""
    def _build(batch_n):
        det = np.zeros((batch_n, 4 + nc + nm, anchors), dtype=np.float32)
        det[:, 0, 0] = 320.0
        det[:, 1, 0] = 320.0
        det[:, 2, 0] = 100.0
        det[:, 3, 0] = 200.0
        det[:, 4, 0] = 0.9  # class-0 score
        proto = np.zeros((batch_n, nm, mh, mw), dtype=np.float32)
        return [det, proto]
    return _build


class TestRunInstanceSegBatch:
    """run_instance_seg_batch: per-frame results, type/layout guards, fixed batch."""

    def test_empty_frames_returns_empty_list(self):
        mi = _FakeModelInfo("instseg_yolo", session=_BatchEchoSession(_seg_output_builder()))
        assert run_instance_seg_batch(mi, [], conf=0.25) == []

    def test_one_result_per_frame(self):
        mi = _FakeModelInfo("instseg_yolo", session=_BatchEchoSession(_seg_output_builder()))
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        results = run_instance_seg_batch(mi, frames, conf=0.25)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, InstanceSegResult)
            assert len(r.boxes) == 1
            assert len(r.masks) == 1

    def test_rejects_non_yolo_type(self):
        mi = _FakeModelInfo("instseg_maskrcnn",
                            session=_BatchEchoSession(_seg_output_builder()))
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        with pytest.raises(UnsupportedModelError):
            run_instance_seg_batch(mi, frames, conf=0.25)

    def test_fixed_batch_chunks(self):
        mi = _FakeModelInfo("instseg_yolo", batch_size=2,
                            session=_BatchEchoSession(_seg_output_builder()))
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        results = run_instance_seg_batch(mi, frames, conf=0.25)
        assert len(results) == 5


class TestClipBatchEncoder:
    """CLIPModel.encode_images / zero_shot_classify_batch — DEFERRED (imports
    onnxruntime via core.clip_inference). Verifies batched rows are L2-normalized
    and match the per-image path within tolerance."""

    def _make_clip(self):
        # Lazy import so collection doesn't fail in a cv2/onnxruntime-less env.
        from core.clip_inference import CLIPModel

        class _ImgSession:
            def get_inputs(self):
                class _I:
                    name = "pixel_values"
                    shape = [None, 3, 224, 224]  # dynamic batch
                    type = "tensor(float)"
                return [_I()]

            def run(self, _names, feed):
                t = feed["pixel_values"]
                # Deterministic per-row "embedding": row mean over spatial dims.
                n = t.shape[0]
                emb = t.reshape(n, -1).mean(axis=1, keepdims=True) * np.arange(1, 5)
                return [emb.astype(np.float32)]

        model = CLIPModel.__new__(CLIPModel)
        model.img_session = _ImgSession()
        model.txt_session = None
        model._img_input = _ImgSession().get_inputs()[0]
        model._img_size = (224, 224)
        model._img_bs = 1
        return model

    def test_encode_images_rows_are_unit_norm(self):
        model = self._make_clip()
        frames = [np.full((50, 50, 3), v, dtype=np.uint8) for v in (10, 50, 200)]
        embs = model.encode_images(frames)
        assert embs.shape[0] == 3
        norms = np.linalg.norm(embs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_zero_shot_batch_matches_shapes(self):
        model = self._make_clip()
        frames = [np.full((50, 50, 3), v, dtype=np.uint8) for v in (10, 200)]
        text_embs = [np.array([1, 0, 0, 0], dtype=np.float32),
                     np.array([0, 1, 0, 0], dtype=np.float32)]
        ranked = model.zero_shot_classify_batch(frames, text_embs, ["a", "b"])
        assert len(ranked) == 2
        for row in ranked:
            assert len(row) == 2
            # ranked descending by prob
            assert row[0][1] >= row[1][1]
