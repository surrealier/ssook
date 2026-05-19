"""Unit tests for the heuristic ONNX classifier."""
from core.model_classifier import _classify_from_io


def test_yolo_detection():
    out = _classify_from_io(
        inputs=[("images", [1, 3, 640, 640])],
        outputs=[("output0", [1, 84, 8400])],
        name_hint="yolo11n.onnx",
    )
    assert out["task_type"] == "detection"
    assert out["confidence"] >= 0.5


def test_classification_2d_logits():
    out = _classify_from_io(
        inputs=[("input", [1, 3, 224, 224])],
        outputs=[("logits", [1, 1000])],
        name_hint="resnet50.onnx",
    )
    assert out["task_type"] == "classification"


def test_clip_image_encoder_embedding_dim():
    out = _classify_from_io(
        inputs=[("pixel_values", [1, 3, 224, 224])],
        outputs=[("image_features", [1, 512])],
        name_hint="clip_image_encoder.onnx",
    )
    assert out["task_type"] == "vlm_image_encoder"


def test_clip_text_encoder_by_input_name():
    out = _classify_from_io(
        inputs=[("input_ids", [1, 77])],
        outputs=[("text_features", [1, 512])],
        name_hint="clip_text.onnx",
    )
    assert out["task_type"] == "vlm_text_encoder"


def test_segmentation_4d_output():
    out = _classify_from_io(
        inputs=[("input", [1, 3, 512, 512])],
        outputs=[("seg", [1, 21, 512, 512])],
        name_hint="deeplabv3.onnx",
    )
    assert out["task_type"] == "segmentation"


def test_instance_segmentation_multi_output():
    out = _classify_from_io(
        inputs=[("images", [1, 3, 640, 640])],
        outputs=[("output0", [1, 116, 8400]), ("proto", [1, 32, 160, 160])],
        name_hint="yolov8-seg.onnx",
    )
    assert out["task_type"] == "instance_segmentation"


def test_pose_56_attrs():
    out = _classify_from_io(
        inputs=[("images", [1, 3, 640, 640])],
        outputs=[("output0", [1, 56, 8400])],
        name_hint="yolo11n-pose.onnx",
    )
    assert out["task_type"] == "pose"


def test_unknown_when_input_not_image():
    out = _classify_from_io(
        inputs=[("audio", [1, 16000])],
        outputs=[("text", [1, 100])],
        name_hint="whisper.onnx",
    )
    assert out["task_type"] == "unknown"


def test_includes_metadata_fields():
    from core.model_classifier import classify
    r = classify("does_not_exist.onnx")
    assert r["task_type"] == "unknown"
    assert "reason" in r
