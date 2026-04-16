"""PT / ONNX 모델 로드 및 names 추출"""
import ast
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import onnxruntime as ort

_DARKNET_DEFAULT_NAMES = {
    0: "person", 1: "face", 2: "red-sign", 3: "wheelchair", 4: "cane",
}


@dataclass
class ModelInfo:
    path: str
    format: Literal["onnx", "pt"]
    names: dict                          # {0: 'person', 1: 'car', ...}
    input_size: tuple                    # (H, W)
    session: Optional[ort.InferenceSession]
    output_layout: Literal["v8", "v5"]   # v8: (1,4+N,8400) / v5: (1,25200,5+N)
    input_name: str = ""
    model_type: str = "yolo"             # "yolo" | "darknet" | "detr" | ...
    task_type: str = "detection"         # "detection" | "classification"
    batch_size: int = 1                  # 고정 배치 크기 (1=단일, 4=4배치 등)
    custom_type_name: str = ""           # custom 모델 타입 이름 (model_type=="custom" 시)
    _batch_buf: Any = field(default=None, repr=False)  # 배치 텐서 캐시 (inference.py에서 사용)


# 지원 모델 타입 목록
MODEL_TYPES = {
    # Detection (9)
    "yolo":      "YOLO (v5/v7/v8/v9/v11/v12)",
    "darknet":   "CenterNet (Darknet)",
    "detr":      "DETR / RT-DETR / RF-DETR",
    "yolo_nas":  "YOLO-NAS",
    "yolov10":   "YOLOv10 (NMS-free)",
    "damo_yolo": "DAMO-YOLO",
    "gold_yolo": "Gold-YOLO",
    "yolox":     "YOLOX",
    "efficientdet": "EfficientDet",
    # Classification (5)
    "cls_resnet":      "ResNet (Classification)",
    "cls_efficientnet":"EfficientNet (Classification)",
    "cls_mobilenet":   "MobileNet (Classification)",
    "cls_vit":         "ViT (Classification)",
    "cls_custom":      "Custom Classification",
    # Segmentation (5)
    "seg_yolo":    "YOLO-Seg (v8/v11)",
    "seg_unet":    "U-Net",
    "seg_deeplabv3":"DeepLabV3",
    "seg_fcn":     "FCN",
    "seg_custom":  "Custom Segmentation",
    # CLIP (5)
    "clip_vit_b32": "CLIP ViT-B/32",
    "clip_vit_b16": "CLIP ViT-B/16",
    "clip_vit_l14": "CLIP ViT-L/14",
    "clip_rn50":    "CLIP ResNet-50",
    "clip_custom":  "Custom CLIP",
    # Embedder (5)
    "emb_vit":         "ViT Embedder",
    "emb_resnet":      "ResNet Embedder",
    "emb_efficientnet":"EfficientNet Embedder",
    "emb_dino":        "DINOv2 Embedder",
    "emb_custom":      "Custom Embedder",
    # Pose Estimation (3)
    "pose_yolo":       "YOLO-Pose (v8/v11)",
    "pose_hrnet":      "HRNet Pose",
    "pose_vitpose":    "ViTPose",
    # Instance Segmentation (3)
    "instseg_yolo":    "YOLO-Seg Instance (v8/v11)",
    "instseg_maskrcnn":"Mask R-CNN",
    "instseg_custom":  "Custom Instance Seg",
    # Tracking — now integrated as Viewer option (ByteTrack / SORT)
    # VLM (3)
    "vlm_vqa":         "VLM — VQA",
    "vlm_caption":     "VLM — Captioning",
    "vlm_grounding":   "VLM — Grounding DINO",
}


def _build_providers() -> list:
    """사용 가능한 ONNX Runtime provider 선택 (GPU 우선)"""
    available = ort.get_available_providers()
    for prov in [
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
        "CoreMLExecutionProvider",
        "OpenVINOExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]:
        if prov in available:
            if prov == "CPUExecutionProvider":
                return ["CPUExecutionProvider"]
            if prov == "OpenVINOExecutionProvider":
                return [
                    ("OpenVINOExecutionProvider", {"device_type": "GPU"}),
                    ("OpenVINOExecutionProvider", {"device_type": "CPU"}),
                    "CPUExecutionProvider",
                ]
            return [prov, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _create_session(path: str, session_options=None) -> ort.InferenceSession:
    providers = _build_providers()
    if session_options is None:
        import os
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_cpu_mem_arena = True
        # DML requires mem_pattern disabled
        is_dml = any("Dml" in str(p) for p in providers)
        session_options.enable_mem_pattern = not is_dml
        # 노트북 환경 최적화: 물리 코어 수 기반 스레드 제한
        phys_cores = os.cpu_count() or 4
        intra = max(2, phys_cores // 2)
        session_options.intra_op_num_threads = intra
        session_options.inter_op_num_threads = max(1, intra // 2)
    try:
        return ort.InferenceSession(path, sess_options=session_options, providers=providers)
    except Exception as e:
        if providers != ["CPUExecutionProvider"]:
            print(f"[ModelLoader] {providers[0]} failed ({e}), falling back to CPU")
            return ort.InferenceSession(path, sess_options=session_options, providers=["CPUExecutionProvider"])
        raise


def _get_names_from_onnx(session: ort.InferenceSession) -> dict:
    meta = session.get_modelmeta().custom_metadata_map
    for key in ("names", "classes"):
        if key in meta:
            try:
                return ast.literal_eval(meta[key])
            except Exception:
                pass
    # fallback: 출력 채널 수로 추정
    out_shape = session.get_outputs()[0].shape
    n = _guess_num_classes(out_shape)
    return {i: f"class_{i}" for i in range(n)}


def _get_names_from_pt(path: str) -> dict:
    """torch.load로 YOLO PT의 names 추출 (추론 없이)"""
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        names = ckpt.get("names")
        if names is None and hasattr(ckpt.get("model"), "names"):
            names = ckpt["model"].names
        if isinstance(names, (list, tuple)):
            names = {i: n for i, n in enumerate(names)}
        if names:
            return names
    except Exception as e:
        print(f"[ModelLoader] Failed to extract PT names: {e}")
    return {}


def _detect_task_type(session: ort.InferenceSession, model_type: str = "yolo") -> str:
    """출력 텐서 shape으로 detection/classification/segmentation 자동 감지"""
    # 모델 타입 접두사로 강제 지정
    if model_type.startswith("cls_"):
        return "classification"
    if model_type.startswith("seg_"):
        return "segmentation"
    if model_type.startswith("clip_"):
        return "embedding"
    if model_type.startswith("emb_"):
        return "embedding"
    if model_type.startswith("pose_"):
        return "pose"
    if model_type.startswith("instseg_"):
        return "instance_segmentation"
    if model_type.startswith("track_"):
        return "detection"
    if model_type.startswith("vlm_"):
        return "vlm"
    shape = session.get_outputs()[0].shape
    # Classification: (1, N) — 2차원, N은 클래스 수
    if len(shape) == 2:
        return "classification"
    # Segmentation: (1, C, H, W) — 4차원, C>1
    if len(shape) == 4:
        return "segmentation"
    # Detection: (1, X, Y) — 3차원
    return "detection"


def _detect_layout(session: ort.InferenceSession, model_type: str = "yolo") -> Literal["v8", "v5"]:
    """출력 텐서 shape으로 YOLO 버전 자동 감지"""
    if model_type in ("detr", "yolo_nas", "yolov10", "custom"):
        return "v8"  # DETR 계열은 별도 postprocess 사용
    shape = session.get_outputs()[0].shape
    # shape: (1, dim1, dim2)
    if len(shape) == 3:
        _, d1, d2 = shape
        # v8: (1, 4+N, 8400) → d1 < d2
        # v5: (1, 25200, 5+N) → d1 > d2
        if isinstance(d1, int) and isinstance(d2, int):
            return "v8" if d1 < d2 else "v5"
    return "v8"


def _guess_num_classes(shape) -> int:
    dims = [d for d in shape if isinstance(d, int) and d > 0]
    # (1, 4+N, 8400) 또는 (1, 25200, 5+N)
    if len(dims) >= 2:
        small = min(dims[1:])
        return max(small - 5, 1)
    return 80


def _get_input_size(session: ort.InferenceSession) -> tuple:
    inp = session.get_inputs()[0]
    shape = inp.shape  # [1, 3, H, W]
    try:
        h = int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else 640
        w = int(shape[3]) if isinstance(shape[3], int) and shape[3] > 0 else 640
        return (h, w)
    except Exception:
        return (640, 640)


def load_model(path: str, model_type: str = "yolo", pt_convert_callback=None,
               session_options=None) -> ModelInfo:
    """
    path: .onnx 또는 .pt 파일 경로
    pt_convert_callback: PT 파일 시 ONNX 변환 여부 묻는 콜백 함수.
                         콜백이 ONNX 경로를 반환하면 그것으로 로드.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".onnx":
        return _load_onnx(path, model_type, session_options)

    elif ext == ".pt":
        names = _get_names_from_pt(path)
        onnx_path = None

        if pt_convert_callback is not None:
            onnx_path = pt_convert_callback(path, names)

        if onnx_path and os.path.isfile(onnx_path):
            info = _load_onnx(onnx_path, model_type, session_options)
            # PT에서 추출한 names가 더 정확할 수 있으므로 병합
            if names:
                info.names = names
            return info
        else:
            # ONNX 변환 없이 PT names만 담은 더미 ModelInfo 반환
            return ModelInfo(
                path=path,
                format="pt",
                names=names or {i: f"class_{i}" for i in range(80)},
                input_size=(640, 640),
                session=None,
                output_layout="v8",
                model_type=model_type,
                task_type="detection",
            )

    else:
        raise ValueError(f"지원하지 않는 모델 형식: {ext}")


def _load_onnx(path: str, model_type: str = "yolo", session_options=None) -> ModelInfo:
    session = _create_session(path, session_options)
    names = _get_names_from_onnx(session)
    if model_type == "darknet" and all(v.startswith("class_") for v in names.values()):
        names = _DARKNET_DEFAULT_NAMES

    inputs = session.get_inputs()

    # CLIP/Embedder: 다중 입력 모델 — 이미지 입력(4D) 찾기
    img_input = inputs[0]
    if model_type.startswith(("clip_", "emb_")):
        for inp in inputs:
            if len(inp.shape) == 4:  # (B, C, H, W) or (B, H, W, C)
                img_input = inp
                break
            if "pixel" in inp.name.lower() or "image" in inp.name.lower():
                img_input = inp
                break

    input_name = img_input.name
    # input_size from the image input
    shape = img_input.shape
    try:
        h = int(shape[2]) if len(shape) >= 4 and isinstance(shape[2], int) and shape[2] > 0 else 640
        w = int(shape[3]) if len(shape) >= 4 and isinstance(shape[3], int) and shape[3] > 0 else 640
        input_size = (h, w)
    except Exception:
        input_size = (640, 640)

    if not model_type.startswith(("clip_", "emb_")):
        input_size = _get_input_size(session)

    task_type = _detect_task_type(session, model_type)
    layout = _detect_layout(session, model_type) if task_type == "detection" else "v8"
    # 배치 크기 감지
    batch_dim = img_input.shape[0]
    batch_size = int(batch_dim) if isinstance(batch_dim, int) and batch_dim > 0 else 1
    print(f"[ModelLoader] ONNX loaded: {os.path.basename(path)}")
    print(f"  input={input_size}, input_name={input_name}, task={task_type}, layout={layout}, type={model_type}, batch={batch_size}, classes={len(names)}")
    return ModelInfo(
        path=path,
        format="onnx",
        names=names,
        input_size=input_size,
        session=session,
        output_layout=layout,
        input_name=input_name,
        model_type=model_type,
        task_type=task_type,
        batch_size=batch_size,
    )
