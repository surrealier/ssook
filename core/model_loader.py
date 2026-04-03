"""PT / ONNX 모델 로드 및 names 추출"""
import ast
import os
from dataclasses import dataclass
from typing import Literal, Optional

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
    model_type: str = "yolo"             # "yolo" | "darknet"
    task_type: str = "detection"         # "detection" | "classification"
    batch_size: int = 1                  # 고정 배치 크기 (1=단일, 4=4배치 등)


def _build_providers() -> list:
    """사용 가능한 ONNX Runtime provider 선택 (GPU 우선)"""
    available = ort.get_available_providers()
    for prov in [
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
        "OpenVINOExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]:
        if prov in available:
            if prov == "CPUExecutionProvider":
                return ["CPUExecutionProvider"]
            return [prov, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _create_session(path: str, session_options=None) -> ort.InferenceSession:
    if session_options is None:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=session_options, providers=_build_providers())


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
        print(f"[ModelLoader] PT names 추출 실패: {e}")
    return {}


def _detect_task_type(session: ort.InferenceSession) -> str:
    """출력 텐서 shape으로 detection/classification 자동 감지"""
    shape = session.get_outputs()[0].shape
    # Classification: (1, N) — 2차원, N은 클래스 수
    if len(shape) == 2:
        return "classification"
    # Detection: (1, X, Y) — 3차원
    return "detection"


def _detect_layout(session: ort.InferenceSession) -> Literal["v8", "v5"]:
    """출력 텐서 shape으로 YOLO 버전 자동 감지"""
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
    input_size = _get_input_size(session)
    task_type = _detect_task_type(session)
    layout = _detect_layout(session) if task_type == "detection" else "v8"
    input_name = session.get_inputs()[0].name
    # 배치 크기 감지
    batch_dim = session.get_inputs()[0].shape[0]
    batch_size = int(batch_dim) if isinstance(batch_dim, int) and batch_dim > 0 else 1
    print(f"[ModelLoader] ONNX 로드 완료: {os.path.basename(path)}")
    print(f"  입력 크기: {input_size}, 태스크: {task_type}, 레이아웃: {layout}, 타입: {model_type}, 배치: {batch_size}, 클래스 수: {len(names)}")
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
