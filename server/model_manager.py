"""전역 모델 상태 관리 + 로딩 헬퍼."""
import os
import threading

from core.model_loader import load_model as _load_model_core

_loaded_model = None
_loaded_model_meta = None
_model_lock = threading.Lock()


def get_model():
    """현재 로드된 모델 참조 반환 (Lock 없이 읽기 — 참조 스냅샷)."""
    return _loaded_model


def get_model_meta():
    return _loaded_model_meta


def ensure_model(model_path: str, model_type: str = "yolo", cfg=None):
    """모델이 로드되지 않았거나 경로/타입이 다르면 로드. 로컬 참조 반환.

    Args:
        model_path: ONNX 모델 경로
        model_type: 모델 타입 (custom: 접두사 이미 제거된 상태)
        cfg: AppConfig 인스턴스 (custom 모델 class_names 적용용)
    Returns:
        ModelInfo 인스턴스
    """
    global _loaded_model, _loaded_model_meta
    custom_name = ""
    if model_type.startswith("custom:"):
        custom_name = model_type.split(":", 1)[1]
        model_type = "custom"

    with _model_lock:
        if (_loaded_model is not None
                and _loaded_model.path == model_path
                and _loaded_model.model_type == model_type):
            return _loaded_model

        info = _load_model_core(model_path, model_type=model_type)
        if custom_name:
            info.custom_type_name = custom_name
            if cfg:
                cmt = cfg.custom_model_types.get(custom_name)
                if cmt and cmt.class_names:
                    info.names = cmt.class_names

        inp = info.session.get_inputs()[0]
        out = info.session.get_outputs()[0]
        _loaded_model = info
        _loaded_model_meta = {
            "ok": True,
            "name": os.path.basename(model_path),
            "input_shape": str(inp.shape),
            "output_shape": str(out.shape),
            "input_size": list(info.input_size) if info.input_size else None,
            "num_classes": len(info.names) if info.names else 0,
            "names": info.names or {},
            "task": info.task_type or "",
            "layout": info.output_layout or "",
            "model_type": info.model_type or "",
            "batch_size": info.batch_size,
        }
        return _loaded_model


def load_fresh(model_path: str, model_type: str = "yolo", cfg=None):
    """항상 새로 로드 (전역 상태 갱신). 이전 세션은 명시적으로 해제."""
    global _loaded_model, _loaded_model_meta
    # Release previous session
    with _model_lock:
        if _loaded_model is not None and _loaded_model.session is not None:
            try:
                del _loaded_model.session
            except Exception:
                pass
            _loaded_model = None
            import gc; gc.collect()

    custom_name = ""
    if model_type.startswith("custom:"):
        custom_name = model_type.split(":", 1)[1]
        model_type = "custom"

    info = _load_model_core(model_path, model_type=model_type)
    if custom_name:
        info.custom_type_name = custom_name
        if cfg:
            cmt = cfg.custom_model_types.get(custom_name)
            if cmt and cmt.class_names:
                info.names = cmt.class_names

    inp = info.session.get_inputs()[0]
    out = info.session.get_outputs()[0]
    meta = {
        "ok": True,
        "name": os.path.basename(model_path),
        "input_shape": str(inp.shape),
        "output_shape": str(out.shape),
        "input_size": list(info.input_size) if info.input_size else None,
        "num_classes": len(info.names) if info.names else 0,
        "names": info.names or {},
        "task": info.task_type or "",
        "layout": info.output_layout or "",
        "model_type": info.model_type or "",
        "batch_size": info.batch_size,
    }
    with _model_lock:
        _loaded_model = info
        _loaded_model_meta = meta
    return meta
