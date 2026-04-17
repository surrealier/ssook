"""/api/config/* 라우터."""
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from core.app_config import AppConfig, ClassStyle, CustomModelType
from core.model_loader import MODEL_TYPES

router = APIRouter()
ROOT = Path(__file__).resolve().parent.parent


@router.get("/api/config")
async def get_config():
    try:
        cfg = AppConfig()
        cs = {}
        for cid, s in cfg.class_styles.items():
            cs[str(cid)] = {"enabled": s.enabled, "color": list(s.color) if s.color else None, "thickness": s.thickness}
        mt = dict(MODEL_TYPES)
        for name in cfg.custom_model_types:
            mt[f"custom:{name}"] = name
        return {
            "model_type": cfg.model_type, "conf_threshold": cfg.conf_threshold,
            "batch_size": cfg.batch_size, "box_thickness": cfg.box_thickness,
            "label_size": cfg.label_size, "show_labels": cfg.show_labels,
            "show_confidence": cfg.show_confidence, "default_model_path": cfg.default_model_path,
            "model_types": mt, "class_styles": cs,
            "samples_dir": str(ROOT / "assets" / "samples"),
        }
    except Exception as e:
        return {"error": str(e)}


class ConfigUpdate(BaseModel):
    conf_threshold: Optional[float] = None
    model_type: Optional[str] = None
    batch_size: Optional[int] = None
    box_thickness: Optional[int] = None
    label_size: Optional[float] = None
    show_labels: Optional[bool] = None
    show_confidence: Optional[bool] = None
    show_label_bg: Optional[bool] = None
    default_model_path: Optional[str] = None


@router.post("/api/config")
async def save_config(cfg: ConfigUpdate):
    try:
        app_cfg = AppConfig()
        for k, v in cfg.dict(exclude_none=True).items():
            setattr(app_cfg, k, v)
        app_cfg.save()
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}


class ClassStyleUpdate(BaseModel):
    class_id: int
    enabled: Optional[bool] = None
    color: Optional[list] = None
    thickness: Optional[int] = None


@router.post("/api/config/class-style")
async def save_class_style(req: ClassStyleUpdate):
    try:
        cfg = AppConfig()
        style = cfg.get_class_style(req.class_id)
        if req.enabled is not None:
            style.enabled = req.enabled
        if req.color is not None:
            style.color = tuple(req.color) if req.color else None
        if req.thickness is not None:
            style.thickness = req.thickness if req.thickness > 0 else None
        cfg.set_class_style(req.class_id, style)
        cfg.save()
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}


class CustomModelTypeRequest(BaseModel):
    name: str
    model_path: str
    output_index: int = 0
    attr_roles: list = []
    dim_roles: list = []
    has_objectness: bool = False
    nms: bool = True
    conf_threshold: float = 0.25
    class_names: Optional[dict] = None


@router.post("/api/config/custom-model-type")
async def save_custom_model_type(req: CustomModelTypeRequest):
    try:
        cfg = AppConfig()
        cmt = CustomModelType(
            name=req.name, output_index=req.output_index,
            dim_roles=req.dim_roles, attr_roles=req.attr_roles,
            has_objectness=req.has_objectness, nms=req.nms,
            conf_threshold=req.conf_threshold,
            class_names={int(k): v for k, v in req.class_names.items()} if req.class_names else None,
        )
        cfg.custom_model_types[req.name] = cmt
        cfg.save()
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}


@router.post("/api/config/custom-model-type/test")
async def test_custom_model_type(req: CustomModelTypeRequest):
    try:
        import onnxruntime as ort
        from core.inference import letterbox, preprocess, postprocess_custom
        session = ort.InferenceSession(req.model_path)
        inp = session.get_inputs()[0]
        h = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        w = int(inp.shape[3]) if isinstance(inp.shape[3], int) else 640
        dummy = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        _, ratio, pad = letterbox(dummy, (h, w))
        tensor = preprocess(dummy, (h, w))
        outputs = session.run(None, {inp.name: tensor})
        cmt = CustomModelType(
            name=req.name, output_index=req.output_index,
            dim_roles=req.dim_roles, attr_roles=req.attr_roles,
            has_objectness=req.has_objectness, nms=req.nms,
            conf_threshold=req.conf_threshold,
        )
        result = postprocess_custom(outputs, cmt, req.conf_threshold, ratio, pad, dummy.shape)
        shapes = [str(list(o.shape)) for o in outputs]
        return {"ok": True, "detections": len(result.boxes), "output_shapes": shapes}
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/config/custom-model-types")
async def list_custom_model_types():
    cfg = AppConfig()
    return {name: {"attr_roles": cmt.attr_roles, "dim_roles": cmt.dim_roles, "class_names": cmt.class_names}
            for name, cmt in cfg.custom_model_types.items()}
