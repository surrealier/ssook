"""/api/config/* 라우터."""
import re
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from core.app_config import AppConfig, ClassStyle, CustomModelType
from core.model_loader import MODEL_TYPES
from server.path_safety import UnsafePathError, safe_model_file

router = APIRouter()
ROOT = Path(__file__).resolve().parent.parent

# Custom-type name becomes a `custom:<name>` key and a YAML mapping key, so
# reject path separators / control chars that could break replay or traversal.
_CUSTOM_NAME_RE = re.compile(r"^[\w.\- ]+$")

# Valid coordinate groups recognized by core.inference.postprocess_custom.
# A custom type with none of these silently yields zero detections, so we
# reject registration up front (mirror the coord branches at inference.py:603+).
_COORD_GROUPS: tuple[frozenset, ...] = (
    frozenset({"x1", "y1", "x2", "y2"}),
    frozenset({"x1", "y1", "width", "height"}),
    frozenset({"x_center", "y_center", "width", "height"}),
)


def _validate_class_names(raw: Optional[dict]) -> Optional[dict]:
    """Validate + coerce free-text class_names into {int: str}.

    Keys originate from a user CSV, so a non-digit key would make int(k)
    raise deep inside save(); validate here and raise a typed error instead.
    """
    if not raw:
        return None
    coerced: dict[int, str] = {}
    for k, v in raw.items():
        key = str(k).strip()
        if not (key.lstrip("-").isdigit()):
            raise UnsafePathError(
                "BAD_CLASS_NAMES",
                f"class_names keys must be integer ids, got '{k}'",
            )
        coerced[int(key)] = v
    return coerced


def _validate_roles(attr_roles: list) -> None:
    """Ensure attr_roles carry a usable coord group + a conf source.

    Without a recognizable coord group or any conf role, postprocess_custom
    returns DetectionResult.empty() with no signal — better to reject at
    registration than to silently save a type that never detects anything.
    """
    roles = set(attr_roles or [])
    if not any(group <= roles for group in _COORD_GROUPS):
        raise UnsafePathError(
            "MISSING_COORDS",
            "attr_roles must include a coord group: x1y1x2y2, x1y1+width/height, "
            "or x_center/y_center/width/height",
        )
    has_conf = (
        ("confidence" in roles and "class_id" in roles)
        or any(r.startswith("conf_class") for r in roles)
    )
    if not has_conf:
        raise UnsafePathError(
            "MISSING_CONF",
            "attr_roles must include a conf source: one or more conf_class* "
            "roles, or both 'confidence' and 'class_id'",
        )


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
        # Mutate scalar fields under the same RLock save()/_load() use, so a
        # concurrent save/reload cannot serialize a torn value.
        with app_cfg._rw_lock:
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


class ClassStyleBatchItem(BaseModel):
    class_id: int
    enabled: Optional[bool] = None
    color: Optional[list] = None
    thickness: Optional[int] = None


class ClassStyleBatchUpdate(BaseModel):
    styles: list[ClassStyleBatchItem]


def _validate_color(color: Optional[list]) -> None:
    """Reject malformed BGR triples before they reach the YAML write.

    A non-3-tuple or out-of-range component would persist a style that the
    renderer can't interpret; fail fast at the boundary instead. An empty
    list is allowed: it mirrors the single-style route where a falsy color
    clears the override back to the auto palette (color → None).
    """
    if not color:  # None or [] → "use auto palette", not a malformed triple
        return
    if len(color) != 3 or not all(isinstance(c, int) and 0 <= c <= 255 for c in color):
        raise UnsafePathError(
            "BAD_COLOR",
            "color must be a list of three integers in 0..255 (BGR), or empty to clear",
        )


def _validate_thickness(thickness: Optional[int]) -> None:
    """Bound box thickness so a runaway value can't be serialized."""
    if thickness is None:
        return
    if not isinstance(thickness, int) or thickness < 0 or thickness > 50:
        raise UnsafePathError(
            "BAD_THICKNESS",
            "thickness must be an integer in 0..50",
        )


@router.post("/api/config/class-styles")
async def save_class_styles(req: ClassStyleBatchUpdate):
    """Apply a LIST of class styles under one lock + one YAML write.

    "Select All" on the settings table previously fired one POST per class
    (N × ~100ms). This batches every change into a single AppConfig lock
    acquisition and a single atomic save(), so toggling many classes costs
    one round trip and one disk write. Mirrors the single-style semantics:
    falsy color → None (auto palette), thickness ≤ 0 → None (global default).
    """
    try:
        # Validate the whole batch first so a single bad item rejects the
        # request before any in-memory mutation (all-or-nothing).
        for item in req.styles:
            _validate_color(item.color)
            _validate_thickness(item.thickness)

        cfg = AppConfig()
        # Mutate every style directly under the SAME lock save()/_load() use.
        # _rw_lock is a plain (non-reentrant) Lock, so we must NOT call
        # set_class_style() here (it re-acquires the lock → deadlock); update
        # the dict in place instead, then save() once after releasing.
        with cfg._rw_lock:
            for item in req.styles:
                style = cfg.class_styles.get(item.class_id)
                if style is None:
                    style = ClassStyle()
                    cfg.class_styles[item.class_id] = style
                if item.enabled is not None:
                    style.enabled = item.enabled
                if item.color is not None:
                    style.color = tuple(item.color) if item.color else None
                if item.thickness is not None:
                    style.thickness = item.thickness if item.thickness > 0 else None
        cfg.save()
        return {"ok": True, "updated": len(req.styles)}
    except UnsafePathError as e:
        return {"error": e.code}
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
        if not _CUSTOM_NAME_RE.match(req.name or ""):
            return {"error": "INVALID_NAME"}
        # Validate the model path before persisting — the path is later
        # replayed by other tabs, so a bad path must be rejected at the source.
        model_path = safe_model_file(req.model_path)
        class_names = _validate_class_names(req.class_names)
        _validate_roles(req.attr_roles)
        cfg = AppConfig()
        cmt = CustomModelType(
            name=req.name, output_index=req.output_index,
            dim_roles=req.dim_roles, attr_roles=req.attr_roles,
            has_objectness=req.has_objectness, nms=req.nms,
            conf_threshold=req.conf_threshold,
            class_names=class_names,
        )
        # Insert under the RLock save()/_load() use so a concurrent save can't
        # serialize a half-updated dict.
        with cfg._rw_lock:
            cfg.custom_model_types[req.name] = cmt
        cfg.save()
        return {"ok": True}
    except UnsafePathError as e:
        return {"error": e.code}
    except Exception as e:
        return {"error": str(e)}


@router.post("/api/config/custom-model-type/test")
async def test_custom_model_type(req: CustomModelTypeRequest):
    try:
        # Validate before opening a session — never let an unvalidated path
        # reach ort.InferenceSession (traversal / wrong-ext / missing file).
        model_path = safe_model_file(req.model_path)
        import onnxruntime as ort
        from core.inference import letterbox, preprocess, postprocess_custom
        session = ort.InferenceSession(model_path)
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
    except UnsafePathError as e:
        return {"error": e.code}
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/config/custom-model-types")
async def list_custom_model_types():
    cfg = AppConfig()
    return {name: {"attr_roles": cmt.attr_roles, "dim_roles": cmt.dim_roles, "class_names": cmt.class_names}
            for name, cmt in cfg.custom_model_types.items()}
