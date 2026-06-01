"""앱 설정 싱글톤 (thread-safe, YAML 영속화)"""
import os
import sys
import threading
from dataclasses import dataclass
from typing import Optional, Tuple
import yaml

_CONFIG_PATH = "settings/app_config.yaml"

_FROZEN = getattr(sys, "frozen", False)
_DEFAULT_VIDEOS = os.path.join("..", "..", "Videos") if _FROZEN else "Videos"
_DEFAULT_MODELS = os.path.join("..", "..", "Models") if _FROZEN else "Models"


@dataclass
class ClassStyle:
    enabled: bool = True
    color: Optional[Tuple[int, int, int]] = None   # BGR; None → 자동 팔레트
    thickness: Optional[int] = None                 # None → 전역 기본값


@dataclass
class CustomModelType:
    """사용자 정의 모델 타입: output shape → 의미 매핑"""
    name: str
    output_index: int = 0                           # 사용할 출력 텐서 인덱스
    dim_roles: Optional[list] = None                # 각 차원의 역할 리스트
    # 예: ["batch", "detections", "attrs"]
    attr_roles: Optional[list] = None               # attrs 차원 내 각 슬롯의 역할
    # 예: ["x_center", "y_center", "width", "height", "conf_class0", ...]
    has_objectness: bool = False
    nms: bool = True
    conf_threshold: float = 0.25
    class_names: Optional[dict] = None              # {0: "person", 1: "car", ...}


class AppConfig:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._rw_lock = threading.Lock()
                    cls._instance._load()
        return cls._instance

    def _load(self):
        if os.path.exists(_CONFIG_PATH):
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}

        with self._rw_lock:
            self.conf_threshold: float = cfg.get("conf_threshold", 0.25)
            self.box_thickness: int = cfg.get("box_thickness", 2)
            self.label_size: float = cfg.get("label_size", 0.55)
            self.show_labels: bool = cfg.get("show_labels", True)
            self.show_confidence: bool = cfg.get("show_confidence", True)
            self.show_label_bg: bool = cfg.get("show_label_bg", True)
            self.videos_dir: str = cfg.get("videos_dir", _DEFAULT_VIDEOS)
            self.models_dir: str = cfg.get("models_dir", _DEFAULT_MODELS)
            self.model_type: str = cfg.get("model_type", "yolo")
            self.batch_size: int = cfg.get("batch_size", 1)
            self.default_model_path: str = cfg.get("default_model_path", "")
            # MJPEG 뷰어 스트림 JPEG 인코딩 품질 (viewer가 읽음).
            self.stream_jpeg_quality: int = cfg.get("stream_jpeg_quality", 75)

            raw = cfg.get("class_styles", {})
            self.class_styles: dict[int, ClassStyle] = {}
            for k, v in raw.items():
                color = tuple(v["color"]) if v.get("color") else None
                self.class_styles[int(k)] = ClassStyle(
                    enabled=v.get("enabled", True),
                    color=color,
                    thickness=v.get("thickness"),
                )

            # 사용자 정의 모델 타입
            self.custom_model_types: dict[str, CustomModelType] = {}
            for name, d in cfg.get("custom_model_types", {}).items():
                raw_cn = d.get("class_names")
                cn = {int(k): v for k, v in raw_cn.items()} if raw_cn else None
                self.custom_model_types[name] = CustomModelType(
                    name=name,
                    output_index=d.get("output_index", 0),
                    dim_roles=d.get("dim_roles"),
                    attr_roles=d.get("attr_roles"),
                    has_objectness=d.get("has_objectness", False),
                    nms=d.get("nms", True),
                    conf_threshold=d.get("conf_threshold", 0.25),
                    class_names=cn,
                )

    def save(self):
        # Hold the lock for the entire serialize+write so two concurrent
        # save() calls cannot interleave a partial YAML on disk.
        with self._rw_lock:
            cs_save = {}
            for cls_id, s in self.class_styles.items():
                cs_save[cls_id] = {
                    "enabled": s.enabled,
                    "color": list(s.color) if s.color else None,
                    "thickness": s.thickness,
                }
            cmt_save = {}
            for name, cmt in self.custom_model_types.items():
                d = {
                    "output_index": cmt.output_index,
                    "dim_roles": cmt.dim_roles,
                    "attr_roles": cmt.attr_roles,
                    "has_objectness": cmt.has_objectness,
                    "nms": cmt.nms,
                    "conf_threshold": cmt.conf_threshold,
                }
                if cmt.class_names:
                    d["class_names"] = cmt.class_names
                cmt_save[name] = d
            cfg = {
                "conf_threshold": self.conf_threshold,
                "box_thickness": self.box_thickness,
                "label_size": self.label_size,
                "show_labels": self.show_labels,
                "show_confidence": self.show_confidence,
                "show_label_bg": self.show_label_bg,
                "videos_dir": self.videos_dir,
                "models_dir": self.models_dir,
                "model_type": self.model_type,
                "batch_size": self.batch_size,
                "default_model_path": self.default_model_path,
                "stream_jpeg_quality": self.stream_jpeg_quality,
                "class_styles": cs_save,
                "custom_model_types": cmt_save,
            }
            os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
            # Atomic-ish write: write to .tmp then replace, so an interrupted
            # save never leaves a half-truncated config file behind.
            tmp = _CONFIG_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, allow_unicode=True)
            os.replace(tmp, _CONFIG_PATH)

    def get_class_style(self, class_id: int) -> ClassStyle:
        return self.class_styles.get(class_id, ClassStyle())

    def set_class_style(self, class_id: int, style: ClassStyle):
        with self._rw_lock:
            self.class_styles[class_id] = style

    def init_class_styles(self, names: dict):
        """모델 로드 시 새 클래스에 대해 기본 스타일 초기화"""
        with self._rw_lock:
            for cls_id in names:
                if cls_id not in self.class_styles:
                    self.class_styles[cls_id] = ClassStyle()
