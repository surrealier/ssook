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
            self.videos_dir: str = cfg.get("videos_dir", _DEFAULT_VIDEOS)
            self.models_dir: str = cfg.get("models_dir", _DEFAULT_MODELS)
            self.model_type: str = cfg.get("model_type", "yolo")
            self.batch_size: int = cfg.get("batch_size", 1)

            raw = cfg.get("class_styles", {})
            self.class_styles: dict[int, ClassStyle] = {}
            for k, v in raw.items():
                color = tuple(v["color"]) if v.get("color") else None
                self.class_styles[int(k)] = ClassStyle(
                    enabled=v.get("enabled", True),
                    color=color,
                    thickness=v.get("thickness"),
                )

    def save(self):
        with self._rw_lock:
            cs_save = {}
            for cls_id, s in self.class_styles.items():
                cs_save[cls_id] = {
                    "enabled": s.enabled,
                    "color": list(s.color) if s.color else None,
                    "thickness": s.thickness,
                }
            cfg = {
                "conf_threshold": self.conf_threshold,
                "box_thickness": self.box_thickness,
                "label_size": self.label_size,
                "show_labels": self.show_labels,
                "show_confidence": self.show_confidence,
                "videos_dir": self.videos_dir,
                "models_dir": self.models_dir,
                "model_type": self.model_type,
                "batch_size": self.batch_size,
                "class_styles": cs_save,
            }
        os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True)

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
