"""비디오 표시 위젯: 원본 비율 유지, 창 크기 연동"""
import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent, QWheelEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem

from core.app_config import AppConfig
from core.inference import DetectionResult


def _generate_palette(n: int) -> list:
    """HSV 균등 분포로 n개의 BGR 색상 생성"""
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        hsv = np.uint8([[[hue, 220, 220]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(x) for x in bgr))
    return colors


_PALETTE_CACHE: list = []


def get_palette_color(class_id: int, total: int) -> tuple:
    global _PALETTE_CACHE
    if len(_PALETTE_CACHE) < total:
        _PALETTE_CACHE = _generate_palette(total)
    if _PALETTE_CACHE:
        return _PALETTE_CACHE[class_id % len(_PALETTE_CACHE)]
    return (0, 255, 0)


class VideoWidget(QWidget):
    file_dropped = Signal(str)   # 드래그앤드롭된 파일 경로

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene, self)
        from PySide6.QtGui import QPainter
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._view.setBackgroundBrush(Qt.black)
        self._view.setStyleSheet("border: none;")
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self._last_frame: np.ndarray | None = None
        self._last_result: DetectionResult | None = None
        self._last_names: dict = {}
        self._total_classes: int = 80
        self._zoom_factor: float = 1.0

        self.setAcceptDrops(True)
        self._show_placeholder()

    def _show_placeholder(self):
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        self._set_pixmap_from_bgr(placeholder)

    def display_frame(self, frame_bgr: np.ndarray, result: DetectionResult,
                      config: AppConfig, names: dict):
        self._last_frame = frame_bgr
        self._last_result = result
        self._last_names = names
        self._total_classes = len(names)
        drawn = self._draw_detections(frame_bgr.copy(), result, config, names)
        self._set_pixmap_from_bgr(drawn)

    def _draw_detections(self, frame: np.ndarray, result: DetectionResult,
                         config: AppConfig, names: dict) -> np.ndarray:
        if result is None or len(result.boxes) == 0:
            return frame

        has_extra = result.extra_attrs is not None
        total = len(names)
        for i, (box, score, cid) in enumerate(zip(result.boxes, result.scores, result.class_ids)):
            cid = int(cid)
            style = config.get_class_style(cid)
            if not style.enabled:
                continue

            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            thickness = style.thickness if style.thickness else config.box_thickness

            # detect.py 기반: centernet은 이벤트별 색상 오버라이드
            extra_label = ""
            cls_name = names.get(cid, "").lower()
            if has_extra:
                color = style.color if style.color else get_palette_color(cid, total)
                ea = result.extra_attrs[i]
                fall, crawl, jump = ea[0], ea[1], ea[2]
                front, back, side = ea[3], ea[4], ea[5]
                mask = ea[6]
                if "person" in cls_name:
                    if fall > 0.5:
                        extra_label += f" fall {fall:.3f}"
                        color = (0, 0, 255)
                    if crawl > 0.5:
                        extra_label += f" crawl {crawl:.3f}"
                    if jump > 0.5:
                        extra_label += f" jump {jump:.3f}"
                        color = (255, 0, 255)
                    if back >= 0.95 and side < 0.05 and front < 0.05:
                        color = (0, 0, 255)
                    dominant = max((front, "F"), (back, "B"), (side, "S"), key=lambda x: x[0])
                    if dominant[0] > 0.5:
                        extra_label += f" {dominant[1]}{dominant[0]:.2f}"
                elif "face" in cls_name or "mask" in cls_name:
                    nomask = 1 - mask
                    extra_label += f" nomask {nomask:.3f}"
                    if nomask > 0.5:
                        color = (0, 0, 255)
                elif "wheelchair" in cls_name or "cane" in cls_name:
                    color = (0, 0, 255)
            else:
                color = style.color if style.color else get_palette_color(cid, total)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            if config.show_labels or config.show_confidence:
                label_parts = []
                if config.show_labels:
                    label_parts.append(names.get(cid, f"cls{cid}"))
                if config.show_confidence:
                    label_parts.append(f"{score:.2f}")
                label = " ".join(label_parts) + extra_label
                font_scale = config.label_size
                font_thick = max(1, thickness - 1)
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
                ty = max(y1 - 4, th + 4)
                cv2.rectangle(frame,
                              (x1, ty - th - baseline - 2),
                              (x1 + tw + 2, ty + 2),
                              color, -1)
                cv2.putText(frame, label,
                            (x1 + 1, ty - baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), font_thick, cv2.LINE_AA)
        return frame

    def _set_pixmap_from_bgr(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888).copy()
        pxm = QPixmap.fromImage(qimg)
        self._pixmap_item.setPixmap(pxm)
        self._scene.setSceneRect(0, 0, w, h)
        self._view.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._zoom_factor == 1.0:
            self._view.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta > 0:
            factor = 1.15
        else:
            factor = 1.0 / 1.15
        new_zoom = self._zoom_factor * factor
        new_zoom = max(0.1, min(new_zoom, 20.0))
        scale_change = new_zoom / self._zoom_factor
        self._zoom_factor = new_zoom
        self._view.scale(scale_change, scale_change)

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        self._zoom_factor = 1.0
        self._view.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def snapshot(self, out_dir: str) -> str:
        """현재 렌더링된 프레임을 스냅샷으로 저장"""
        import os
        from datetime import datetime
        if self._last_frame is None:
            return ""
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"snapshot_{ts}.jpg")
        # 박스가 그려진 최신 프레임 재생성
        frame = self._last_frame.copy()
        if self._last_result is not None:
            from core.app_config import AppConfig
            cfg = AppConfig()
            frame = self._draw_detections(frame, self._last_result, cfg, self._last_names)
        cv2.imwrite(path, frame)
        return path

    # --- 드래그앤드롭 ---
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.file_dropped.emit(path)
                break
