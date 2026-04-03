"""분석 탭: 병목 분석 + 추론 결과 분석"""
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
    QLabel, QPushButton, QProgressBar, QLineEdit, QComboBox,
    QGroupBox, QFormLayout, QSlider, QTableWidget, QTableWidgetItem,
    QScrollArea, QFileDialog, QMessageBox, QSpinBox,
    QCheckBox, QHeaderView,
)

from core.app_config import AppConfig
from core.bottleneck_analyzer import (
    BottleneckAnalyzer, BottleneckReport, _BOTTLENECK_LABELS, _ONNX_BOTTLENECK_OPS,
)
from core.inference import (
    DetectionResult, letterbox, preprocess_darknet,
    postprocess_darknet, postprocess_v5, postprocess_v8,
)
from core.model_loader import ModelInfo, load_model

# ── Matplotlib 옵셔널 임포트 ─────────────────────────────────────────────────
Figure: Any = None
FigureCanvasQTAgg: Any = None
try:
    from matplotlib.figure import Figure  # type: ignore[assignment]
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # type: ignore[assignment]
    _MPL = True
except ImportError:
    _MPL = False

# ── Matplotlib 한국어 폰트 설정 ───────────────────────────────────────────────
if _MPL:
    import matplotlib as _mpl_mod
    import matplotlib.font_manager as _fm
    import os as _os
    _CJK_CANDIDATES = [
        "Malgun Gothic",
        "NanumGothic",
        "Apple SD Gothic Neo",
        "Noto Sans CJK KR",
        "UnDotum",
    ]
    _win_fonts = r"C:\Windows\Fonts"
    if _os.path.isdir(_win_fonts):
        for _ttf in ("malgun.ttf", "malgunbd.ttf"):
            _fp = _os.path.join(_win_fonts, _ttf)
            if _os.path.isfile(_fp):
                try:
                    _fm.fontManager.addfont(_fp)
                except Exception:
                    pass
    _loaded_names = {f.name for f in _fm.fontManager.ttflist}
    for _cjk in _CJK_CANDIDATES:
        if _cjk in _loaded_names:
            _mpl_mod.rcParams["font.family"] = _cjk
            break
    del _mpl_mod, _fm, _os, _CJK_CANDIDATES, _win_fonts, _loaded_names


# ── 공통 유틸리티 ────────────────────────────────────────────────────────────

def _bgr_to_pixmap(img_bgr: np.ndarray, max_w: int = 600, max_h: int = 400) -> QPixmap:
    """OpenCV BGR 이미지를 QPixmap으로 변환 (비율 유지 축소)"""
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img_bgr = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h2, w2 = rgb.shape[:2]
    qimg = QImage(rgb.data, w2, h2, w2 * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _bgr_to_full_pixmap(img_bgr: np.ndarray) -> QPixmap:
    """OpenCV BGR 이미지를 원본 크기 QPixmap으로 변환"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _class_color(class_id: int) -> "tuple[int, int, int]":
    """클래스 ID → BGR 색상 (HSV 균등 분배)"""
    h = (class_id * 37) % 180
    bgr = cv2.cvtColor(np.array([[[h, 220, 200]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_detections(
    frame: np.ndarray,
    result: DetectionResult,
    conf_thr: float,
    names: dict,
) -> np.ndarray:
    """프레임에 탐지 박스 + 레이블 오버레이"""
    vis = frame.copy()
    has_extra = result.extra_attrs is not None
    for i, (box, score, cid) in enumerate(zip(result.boxes, result.scores, result.class_ids)):
        if float(score) < conf_thr:
            continue
        cid = int(cid)
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cls_name = names.get(cid, str(cid)).lower()

        extra_label = ""
        if has_extra:
            color = _class_color(cid)
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
                # front/back/side 중 가장 높은 것만 표시
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
            color = _class_color(cid)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(cid, str(cid))} {float(score):.2f}{extra_label}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(y1 - 4, th + 4)
        cv2.rectangle(vis, (x1, ly - th - 4), (x1 + tw + 2, ly), color, -1)
        cv2.putText(vis, label, (x1 + 1, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def _draw_letterbox_overlay(img_bgr: np.ndarray, pad_w: float, pad_h: float) -> np.ndarray:
    """Letterbox 패딩 영역을 반투명 파란색으로 강조"""
    vis = img_bgr.copy()
    h, w = vis.shape[:2]
    left   = int(round(pad_w - 0.1))
    top    = int(round(pad_h - 0.1))
    right  = w - int(round(pad_w + 0.1))
    bottom = h - int(round(pad_h + 0.1))

    overlay = vis.copy()
    fill = (200, 100, 50)  # BGR: 파란 계열
    if top > 0:
        cv2.rectangle(overlay, (0, 0), (w, top), fill, -1)
    if bottom < h:
        cv2.rectangle(overlay, (0, bottom), (w, h), fill, -1)
    if left > 0:
        cv2.rectangle(overlay, (0, top), (left, bottom), fill, -1)
    if right < w:
        cv2.rectangle(overlay, (right, top), (w, bottom), fill, -1)
    return cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)


class ClickableImageLabel(QLabel):
    """클릭 시 원본 크기 팝업을 띄우는 이미지 라벨"""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._full_pixmap: "QPixmap | None" = None
        self.setCursor(Qt.PointingHandCursor)

    def setFullPixmap(self, pixmap: QPixmap, scaled: QPixmap):
        """원본 pixmap 저장 + 축소 pixmap 표시"""
        self._full_pixmap = pixmap
        self.setPixmap(scaled)

    def mouseDoubleClickEvent(self, event):
        if self._full_pixmap:
            from PySide6.QtWidgets import QDialog
            dlg = QDialog(self)
            dlg.setWindowTitle("이미지 확대")
            dlg.resize(min(self._full_pixmap.width() + 20, 1600),
                       min(self._full_pixmap.height() + 20, 900))
            lay = QVBoxLayout(dlg)
            lay.setContentsMargins(0, 0, 0, 0)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            lbl = QLabel()
            lbl.setPixmap(self._full_pixmap)
            lbl.setAlignment(Qt.AlignCenter)
            scroll.setWidget(lbl)
            lay.addWidget(scroll)
            dlg.exec()


def _draw_detections_diff(
    frame: np.ndarray,
    orig_result: DetectionResult,
    new_result: DetectionResult,
    conf_thr: float,
    names: dict,
) -> np.ndarray:
    """전처리 후 추가된 탐지는 빨간 점선 박스로 표시, 기존 탐지는 원래 색상 유지"""
    vis = frame.copy()
    orig_filtered = []
    if len(orig_result.boxes) > 0:
        for box, score, cid in zip(orig_result.boxes, orig_result.scores, orig_result.class_ids):
            if float(score) >= conf_thr:
                orig_filtered.append((box, int(cid)))

    def _iou(a, b):
        x1 = max(float(a[0]), float(b[0])); y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2])); y2 = min(float(a[3]), float(b[3]))
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        aa = (float(a[2])-float(a[0])) * (float(a[3])-float(a[1]))
        ab = (float(b[2])-float(b[0])) * (float(b[3])-float(b[1]))
        return inter / (aa + ab - inter) if (aa + ab - inter) > 0 else 0

    has_extra = new_result.extra_attrs is not None
    for i, (box, score, cid) in enumerate(zip(new_result.boxes, new_result.scores, new_result.class_ids)):
        if float(score) < conf_thr:
            continue
        cid = int(cid)
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        is_new = not any(_iou(box, ob) >= 0.5 and oc == cid for ob, oc in orig_filtered)

        # 기존 탐지: _draw_detections와 동일한 색상 로직
        extra_label = ""
        cls_name = names.get(cid, str(cid)).lower()
        if has_extra:
            color = _class_color(cid)
            ea = new_result.extra_attrs[i]
            if "person" in cls_name:
                if ea[0] > 0.5:
                    extra_label += f" fall {ea[0]:.2f}"
                    color = (0, 0, 255)
                if ea[1] > 0.5:
                    extra_label += f" crawl {ea[1]:.2f}"
                if ea[2] > 0.5:
                    extra_label += f" jump {ea[2]:.2f}"
                    color = (255, 0, 255)
            elif "face" in cls_name or "mask" in cls_name:
                nomask = 1 - ea[6]
                if nomask > 0.5:
                    extra_label += f" nomask {nomask:.2f}"
                    color = (0, 0, 255)
            elif "wheelchair" in cls_name or "cane" in cls_name:
                color = (0, 0, 255)
        else:
            color = _class_color(cid)

        # 새 탐지: 밝은 빨간색 + [NEW] 라벨
        if is_new:
            color = (0, 0, 255)
            thickness = 3
        else:
            thickness = 2

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        prefix = "[NEW] " if is_new else ""
        label = f"{prefix}{names.get(cid, str(cid))} {float(score):.2f}{extra_label}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(y1 - 4, th + 4)
        cv2.rectangle(vis, (x1, ly - th - 4), (x1 + tw + 2, ly), color, -1)
        cv2.putText(vis, label, (x1 + 1, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def _clear_layout(layout):
    """레이아웃의 모든 위젯 제거"""
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w:
            w.deleteLater()


# ── 추론 작업자 ──────────────────────────────────────────────────────────────

@dataclass
class InferenceAnalysisResult:
    orig_frame: np.ndarray
    preprocessed_frame: np.ndarray   # letterboxed BGR (모델 입력 해상도)
    ratio: float
    pad_w: float
    pad_h: float
    tensor: np.ndarray               # 모델에 실제 입력된 텐서 (1,3,H,W)
    result: DetectionResult          # conf=0.01로 수집한 전체 탐지 결과
    model_input_size: "tuple[int, int]"  # (H, W)
    orig_size: "tuple[int, int]"         # (H, W)
    model_path: str
    input_name: str


class InferenceWorker(QThread):
    inference_done = Signal(object)   # InferenceAnalysisResult
    error = Signal(str)

    _LOW_CONF = 0.01   # 최대한 많은 탐지 결과 수집

    def __init__(self, frame: np.ndarray, model_info: ModelInfo):
        super().__init__()
        self._frame = frame
        self._model_info = model_info

    def run(self):
        model_info = self._model_info
        frame = self._frame
        if model_info.session is None:
            self.error.emit("세션이 없습니다")
            return

        orig_h, orig_w = frame.shape[:2]
        model_h, model_w = model_info.input_size

        try:
            # 모델 배치 크기 확인
            raw_batch = model_info.session.get_inputs()[0].shape[0]
            batch = raw_batch if isinstance(raw_batch, int) and raw_batch > 1 else 1

            if model_info.model_type == "darknet":
                resized = cv2.resize(frame, (model_w, model_h))
                tensor = preprocess_darknet(frame, model_info.input_size)
                if batch > 1:
                    tensor = np.repeat(tensor, batch, axis=0)
                out = model_info.session.run(None, {model_info.input_name: tensor})[0]
                result = postprocess_darknet(
                    out[0:1] if batch > 1 else out,
                    self._LOW_CONF, frame.shape,
                )
                result.infer_ms = 0.0
                analysis = InferenceAnalysisResult(
                    orig_frame=frame,
                    preprocessed_frame=resized,
                    ratio=min(model_w / orig_w, model_h / orig_h),
                    pad_w=0.0, pad_h=0.0,
                    tensor=tensor[:1],
                    result=result,
                    model_input_size=(model_h, model_w),
                    orig_size=(orig_h, orig_w),
                    model_path=model_info.path,
                    input_name=model_info.input_name,
                )
            else:
                padded, ratio, (pad_w, pad_h) = letterbox(frame, model_info.input_size)
                rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
                tensor = np.ascontiguousarray(
                    rgb.transpose(2, 0, 1)[np.newaxis], dtype=np.float32
                ) / 255.0
                if batch > 1:
                    tensor = np.repeat(tensor, batch, axis=0)
                out = model_info.session.run(None, {model_info.input_name: tensor})
                single_out = out[0][0:1] if batch > 1 else out[0]
                if model_info.output_layout == "v8":
                    result = postprocess_v8(single_out, self._LOW_CONF, ratio,
                                            (pad_w, pad_h), frame.shape)
                else:
                    result = postprocess_v5(single_out, self._LOW_CONF, ratio,
                                            (pad_w, pad_h), frame.shape)
                result.infer_ms = 0.0
                analysis = InferenceAnalysisResult(
                    orig_frame=frame,
                    preprocessed_frame=padded,
                    ratio=ratio,
                    pad_w=pad_w, pad_h=pad_h,
                    tensor=tensor[:1],
                    result=result,
                    model_input_size=(model_h, model_w),
                    orig_size=(orig_h, orig_w),
                    model_path=model_info.path,
                    input_name=model_info.input_name,
                )
        except Exception as e:
            self.error.emit(f"추론 실패: {e}")
            return

        self.inference_done.emit(analysis)


# ── 특징맵 추출 작업자 ───────────────────────────────────────────────────────

class FeatureMapWorker(QThread):
    maps_ready = Signal(object)   # np.ndarray (1, C, H, W) 또는 None
    error = Signal(str)

    def __init__(self, model_path: str, input_name: str,
                 node_name: str, tensor: np.ndarray):
        super().__init__()
        self._model_path = model_path
        self._input_name = input_name
        self._node_name = node_name
        self._tensor = tensor

    def run(self):
        try:
            import onnx
            tmp = None
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                tmp = f.name
            onnx.utils.extract_model(
                self._model_path, tmp,
                [self._input_name], [self._node_name]
            )
            try:
                sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
                maps = sess.run(None, {self._input_name: self._tensor.astype(np.float32)})[0]
                self.maps_ready.emit(maps)
            finally:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
        except Exception as e:
            self.error.emit(f"특징맵 추출 실패: {e}")


# ── 전처리 탭 ────────────────────────────────────────────────────────────────

class PreprocessingTab(QWidget):
    def __init__(self):
        super().__init__()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # 이미지 비교 영역 (확대)
        img_row = QHBoxLayout()
        img_row.setSpacing(8)

        orig_box = QGroupBox("원본 프레임")
        orig_box_lay = QVBoxLayout(orig_box)
        self._lbl_orig = QLabel("이미지 없음")
        self._lbl_orig.setAlignment(Qt.AlignCenter)
        self._lbl_orig.setMinimumSize(400, 350)
        orig_box_lay.addWidget(self._lbl_orig)
        img_row.addWidget(orig_box, stretch=1)

        pre_box = QGroupBox("전처리 결과 (모델 입력)")
        pre_box_lay = QVBoxLayout(pre_box)
        self._lbl_pre = QLabel("이미지 없음")
        self._lbl_pre.setAlignment(Qt.AlignCenter)
        self._lbl_pre.setMinimumSize(400, 350)
        pre_box_lay.addWidget(self._lbl_pre)
        img_row.addWidget(pre_box, stretch=1)

        layout.addLayout(img_row)

        # 기본 메트릭
        metrics_group = QGroupBox("이미지 품질 메트릭")
        metrics_form = QFormLayout(metrics_group)
        metrics_form.setSpacing(4)
        self._lbl_blur     = QLabel("—")
        self._lbl_bright   = QLabel("—")
        self._lbl_contrast = QLabel("—")
        self._lbl_scale    = QLabel("—")
        # 추가 메트릭
        self._lbl_snr      = QLabel("—")
        self._lbl_entropy  = QLabel("—")
        self._lbl_edge     = QLabel("—")
        self._lbl_colorful = QLabel("—")
        self._lbl_dynamic  = QLabel("—")

        metrics_form.addRow("흐림도 (Laplacian):", self._lbl_blur)
        metrics_form.addRow("밝기 (평균 픽셀):",   self._lbl_bright)
        metrics_form.addRow("대비 (표준편차):",     self._lbl_contrast)
        metrics_form.addRow("스케일 (원본→모델):",  self._lbl_scale)
        metrics_form.addRow("SNR (신호 대 잡음비):", self._lbl_snr)
        metrics_form.addRow("엔트로피 (정보량):",    self._lbl_entropy)
        metrics_form.addRow("에지 밀도 (Canny):",   self._lbl_edge)
        metrics_form.addRow("색상 풍부도:",          self._lbl_colorful)
        metrics_form.addRow("다이나믹 레인지:",      self._lbl_dynamic)

        self._lbl_snr.setToolTip("높을수록 노이즈 적음. <10: 노이즈 심함, 10~20: 보통, >20: 양호")
        self._lbl_entropy.setToolTip("이미지 정보 복잡도. 높을수록 디테일 풍부. <5: 단순, 5~7: 보통, >7: 복잡")
        self._lbl_edge.setToolTip("에지 픽셀 비율. 높을수록 경계가 뚜렷. 객체 탐지 성능과 양의 상관")
        self._lbl_colorful.setToolTip("색상 다양성 지표. 높을수록 색상 풍부. 단조로운 환경에서 낮음")
        self._lbl_dynamic.setToolTip("밝기 범위 활용도. 낮으면 콘트라스트 부족 → CLAHE 등 히스토그램 균등화 권장")
        layout.addWidget(metrics_group)

        # 인사이트
        self._insight_group = QGroupBox("분석 인사이트")
        self._insight_lay = QVBoxLayout(self._insight_group)
        self._lbl_insight = QLabel("추론 실행 후 인사이트가 표시됩니다.")
        self._lbl_insight.setWordWrap(True)
        self._lbl_insight.setStyleSheet("font-size: 11px; padding: 4px;")
        self._insight_lay.addWidget(self._lbl_insight)
        layout.addWidget(self._insight_group)

        # 히스토그램 영역
        hist_group = QGroupBox("채널별 픽셀 분포 (원본 BGR)")
        hist_lay = QVBoxLayout(hist_group)
        self._hist_placeholder = QWidget()
        self._hist_placeholder.setLayout(QVBoxLayout())
        self._hist_placeholder.setMinimumHeight(160)
        hist_lay.addWidget(self._hist_placeholder)
        layout.addWidget(hist_group)

        # ── 전처리 실시간 추론 비교 ──────────────────────────────────
        pp_group = QGroupBox("전처리 적용 → 실시간 추론 비교")
        pp_main = QHBoxLayout(pp_group)
        pp_main.setSpacing(8)

        # 좌측: 전처리 옵션 세로 정렬
        pp_left = QVBoxLayout()
        pp_left.setSpacing(3)
        pp_left_lbl = QLabel("전처리 방식")
        pp_left_lbl.setStyleSheet("font-weight: bold; font-size: 11px;")
        pp_left.addWidget(pp_left_lbl)

        self._pp_checks: dict = {}
        _PP_METHODS = {
            "clahe":        "CLAHE (적응형 히스토그램 균등화) — 저조도/저대비 개선",
            "gamma":        "감마 보정 (γ=0.7) — 어두운 이미지 밝기 보정",
            "sharpen":      "언샤프 마스크 — 흐릿한 이미지 선명도 향상",
            "denoise":      "바이래터럴 필터 — 에지 보존 노이즈 제거",
            "hist_eq":      "히스토그램 균등화 — 전체 대비 향상",
            "white_bal":    "화이트 밸런스 (Gray World) — 색온도 보정",
            "color_jitter": "색상 지터 — 채도/밝기 변환 시뮬레이션",
            "median_blur":  "미디언 블러 — 소금-후추 노이즈 제거",
            "auto_contrast":"자동 대비 (Percentile Stretch) — 다이나믹 레인지 확장",
        }
        self._pp_methods_map = _PP_METHODS
        for key, desc in _PP_METHODS.items():
            chk = QCheckBox(key)
            chk.setToolTip(desc)
            chk.toggled.connect(self._on_pp_changed)
            self._pp_checks[key] = chk
            pp_left.addWidget(chk)

        pp_left.addSpacing(8)
        self._btn_pp_auto = QPushButton("Auto 최적화")
        self._btn_pp_auto.setToolTip("탐지 수가 최대가 되는 전처리 조합을 자동 탐색")
        self._btn_pp_auto.setEnabled(False)
        self._btn_pp_auto.clicked.connect(self._on_pp_auto)
        pp_left.addWidget(self._btn_pp_auto)
        pp_left.addStretch()
        pp_main.addLayout(pp_left)

        # 우측: 결과 이미지 가로 배치 (1920x1080 비율 고려)
        pp_right = QVBoxLayout()
        pp_right.setSpacing(4)

        pp_img_row = QHBoxLayout()
        pp_img_row.setSpacing(6)

        before_box = QGroupBox("원본 추론 결과")
        before_lay = QVBoxLayout(before_box)
        self._lbl_pp_before = ClickableImageLabel("원본 추론 결과")
        self._lbl_pp_before.setAlignment(Qt.AlignCenter)
        self._lbl_pp_before.setMinimumSize(380, 220)
        before_lay.addWidget(self._lbl_pp_before)
        pp_img_row.addWidget(before_box, stretch=1)

        after_box = QGroupBox("전처리 후 추론 결과")
        after_lay = QVBoxLayout(after_box)
        self._lbl_pp_after = ClickableImageLabel("전처리 후 추론 결과")
        self._lbl_pp_after.setAlignment(Qt.AlignCenter)
        self._lbl_pp_after.setMinimumSize(380, 220)
        after_lay.addWidget(self._lbl_pp_after)
        pp_img_row.addWidget(after_box, stretch=1)

        pp_right.addLayout(pp_img_row)

        self._lbl_pp_compare = QLabel("")
        self._lbl_pp_compare.setWordWrap(True)
        self._lbl_pp_compare.setStyleSheet("font-size: 11px; padding: 4px; background: #F5F5F5; border-radius: 4px;")
        pp_right.addWidget(self._lbl_pp_compare)
        pp_main.addLayout(pp_right, stretch=1)

        layout.addWidget(pp_group)
        self._pp_analysis: "InferenceAnalysisResult | None" = None
        self._pp_model_info: "ModelInfo | None" = None
        self._pp_worker: "InferenceWorker | None" = None
        self._pp_auto_running = False
        self._pp_auto_queue: list = []
        self._pp_auto_best: "tuple[list, int]" = ([], 0)

        layout.addStretch()
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def set_data(self, analysis: InferenceAnalysisResult):
        orig = analysis.orig_frame
        pre  = analysis.preprocessed_frame
        oh, ow = orig.shape[:2]
        mh, mw = analysis.model_input_size

        # 이미지 표시 (확대)
        self._lbl_orig.setPixmap(_bgr_to_pixmap(orig, 560, 420))
        overlay = _draw_letterbox_overlay(pre, analysis.pad_w, analysis.pad_h)
        self._lbl_pre.setPixmap(_bgr_to_pixmap(overlay, 560, 420))

        # 기본 메트릭
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_label = "선명" if lap_var > 200 else ("보통" if lap_var > 80 else "흐림")
        self._lbl_blur.setText(f"{lap_var:.1f}  ({blur_label})")

        mean_bright = float(np.mean(orig))
        std_contrast = float(np.std(orig))
        self._lbl_bright.setText(f"{mean_bright:.1f}")
        self._lbl_contrast.setText(f"{std_contrast:.1f}")
        self._lbl_scale.setText(
            f"{ow}×{oh}  →  {mw}×{mh}  (ratio={analysis.ratio:.4f},"
            f" pad_w={analysis.pad_w:.1f}, pad_h={analysis.pad_h:.1f})"
        )

        # 추가 메트릭 계산
        # 1) SNR (Signal-to-Noise Ratio)
        noise = cv2.GaussianBlur(gray, (7, 7), 0).astype(float) - gray.astype(float)
        noise_std = float(np.std(noise))
        snr = float(np.mean(gray)) / noise_std if noise_std > 0 else 999
        snr_label = "양호" if snr > 20 else ("보통" if snr > 10 else "노이즈 심함")
        self._lbl_snr.setText(f"{snr:.1f}  ({snr_label})")

        # 2) 엔트로피 (정보량)
        hist_g = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        prob = hist_g / hist_g.sum()
        prob = prob[prob > 0]
        entropy = float(-np.sum(prob * np.log2(prob)))
        ent_label = "복잡" if entropy > 7 else ("보통" if entropy > 5 else "단순")
        self._lbl_entropy.setText(f"{entropy:.2f}  ({ent_label})")

        # 3) 에지 밀도 (Canny)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.count_nonzero(edges)) / edges.size * 100
        edge_label = "풍부" if edge_ratio > 15 else ("보통" if edge_ratio > 5 else "부족")
        self._lbl_edge.setText(f"{edge_ratio:.1f}%  ({edge_label})")

        # 4) 색상 풍부도 (Hasler & Süsstrunk metric)
        b, g, r = orig[:,:,0].astype(float), orig[:,:,1].astype(float), orig[:,:,2].astype(float)
        rg = r - g
        yb = 0.5 * (r + g) - b
        colorful = float(np.sqrt(np.mean(rg**2) + np.mean(yb**2)) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2))
        cf_label = "풍부" if colorful > 50 else ("보통" if colorful > 20 else "단조")
        self._lbl_colorful.setText(f"{colorful:.1f}  ({cf_label})")

        # 5) 다이나믹 레인지
        p_low, p_high = np.percentile(gray, [1, 99])
        dyn_range = float(p_high - p_low)
        dr_label = "양호" if dyn_range > 180 else ("보통" if dyn_range > 100 else "부족")
        self._lbl_dynamic.setText(f"{dyn_range:.0f} / 255  ({dr_label})")

        # 인사이트 생성
        insights = []
        if lap_var < 80:
            insights.append("흐림도 높음 → 샤프닝 필터 또는 Unsharp Mask 적용 권장")
        if mean_bright < 60:
            insights.append("이미지 어두움 → 감마 보정 또는 CLAHE 적용으로 밝기 개선 권장")
        elif mean_bright > 200:
            insights.append("이미지 과노출 → 히스토그램 균등화 또는 노출 보정 권장")
        if snr < 10:
            insights.append("노이즈 심함 → 가우시안/바이래터럴 필터 또는 디노이징 전처리 권장")
        if dyn_range < 100:
            insights.append("다이나믹 레인지 부족 → CLAHE 또는 히스토그램 스트레칭 권장")
        if edge_ratio < 5:
            insights.append("에지 부족 → 객체 경계가 불명확할 수 있음, 샤프닝 또는 해상도 향상 검토")
        if colorful < 20:
            insights.append("색상 단조 → 야간/실내 환경 가능성, 색상 증강(Color Jitter) 학습 데이터 보강 검토")
        if not insights:
            insights.append("이미지 품질 양호 — 전처리 없이도 안정적 추론 가능")
        self._lbl_insight.setText("\n".join(insights))

        # 히스토그램
        self._update_histogram(orig)

        # 전처리 실시간 추론용 데이터 저장
        self._pp_analysis = analysis
        self._btn_pp_auto.setEnabled(self._pp_model_info is not None)
        # 원본 결과(bbox 포함) 즉시 표시
        if self._pp_model_info:
            names = self._pp_model_info.names if self._pp_model_info else {}
            orig_vis = _draw_detections(orig, analysis.result, 0.25, names)
            self._lbl_pp_before.setFullPixmap(
                _bgr_to_full_pixmap(orig_vis), _bgr_to_pixmap(orig_vis, 540, 300))
            orig_cnt = int(np.sum(analysis.result.scores >= 0.25))
            self._lbl_pp_compare.setText(f"원본 탐지 수: {orig_cnt}개 — 좌측에서 전처리를 선택하면 자동으로 비교됩니다 (더블클릭으로 확대)")
            self._lbl_pp_after.setText("← 전처리를 선택하세요")

    def _update_histogram(self, img_bgr: np.ndarray):
        _clear_layout(self._hist_placeholder.layout())
        if _MPL:
            fig = Figure(figsize=(7, 2.0), dpi=80, tight_layout=True)  # type: ignore[possibly-undefined]
            ax = fig.add_subplot(111)
            colors = ["#2196F3", "#4CAF50", "#F44336"]
            labels = ["B", "G", "R"]
            for c, (col, lbl) in enumerate(zip(colors, labels)):
                ax.hist(img_bgr[:, :, c].ravel(), bins=50, range=(0, 255),
                        color=col, alpha=0.65, label=lbl)
            ax.set_xlabel("픽셀 값", fontsize=8)
            ax.set_ylabel("빈도",    fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=8)
            canvas = FigureCanvasQTAgg(fig)
            canvas.setFixedHeight(160)
            self._hist_placeholder.layout().addWidget(canvas)
        else:
            lbl = QLabel("matplotlib 미설치 — pip install matplotlib")
            lbl.setAlignment(Qt.AlignCenter)
            self._hist_placeholder.layout().addWidget(lbl)

    def set_model_info(self, model_info: "ModelInfo | None"):
        self._pp_model_info = model_info
        self._btn_pp_auto.setEnabled(self._pp_analysis is not None and model_info is not None)
        # 모델이 나중에 설정된 경우 원본 결과 표시
        if model_info and self._pp_analysis:
            names = model_info.names if model_info else {}
            orig_vis = _draw_detections(self._pp_analysis.orig_frame,
                                         self._pp_analysis.result, 0.25, names)
            self._lbl_pp_before.setFullPixmap(
                _bgr_to_full_pixmap(orig_vis), _bgr_to_pixmap(orig_vis, 540, 300))
            orig_cnt = int(np.sum(self._pp_analysis.result.scores >= 0.25))
            self._lbl_pp_compare.setText(f"원본 탐지 수: {orig_cnt}개 — 좌측에서 전처리를 선택하면 자동으로 비교됩니다 (더블클릭으로 확대)")
            self._lbl_pp_after.setText("← 전처리를 선택하세요")

    def _apply_preprocessing_with_keys(self, img: np.ndarray, keys: list) -> np.ndarray:
        """지정된 전처리 키 목록을 순서대로 적용"""
        out = img.copy()
        if "clahe" in keys:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        if "gamma" in keys:
            table = np.array([((i / 255.0) ** 0.7) * 255 for i in range(256)]).astype(np.uint8)
            out = cv2.LUT(out, table)
        if "sharpen" in keys:
            blur = cv2.GaussianBlur(out, (0, 0), 3)
            out = cv2.addWeighted(out, 1.5, blur, -0.5, 0)
        if "denoise" in keys:
            out = cv2.bilateralFilter(out, 9, 75, 75)
        if "hist_eq" in keys:
            yuv = cv2.cvtColor(out, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            out = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        if "white_bal" in keys:
            r = out.copy().astype(np.float32)
            for c in range(3):
                avg = r[:, :, c].mean()
                if avg > 0:
                    r[:, :, c] *= (128.0 / avg)
            out = np.clip(r, 0, 255).astype(np.uint8)
        if "color_jitter" in keys:
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if "median_blur" in keys:
            out = cv2.medianBlur(out, 5)
        if "auto_contrast" in keys:
            for c in range(3):
                lo, hi = np.percentile(out[:, :, c], [2, 98])
                if hi > lo:
                    out[:, :, c] = np.clip((out[:, :, c].astype(float) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        return out

    def _on_pp_changed(self):
        """체크박스 변경 시 자동 추론 실행"""
        if not self._pp_analysis or not self._pp_model_info:
            return
        if self._pp_auto_running:
            return
        selected = [k for k, chk in self._pp_checks.items() if chk.isChecked()]
        # 원본 결과 (bbox 포함) 항상 표시
        names = self._pp_model_info.names if self._pp_model_info else {}
        orig_vis = _draw_detections(self._pp_analysis.orig_frame,
                                     self._pp_analysis.result, 0.25, names)
        self._lbl_pp_before.setFullPixmap(
            _bgr_to_full_pixmap(orig_vis), _bgr_to_pixmap(orig_vis, 540, 300))
        if not selected:
            self._lbl_pp_after.setText("전처리를 선택하세요")
            self._lbl_pp_compare.setText("")
            return
        # 전처리 적용 후 추론
        processed = self._apply_preprocessing_with_keys(self._pp_analysis.orig_frame, selected)
        self._lbl_pp_compare.setText("추론 중...")
        if self._pp_worker and self._pp_worker.isRunning():
            self._pp_worker.terminate()
        self._pp_worker = InferenceWorker(processed, self._pp_model_info)
        self._pp_worker.inference_done.connect(self._on_pp_done)
        self._pp_worker.error.connect(lambda msg: self._lbl_pp_compare.setText(f"오류: {msg}"))
        self._pp_worker.start()

    def _on_pp_done(self, analysis: InferenceAnalysisResult):
        if self._pp_auto_running:
            self._on_pp_auto_step_done(analysis)
            return
        names = self._pp_model_info.names if self._pp_model_info else {}
        # 추가 탐지는 빨간색으로 표시
        vis = _draw_detections_diff(
            analysis.orig_frame, self._pp_analysis.result if self._pp_analysis else DetectionResult.empty(),
            analysis.result, 0.25, names)
        self._lbl_pp_after.setFullPixmap(
            _bgr_to_full_pixmap(vis), _bgr_to_pixmap(vis, 540, 300))
        orig_cnt = int(np.sum(self._pp_analysis.result.scores >= 0.25)) if self._pp_analysis else 0
        pp_cnt = int(np.sum(analysis.result.scores >= 0.25))
        selected = [k for k, chk in self._pp_checks.items() if chk.isChecked()]
        lines = [f"적용: {', '.join(selected)}",
                 f"원본 {orig_cnt}개  →  전처리 후 {pp_cnt}개"]
        diff = pp_cnt - orig_cnt
        if diff > 0:
            lines.append(f"+{diff}개 추가 탐지")
        elif diff < 0:
            lines.append(f"-{abs(diff)}개 탐지 감소")
        else:
            lines.append("→ 탐지 수 동일")
        self._lbl_pp_compare.setText("\n".join(lines))

    # ── Auto 최적화 ─────────────────────────────────────────────────
    def _on_pp_auto(self):
        """각 전처리를 개별 시도하여 탐지 수가 최대인 조합 탐색"""
        if not self._pp_analysis or not self._pp_model_info:
            return
        self._pp_auto_running = True
        self._btn_pp_auto.setEnabled(False)
        self._btn_pp_auto.setText("탐색 중...")
        keys = list(self._pp_checks.keys())
        # 개별 전처리 각각 시도 + 조합은 상위 2개 조합
        self._pp_auto_queue = [[k] for k in keys]
        self._pp_auto_best = ([], int(np.sum(self._pp_analysis.result.scores >= 0.25)))
        self._pp_auto_results: list = []
        self._on_pp_auto_next()

    def _on_pp_auto_next(self):
        if not self._pp_auto_queue:
            # 개별 결과에서 상위 3개 조합 추가 시도
            if self._pp_auto_results and not hasattr(self, '_pp_auto_phase2'):
                self._pp_auto_phase2 = True
                top = sorted(self._pp_auto_results, key=lambda x: -x[1])[:3]
                if len(top) >= 2:
                    from itertools import combinations
                    for combo in combinations([t[0] for t in top], 2):
                        merged = []
                        for c in combo:
                            merged.extend(c)
                        self._pp_auto_queue.append(list(set(merged)))
                    self._on_pp_auto_next()
                    return
            # 완료
            self._pp_auto_running = False
            self._btn_pp_auto.setEnabled(True)
            self._btn_pp_auto.setText("Auto 최적화")
            if hasattr(self, '_pp_auto_phase2'):
                del self._pp_auto_phase2
            best_keys, best_cnt = self._pp_auto_best
            if best_keys:
                for k, chk in self._pp_checks.items():
                    chk.blockSignals(True)
                    chk.setChecked(k in best_keys)
                    chk.blockSignals(False)
                self._on_pp_changed()
                self._lbl_pp_compare.setText(
                    f"최적 전처리: {', '.join(best_keys)}\n"
                    f"탐지 수: {best_cnt}개"
                )
            else:
                self._lbl_pp_compare.setText("최적화 결과: 원본이 가장 좋음")
            return

        keys = self._pp_auto_queue.pop(0)
        processed = self._apply_preprocessing_with_keys(self._pp_analysis.orig_frame, keys)
        self._lbl_pp_compare.setText(f"탐색 중: {', '.join(keys)} ...")
        self._pp_auto_current_keys = keys
        if self._pp_worker and self._pp_worker.isRunning():
            self._pp_worker.terminate()
        self._pp_worker = InferenceWorker(processed, self._pp_model_info)
        self._pp_worker.inference_done.connect(self._on_pp_done)
        self._pp_worker.error.connect(lambda msg: self._on_pp_auto_next())
        self._pp_worker.start()

    def _on_pp_auto_step_done(self, analysis: InferenceAnalysisResult):
        cnt = int(np.sum(analysis.result.scores >= 0.25))
        keys = self._pp_auto_current_keys
        self._pp_auto_results.append((keys, cnt))
        if cnt > self._pp_auto_best[1]:
            self._pp_auto_best = (keys, cnt)
            # 중간 결과 표시
            names = self._pp_model_info.names if self._pp_model_info else {}
            vis = _draw_detections_diff(
                analysis.orig_frame, self._pp_analysis.result if self._pp_analysis else DetectionResult.empty(),
                analysis.result, 0.25, names)
            self._lbl_pp_after.setFullPixmap(
                _bgr_to_full_pixmap(vis), _bgr_to_pixmap(vis, 540, 300))
        self._on_pp_auto_next()


# ── 탐지 결과 탭 ─────────────────────────────────────────────────────────────

class DetectionTab(QWidget):
    def __init__(self):
        super().__init__()
        self._full_result: "DetectionResult | None" = None
        self._orig_frame: "np.ndarray | None" = None
        self._names: dict = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 탐지 이미지 + Confidence 슬라이더
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Confidence 임계값:"))
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(1, 100)
        self._slider.setValue(25)
        self._slider.setFixedWidth(200)
        self._slider.valueChanged.connect(self._on_threshold_changed)
        ctrl_row.addWidget(self._slider)
        self._lbl_conf_val = QLabel("0.25")
        self._lbl_conf_val.setFixedWidth(40)
        ctrl_row.addWidget(self._lbl_conf_val)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        splitter = QSplitter(Qt.Horizontal)

        # 탐지 이미지
        img_group = QGroupBox("탐지 결과")
        img_lay = QVBoxLayout(img_group)
        self._lbl_img = QLabel("이미지 없음")
        self._lbl_img.setAlignment(Qt.AlignCenter)
        self._lbl_img.setMinimumSize(360, 270)
        img_lay.addWidget(self._lbl_img)
        splitter.addWidget(img_group)

        # 통계 패널
        stats_widget = QWidget()
        stats_lay = QVBoxLayout(stats_widget)
        stats_lay.setSpacing(6)

        # 클래스별 탐지 수 테이블
        tbl_group = QGroupBox("클래스별 탐지 통계")
        tbl_lay = QVBoxLayout(tbl_group)
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["클래스", "탐지 수", "평균 Conf", "최대 Conf"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setMaximumHeight(200)
        tbl_lay.addWidget(self._table)
        stats_lay.addWidget(tbl_group)

        # 신뢰도 히스토그램
        hist_group = QGroupBox("Confidence 분포")
        hist_lay = QVBoxLayout(hist_group)
        self._hist_placeholder = QWidget()
        self._hist_placeholder.setLayout(QVBoxLayout())
        self._hist_placeholder.setMinimumHeight(150)
        hist_lay.addWidget(self._hist_placeholder)
        stats_lay.addWidget(hist_group)

        splitter.addWidget(stats_widget)
        splitter.setSizes([420, 300])
        layout.addWidget(splitter, stretch=1)

    def set_data(self, analysis: InferenceAnalysisResult, names: dict):
        self._full_result = analysis.result
        self._orig_frame  = analysis.orig_frame
        self._names       = names
        self._apply_threshold(self._slider.value() / 100.0)

    def _on_threshold_changed(self, value: int):
        conf = value / 100.0
        self._lbl_conf_val.setText(f"{conf:.2f}")
        self._apply_threshold(conf)

    def _apply_threshold(self, conf: float):
        if self._full_result is None or self._orig_frame is None:
            return

        # 이미지 업데이트
        vis = _draw_detections(self._orig_frame, self._full_result, conf, self._names)
        self._lbl_img.setPixmap(_bgr_to_pixmap(vis, 480, 360))

        # 통계 계산
        mask = self._full_result.scores >= conf
        scores   = self._full_result.scores[mask]
        class_ids = self._full_result.class_ids[mask]

        # 클래스별 집계
        self._table.setRowCount(0)
        if len(scores) > 0:
            from collections import defaultdict
            cls_data: dict = defaultdict(list)
            for s, cid in zip(scores, class_ids):
                cls_data[int(cid)].append(float(s))
            for cid, slist in sorted(cls_data.items(), key=lambda x: -len(x[1])):
                row = self._table.rowCount()
                self._table.insertRow(row)
                self._table.setItem(row, 0, QTableWidgetItem(
                    self._names.get(cid, f"class_{cid}")
                ))
                self._table.setItem(row, 1, QTableWidgetItem(str(len(slist))))
                self._table.setItem(row, 2, QTableWidgetItem(f"{np.mean(slist):.3f}"))
                self._table.setItem(row, 3, QTableWidgetItem(f"{max(slist):.3f}"))

        # 히스토그램 업데이트
        self._update_hist(self._full_result.scores, conf)

    def _update_hist(self, all_scores: np.ndarray, conf_thr: float):
        _clear_layout(self._hist_placeholder.layout())
        if not _MPL:
            lbl = QLabel("matplotlib 미설치")
            lbl.setAlignment(Qt.AlignCenter)
            self._hist_placeholder.layout().addWidget(lbl)
            return
        fig = Figure(figsize=(5, 2.0), dpi=80, tight_layout=True)
        ax = fig.add_subplot(111)
        if len(all_scores) > 0:
            ax.hist(all_scores, bins=25, range=(0, 1),
                    color="#2196F3", edgecolor="white", linewidth=0.4)
        ax.axvline(conf_thr, color="#F44336", linestyle="--", linewidth=1.2,
                   label=f"임계값 {conf_thr:.2f}")
        ax.set_xlabel("Confidence", fontsize=8)
        ax.set_ylabel("탐지 수",    fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=8)
        canvas = FigureCanvasQTAgg(fig)
        canvas.setFixedHeight(150)
        self._hist_placeholder.layout().addWidget(canvas)


# ── 특징맵 탭 ────────────────────────────────────────────────────────────────

def _get_conv_node_outputs(model_path: str) -> list:
    """ONNX 그래프에서 주요 Conv 레이어 출력 노드 이름 (최대 10개) 반환.
    균등 간격으로 샘플링하여 네트워크 전반의 특징을 볼 수 있게 함."""
    try:
        import onnx
        model = onnx.load(model_path)
        all_conv = []
        for node in model.graph.node:
            if node.op_type == "Conv" and node.output:
                all_conv.append(node.output[0])
        if not all_conv:
            return []
        if len(all_conv) <= 10:
            return all_conv
        # 균등 간격 샘플링 (첫/끝 포함)
        step = (len(all_conv) - 1) / 9
        indices = [int(round(i * step)) for i in range(10)]
        return [all_conv[i] for i in indices]
    except Exception:
        return []


class FeatureMapTab(QWidget):
    def __init__(self):
        super().__init__()
        self._tensor: "np.ndarray | None" = None
        self._model_path = ""
        self._input_name = ""
        self._worker: "FeatureMapWorker | None" = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ONNX 패키지 확인
        try:
            import onnx  # noqa: F401
            self._onnx_ok = True
        except ImportError:
            self._onnx_ok = False

        if not self._onnx_ok:
            msg = QLabel(
                "특징맵 추출을 사용하려면 onnx 패키지가 필요합니다.\n\n"
                "pip install onnx"
            )
            msg.setAlignment(Qt.AlignCenter)
            msg.setStyleSheet("color: #888; font-size: 13px;")
            layout.addWidget(msg)
            return

        # 노드 선택
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Conv 레이어:"))
        self._node_combo = QComboBox()
        self._node_combo.setMinimumWidth(300)
        ctrl.addWidget(self._node_combo, stretch=1)
        self._btn_extract = QPushButton("특징맵 추출")
        self._btn_extract.setEnabled(False)
        self._btn_extract.clicked.connect(self._on_extract)
        ctrl.addWidget(self._btn_extract)
        layout.addLayout(ctrl)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # 특징맵 표시 영역 (스크롤)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._maps_container = QWidget()
        self._maps_grid = QHBoxLayout(self._maps_container)
        self._maps_grid.setSpacing(4)
        scroll.setWidget(self._maps_container)
        layout.addWidget(scroll, stretch=1)

        self._lbl_status = QLabel("추론 결과 분석 후 특징맵을 추출할 수 있습니다.")
        self._lbl_status.setAlignment(Qt.AlignCenter)
        self._maps_grid.addWidget(self._lbl_status)

    def set_model(self, model_path: str, input_name: str):
        if not self._onnx_ok:
            return
        self._model_path = model_path
        self._input_name = input_name
        nodes = _get_conv_node_outputs(model_path)
        self._node_combo.clear()
        self._node_combo.addItems(nodes)  # 이미 10개 이내로 필터링됨
        self._btn_extract.setEnabled(bool(nodes) and self._tensor is not None)

    def set_tensor(self, tensor: np.ndarray):
        if not self._onnx_ok:
            return
        self._tensor = tensor
        self._btn_extract.setEnabled(
            bool(self._model_path) and self._node_combo.count() > 0
        )

    def _on_extract(self):
        if self._tensor is None or not self._model_path:
            return
        node = self._node_combo.currentText()
        if not node:
            return
        self._btn_extract.setEnabled(False)
        self._progress.setVisible(True)

        self._worker = FeatureMapWorker(
            self._model_path, self._input_name, node, self._tensor
        )
        self._worker.maps_ready.connect(self._on_maps_ready)
        self._worker.error.connect(self._on_extract_error)
        self._worker.finished.connect(lambda: (
            self._btn_extract.setEnabled(True),
            self._progress.setVisible(False),
        ))
        self._worker.start()

    def _on_maps_ready(self, maps: np.ndarray):
        """특징맵 시각화 (1, C, H, W)"""
        _clear_layout(self._maps_grid)
        if maps is None or maps.ndim != 4:
            self._maps_grid.addWidget(QLabel("특징맵 형식 오류"))
            return

        channels = maps[0]  # (C, H, W)
        # activation energy 기준 상위 16개 채널 선택
        energy = np.sum(np.abs(channels), axis=(1, 2))
        top_idx = np.argsort(energy)[::-1][:16]

        if _MPL:
            n = len(top_idx)
            cols = min(n, 8)
            rows = (n + cols - 1) // cols
            fig = Figure(figsize=(cols * 1.4, rows * 1.4), dpi=70)
            fig.subplots_adjust(wspace=0.05, hspace=0.2)
            for k, idx in enumerate(top_idx):
                ax = fig.add_subplot(rows, cols, k + 1)
                fm = channels[idx]
                ax.imshow(fm, cmap="jet", aspect="auto")
                ax.set_title(f"ch{idx}", fontsize=6)
                ax.axis("off")
            canvas = FigureCanvasQTAgg(fig)
            self._maps_grid.addWidget(canvas)
        else:
            for idx in top_idx:
                fm = channels[idx]
                fm_norm = cv2.normalize(fm, None, 0, 255,
                                        cv2.NORM_MINMAX).astype(np.uint8)
                fm_color = cv2.applyColorMap(fm_norm, cv2.COLORMAP_JET)
                lbl = QLabel()
                lbl.setPixmap(_bgr_to_pixmap(fm_color, 100, 100))
                lbl.setToolTip(f"채널 {idx}")
                self._maps_grid.addWidget(lbl)

    def _on_extract_error(self, msg: str):
        _clear_layout(self._maps_grid)
        self._maps_grid.addWidget(QLabel(f"오류: {msg}"))


# ── 어텐션맵 탭 (Grad-CAM 유사) ──────────────────────────────────────────────

class AttentionMapWorker(QThread):
    """백본 마지막 Conv 레이어의 GAP 가중합으로 어텐션맵 생성 (CAM 방식)"""
    maps_ready = Signal(object)  # np.ndarray BGR heatmap overlay
    error = Signal(str)

    def __init__(self, model_path: str, input_name: str,
                 tensor: np.ndarray, orig_frame: np.ndarray):
        super().__init__()
        self._model_path = model_path
        self._input_name = input_name
        self._tensor = tensor
        self._orig = orig_frame

    def run(self):
        try:
            import onnx
            model = onnx.load(self._model_path)

            # Conv 노드 출력 수집 (후보: 뒤쪽 Conv 중 공간 해상도가 적절한 것)
            conv_outputs = []
            for node in model.graph.node:
                if node.op_type == "Conv" and node.output:
                    conv_outputs.append(node.output[0])
            if not conv_outputs:
                self.error.emit("Conv 레이어를 찾을 수 없습니다")
                return

            # 뒤쪽 1/3 지점부터 역순으로 시도 (백본 끝단 ~ 넥 영역)
            start = max(len(conv_outputs) * 2 // 3, 0)
            candidates = list(reversed(conv_outputs[start:]))
            if not candidates:
                candidates = [conv_outputs[-1]]

            fmaps = None
            for node_name in candidates:
                try:
                    tmp = None
                    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                        tmp = f.name
                    onnx.utils.extract_model(
                        self._model_path, tmp,
                        [self._input_name], [node_name]
                    )
                    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
                    result = sess.run(None, {self._input_name: self._tensor.astype(np.float32)})[0]
                    try:
                        os.unlink(tmp)
                    except Exception:
                        pass
                    # 공간 해상도가 4x4 이상인 것만 사용
                    if result.ndim == 4 and result.shape[2] >= 4 and result.shape[3] >= 4:
                        fmaps = result
                        break
                except Exception:
                    try:
                        os.unlink(tmp)
                    except Exception:
                        pass
                    continue

            if fmaps is None:
                self.error.emit("적절한 특징맵을 추출할 수 없습니다. 모델 구조를 확인하세요.")
                return

            # (1, C, H, W) → 채널 가중합 → 히트맵
            fm = fmaps[0]  # (C, H, W)
            weights = np.mean(fm, axis=(1, 2))  # GAP
            cam = np.sum(weights[:, None, None] * fm, axis=0)
            cam = np.maximum(cam, 0)  # ReLU
            if cam.max() > 0:
                cam = cam / cam.max()
            cam_resized = cv2.resize(cam.astype(np.float32),
                                     (self._orig.shape[1], self._orig.shape[0]))
            heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8),
                                        cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(self._orig, 0.55, heatmap, 0.45, 0)
            self.maps_ready.emit(overlay)
        except Exception as e:
            self.error.emit(f"어텐션맵 추출 실패: {e}")


class AttentionMapTab(QWidget):
    def __init__(self):
        super().__init__()
        self._worker: "AttentionMapWorker | None" = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        ctrl = QHBoxLayout()
        self._btn_extract = QPushButton("어텐션맵 추출 (Grad-CAM)")
        self._btn_extract.setEnabled(False)
        self._btn_extract.clicked.connect(self._on_extract)
        ctrl.addWidget(self._btn_extract)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._lbl_img = QLabel("추론 실행 후 어텐션맵을 추출할 수 있습니다.\n"
                               "모델이 이미지의 어느 영역에 주목하여 판단했는지 시각화합니다.")
        self._lbl_img.setAlignment(Qt.AlignCenter)
        self._lbl_img.setMinimumSize(400, 350)
        layout.addWidget(self._lbl_img, stretch=1)

        self._analysis: "InferenceAnalysisResult | None" = None

    def set_data(self, analysis: InferenceAnalysisResult):
        self._analysis = analysis
        self._btn_extract.setEnabled(True)

    def _on_extract(self):
        if not self._analysis:
            return
        a = self._analysis
        self._btn_extract.setEnabled(False)
        self._progress.setVisible(True)
        self._worker = AttentionMapWorker(
            a.model_path, a.input_name, a.tensor, a.orig_frame
        )
        self._worker.maps_ready.connect(self._on_ready)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(lambda: (
            self._btn_extract.setEnabled(True),
            self._progress.setVisible(False),
        ))
        self._worker.start()

    def _on_ready(self, overlay: np.ndarray):
        self._lbl_img.setPixmap(_bgr_to_pixmap(overlay, 640, 480))

    def _on_error(self, msg: str):
        self._lbl_img.setText(f"오류: {msg}")


# ── 추론 결과 분석 위젯 ──────────────────────────────────────────────────────

class InferenceAnalysisWidget(QWidget):
    def __init__(self, config: AppConfig):
        super().__init__()
        self._config = config
        self._model_info: "ModelInfo | None" = None
        self._cap_total = 0
        self._worker: "InferenceWorker | None" = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── 입력 패널 ──────────────────────────────────────────────────
        input_group = QGroupBox("입력")
        input_lay = QVBoxLayout(input_group)
        input_lay.setSpacing(4)

        # 파일 선택 행
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("파일:"))
        self._file_edit = QLineEdit()
        self._file_edit.setPlaceholderText("이미지 또는 비디오 파일 선택...")
        self._file_edit.textChanged.connect(self._on_file_changed)
        file_row.addWidget(self._file_edit, stretch=1)
        btn_browse = QPushButton("찾아보기")
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._on_browse)
        file_row.addWidget(btn_browse)
        input_lay.addLayout(file_row)

        # 프레임 슬라이더 행 (비디오 전용)
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("프레임:"))
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setRange(0, 0)
        frame_row.addWidget(self._frame_slider, stretch=1)
        self._lbl_frame = QLabel("0 / 0")
        self._lbl_frame.setFixedWidth(80)
        frame_row.addWidget(self._lbl_frame)
        self._frame_slider_row = QWidget()
        self._frame_slider_row.setLayout(frame_row)
        self._frame_slider_row.setVisible(False)
        self._frame_slider.valueChanged.connect(
            lambda v: self._lbl_frame.setText(f"{v} / {self._cap_total}")
        )
        input_lay.addWidget(self._frame_slider_row)

        # 모델 행 — 모델 선택이 기본, 체크박스는 옆에 작게
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("모델:"))
        self._model_path_edit = QLineEdit()
        self._model_path_edit.setPlaceholderText("ONNX 모델 경로...")
        model_row.addWidget(self._model_path_edit, stretch=1)
        btn_model = QPushButton("모델 선택")
        btn_model.setFixedWidth(80)
        btn_model.clicked.connect(self._on_browse_model)
        model_row.addWidget(btn_model)
        self._chk_use_current = QCheckBox("뷰어 모델")
        self._chk_use_current.setToolTip("체크 시 뷰어 탭에서 로드한 모델을 사용합니다")
        self._chk_use_current.setChecked(True)
        self._chk_use_current.toggled.connect(self._on_use_current_toggled)
        model_row.addWidget(self._chk_use_current)
        self._infer_type_combo = QComboBox()
        self._infer_type_combo.addItems(["YOLO", "CenterNet"])
        self._infer_type_combo.setFixedWidth(90)
        self._infer_type_combo.setToolTip("모델 아키텍처 타입")
        model_row.addWidget(self._infer_type_combo)
        self._btn_model_browse = btn_model
        self._on_use_current_toggled(True)  # 초기 상태 반영
        input_lay.addLayout(model_row)

        # 실행 버튼
        run_row = QHBoxLayout()
        self._btn_run = QPushButton("추론 실행")
        self._btn_run.setFixedHeight(28)
        self._btn_run.clicked.connect(self._on_run)
        run_row.addStretch()
        run_row.addWidget(self._btn_run)
        input_lay.addLayout(run_row)

        layout.addWidget(input_group)

        # ── 결과 탭 ────────────────────────────────────────────────────
        self._result_tabs = QTabWidget()
        self._pre_tab     = PreprocessingTab()
        self._det_tab     = DetectionTab()
        self._fmap_tab    = FeatureMapTab()
        self._attn_tab    = AttentionMapTab()
        self._result_tabs.addTab(self._pre_tab,  "전처리")
        self._result_tabs.addTab(self._det_tab,  "탐지 결과")
        self._result_tabs.addTab(self._fmap_tab, "특징맵")
        self._result_tabs.addTab(self._attn_tab, "어텐션맵")
        layout.addWidget(self._result_tabs, stretch=1)

    # ── 외부 API ────────────────────────────────────────────────────────

    def set_model_info(self, model_info: "ModelInfo | None"):
        """main_window에서 모델 로드 시 호출"""
        self._model_info = model_info
        if model_info and self._chk_use_current.isChecked():
            self._model_path_edit.setText(os.path.basename(model_info.path))
            self._infer_type_combo.setCurrentIndex(
                1 if model_info.model_type == "darknet" else 0
            )

    # ── 내부 슬롯 ───────────────────────────────────────────────────────

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "파일 선택", "",
            "미디어 파일 (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv *.ts)"
        )
        if path:
            self._file_edit.setText(path)

    def _on_browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "모델 선택", "", "ONNX 모델 (*.onnx)"
        )
        if path:
            self._model_path_edit.setText(path)

    def _on_file_changed(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        is_video = ext in (".mp4", ".avi", ".mov", ".mkv", ".ts", ".m4v", ".wmv", ".flv")
        self._frame_slider_row.setVisible(is_video)
        if is_video and os.path.isfile(path):
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self._cap_total = max(total - 1, 0)
            self._frame_slider.setRange(0, self._cap_total)
            self._frame_slider.setValue(0)
            self._lbl_frame.setText(f"0 / {self._cap_total}")

    def _on_use_current_toggled(self, checked: bool):
        self._model_path_edit.setEnabled(not checked)
        self._btn_model_browse.setEnabled(not checked)
        if checked and self._model_info:
            self._model_path_edit.setText(os.path.basename(self._model_info.path))
        elif checked:
            self._model_path_edit.setPlaceholderText("뷰어 탭에서 모델을 로드하세요")

    def _on_run(self):
        path = self._file_edit.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "경고", "유효한 파일을 선택하세요.")
            return

        # 모델 결정
        if self._chk_use_current.isChecked():
            model_info = self._model_info
            if model_info is None or model_info.session is None:
                QMessageBox.warning(self, "경고",
                                    "뷰어 탭에서 모델을 먼저 로드하세요.")
                return
        else:
            model_path = self._model_path_edit.text().strip()
            if not model_path or not os.path.isfile(model_path):
                QMessageBox.warning(self, "경고", "모델 파일을 선택하세요.")
                return
            try:
                mtype = "yolo" if self._infer_type_combo.currentIndex() == 0 else "darknet"
                model_info = load_model(model_path, mtype)
            except Exception as e:
                QMessageBox.critical(self, "모델 로드 실패", str(e))
                return

        # 프레임 획득
        ext = os.path.splitext(path)[1].lower()
        is_video = ext in (".mp4", ".avi", ".mov", ".mkv", ".ts", ".m4v", ".wmv", ".flv")
        if is_video:
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self._frame_slider.value())
            ret, frame = cap.read()
            cap.release()
            if not ret:
                QMessageBox.warning(self, "경고", "프레임을 읽을 수 없습니다.")
                return
        else:
            frame = cv2.imread(path)
            if frame is None:
                QMessageBox.warning(self, "경고", "이미지를 읽을 수 없습니다.")
                return

        self._btn_run.setEnabled(False)
        self._btn_run.setText("추론 중...")

        self._last_model_info = model_info  # 실제 추론에 사용된 모델 저장
        self._worker = InferenceWorker(frame, model_info)
        self._worker.inference_done.connect(self._on_inference_done)
        self._worker.error.connect(self._on_inference_error)
        self._worker.finished.connect(lambda: (
            self._btn_run.setEnabled(True),
            self._btn_run.setText("추론 실행"),
        ))
        self._worker.start()

    def _on_inference_done(self, analysis: InferenceAnalysisResult):
        mi = self._last_model_info or self._model_info
        names = mi.names if mi else {}
        self._pre_tab.set_data(analysis)
        self._det_tab.set_data(analysis, names)
        self._fmap_tab.set_model(analysis.model_path, analysis.input_name)
        self._fmap_tab.set_tensor(analysis.tensor)
        self._attn_tab.set_data(analysis)
        self._pre_tab.set_model_info(mi)

    def _on_inference_error(self, msg: str):
        QMessageBox.critical(self, "추론 오류", msg)


# ── 병목 분석 위젯 ───────────────────────────────────────────────────────────

class BottleneckAnalysisWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._analyzer: "BottleneckAnalyzer | None" = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── 설정 패널 ──────────────────────────────────────────────────
        cfg_group = QGroupBox("분석 설정")
        cfg_lay = QFormLayout(cfg_group)
        cfg_lay.setSpacing(4)

        path_row = QHBoxLayout()
        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText("ONNX 모델 파일 경로...")
        path_row.addWidget(self._model_edit, stretch=1)
        btn_browse = QPushButton("찾아보기")
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._on_browse)
        path_row.addWidget(btn_browse)
        cfg_lay.addRow("모델:", path_row)

        self._type_combo = QComboBox()
        self._type_combo.addItems(["YOLO (v5/v7/v8/v9/v11)", "CenterNet"])
        cfg_lay.addRow("모델 타입:", self._type_combo)

        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 32)
        self._batch_spin.setValue(1)
        cfg_lay.addRow("배치 크기:", self._batch_spin)

        self._src_edit = QLineEdit("1920×1080")
        self._src_edit.setFixedWidth(120)
        cfg_lay.addRow("소스 해상도:", self._src_edit)

        from core.ep_manager import EP_VARIANTS, is_ep_available
        self._ep_combo = QComboBox()
        self._ep_combo.addItem("자동  (현재 환경 기본값)", "auto")
        for ep_key, info in EP_VARIANTS.items():
            label = info["label"]
            avail = is_ep_available(ep_key)
            display = label if avail else f"{label}  [미설치]"
            self._ep_combo.addItem(display, ep_key)
            if not avail:
                idx = self._ep_combo.count() - 1
                item = self._ep_combo.model().item(idx)
                if item:
                    item.setEnabled(False)
        self._ep_combo.setToolTip(
            "분석에 사용할 Execution Provider.\n"
            "미설치 항목은 scripts/setup_ep.bat을 실행하여 설치하세요."
        )
        cfg_lay.addRow("EP:", self._ep_combo)

        btn_row = QHBoxLayout()
        self._btn_run  = QPushButton("분석 실행")
        self._btn_stop = QPushButton("중지")
        self._btn_stop.setEnabled(False)
        self._btn_run.clicked.connect(self._on_run)
        self._btn_stop.clicked.connect(self._on_stop)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_run)
        btn_row.addWidget(self._btn_stop)
        cfg_lay.addRow("", btn_row)

        layout.addWidget(cfg_group)

        # 진행바
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress_lbl = QLabel("")
        prog_row = QHBoxLayout()
        prog_row.addWidget(self._progress, stretch=1)
        prog_row.addWidget(self._progress_lbl)
        layout.addLayout(prog_row)

        # ── 결과 영역 ──────────────────────────────────────────────────
        result_splitter = QSplitter(Qt.Horizontal)

        # 좌: 진단 카드
        diag_scroll = QScrollArea()
        diag_scroll.setWidgetResizable(True)
        diag_scroll.setFixedWidth(280)
        diag_widget = QWidget()
        diag_lay = QVBoxLayout(diag_widget)
        diag_lay.setSpacing(8)

        diag_group = QGroupBox("병목 진단")
        diag_form = QFormLayout(diag_group)
        diag_form.setSpacing(5)
        self._lbl_btype  = QLabel("—")
        self._lbl_bscore = QLabel("—")
        self._lbl_pre_ms  = QLabel("—")
        self._lbl_inf_ms  = QLabel("—")
        self._lbl_post_ms = QLabel("—")
        self._lbl_total_ms = QLabel("—")
        self._lbl_btype.setStyleSheet("font-weight: bold; font-size: 13px;")

        # 툴팁: 병목 유형 평가 기준 설명
        self._lbl_btype.setToolTip(
            "병목 유형 판정 기준:\n"
            "• GPU 계산 병목: SM 점유율 >75%, 전처리 비중 <25%\n"
            "• GPU 메모리 대역폭: 메모리 점유율 >85%, SM <70%\n"
            "• PCIe 전송 병목: RX >2GB/s, SM <65%\n"
            "• CPU 전처리 병목: 전처리 비중 >30%, GPU <55%\n"
            "• CPU 병목: 코어 최대 >85%, GPU <40%\n"
            "• 균형 상태: 위 조건 모두 미해당"
        )
        self._lbl_bscore.setToolTip("병목 분류 신뢰도 (0~100%). 높을수록 해당 병목 유형에 확신")
        self._lbl_pre_ms.setToolTip("Letterbox/Resize + 색상변환 + 텐서 변환 평균 시간")
        self._lbl_inf_ms.setToolTip("ONNX Runtime session.run() 평균 실행 시간")
        self._lbl_post_ms.setToolTip("NMS + 좌표 변환 + 결과 필터링 평균 시간")

        diag_form.addRow("병목 유형:", self._lbl_btype)
        diag_form.addRow("신뢰도:",   self._lbl_bscore)
        diag_form.addRow("전처리:",   self._lbl_pre_ms)
        diag_form.addRow("추론:",     self._lbl_inf_ms)
        diag_form.addRow("후처리:",   self._lbl_post_ms)
        diag_form.addRow("총합:",     self._lbl_total_ms)
        diag_lay.addWidget(diag_group)

        rec_group = QGroupBox("최적화 권장 사항")
        rec_lay = QVBoxLayout(rec_group)
        self._lbl_rec = QLabel("—")
        self._lbl_rec.setWordWrap(True)
        self._lbl_rec.setStyleSheet("font-size: 11px;")
        rec_lay.addWidget(self._lbl_rec)
        diag_lay.addWidget(rec_group)

        sys_group = QGroupBox("시스템 리소스")
        sys_form = QFormLayout(sys_group)
        sys_form.setSpacing(3)
        self._lbl_gpu_sm   = QLabel("—")
        self._lbl_gpu_mem  = QLabel("—")
        self._lbl_pcie_rx  = QLabel("—")
        self._lbl_pcie_link = QLabel("—")
        sys_form.addRow("GPU SM:",     self._lbl_gpu_sm)
        sys_form.addRow("GPU 메모리:", self._lbl_gpu_mem)
        sys_form.addRow("PCIe RX:",    self._lbl_pcie_rx)
        sys_form.addRow("PCIe 링크:",  self._lbl_pcie_link)
        diag_lay.addWidget(sys_group)
        diag_lay.addStretch()

        diag_scroll.setWidget(diag_widget)
        result_splitter.addWidget(diag_scroll)

        # 우: 차트 영역
        charts_scroll = QScrollArea()
        charts_scroll.setWidgetResizable(True)
        charts_widget = QWidget()
        self._charts_lay = QVBoxLayout(charts_widget)
        self._charts_lay.setSpacing(8)
        self._charts_lay.addStretch()
        charts_scroll.setWidget(charts_widget)
        result_splitter.addWidget(charts_scroll)
        result_splitter.setSizes([280, 600])

        layout.addWidget(result_splitter, stretch=1)

    # ── 내부 슬롯 ───────────────────────────────────────────────────────

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "ONNX 모델 선택", "", "ONNX 모델 (*.onnx)"
        )
        if path:
            self._model_edit.setText(path)

    def _on_run(self):
        path = self._model_edit.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "경고", "ONNX 모델 파일을 선택하세요.")
            return
        model_type = "yolo" if self._type_combo.currentIndex() == 0 else "darknet"
        src_hw = self._parse_src_hw(self._src_edit.text())

        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress.setValue(0)

        ep_key = self._ep_combo.currentData()
        analyzer = BottleneckAnalyzer(
            model_path=path,
            model_type=model_type,
            batch_size=self._batch_spin.value(),
            src_hw=src_hw,
            ep_key=ep_key,
        )
        self._analyzer = analyzer
        analyzer.progress_updated.connect(self._on_progress)
        analyzer.report_ready.connect(self._on_report)
        analyzer.error.connect(
            lambda msg: QMessageBox.critical(self, "오류", msg)
        )
        analyzer.finished.connect(self._on_finished)
        analyzer.start()

    def _on_stop(self):
        if self._analyzer:
            self._analyzer.stop()

    def _on_progress(self, done: int, total: int, msg: str):
        pct = int(done / total * 100) if total > 0 else 0
        self._progress.setValue(pct)
        self._progress_lbl.setText(msg)

    def _on_finished(self):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.setValue(100)
        self._progress_lbl.setText("완료")

    def _on_report(self, report: BottleneckReport):
        # 진단 카드 업데이트
        label = _BOTTLENECK_LABELS.get(report.bottleneck_type, report.bottleneck_type)
        self._lbl_btype.setText(label)
        self._lbl_bscore.setText(f"{report.bottleneck_score * 100:.0f}%")
        self._lbl_pre_ms.setText(f"{report.mean_pre_ms:.2f} ms")
        self._lbl_inf_ms.setText(f"{report.mean_infer_ms:.2f} ms")
        self._lbl_post_ms.setText(f"{report.mean_post_ms:.2f} ms")
        self._lbl_total_ms.setText(f"{report.mean_total_ms:.2f} ms")
        self._lbl_rec.setText(report.recommendation)

        if report.gpu_sm_util is not None:
            self._lbl_gpu_sm.setText(f"{report.gpu_sm_util}%")
        if report.gpu_mem_util is not None:
            self._lbl_gpu_mem.setText(f"{report.gpu_mem_util}%")
        if report.pcie_rx_mbps is not None:
            self._lbl_pcie_rx.setText(f"{report.pcie_rx_mbps:.0f} MB/s")
        if report.pcie_gen is not None and report.pcie_width is not None:
            max_bw = report.pcie_width * (1 << (report.pcie_gen - 1)) * 985  # ≈ MB/s
            self._lbl_pcie_link.setText(
                f"PCIe {report.pcie_gen}.0 x{report.pcie_width}"
                f" (≈{max_bw//1024} GB/s 이론)"
            )

        # 차트 업데이트
        _clear_layout(self._charts_lay)
        self._add_stage_chart(report)
        self._add_ops_chart(report)
        self._add_onnx_bottleneck_chart(report)
        self._add_cpu_chart(report)
        self._charts_lay.addStretch()

    def _add_stage_chart(self, r: BottleneckReport):
        group = QGroupBox("단계별 시간 비율")
        lay = QVBoxLayout(group)
        if _MPL:
            total = r.mean_total_ms or 1
            fig = Figure(figsize=(5, 0.8), dpi=80, tight_layout=True)
            ax = fig.add_subplot(111)
            pre_pct   = r.mean_pre_ms  / total * 100
            infer_pct = r.mean_infer_ms / total * 100
            post_pct  = r.mean_post_ms  / total * 100
            ax.barh([""], [pre_pct],  color="#4CAF50",
                    label=f"전처리 {r.mean_pre_ms:.1f}ms")
            ax.barh([""], [infer_pct], left=[pre_pct], color="#2196F3",
                    label=f"추론 {r.mean_infer_ms:.1f}ms")
            ax.barh([""], [post_pct], left=[pre_pct + infer_pct], color="#FF9800",
                    label=f"후처리 {r.mean_post_ms:.1f}ms")
            ax.set_xlim(0, 100)
            ax.set_xlabel("비율 (%)", fontsize=8)
            ax.legend(loc="upper right", fontsize=7, ncol=3)
            ax.tick_params(left=False, labelleft=False, labelsize=7)
            canvas = FigureCanvasQTAgg(fig)
            canvas.setFixedHeight(80)
            lay.addWidget(canvas)
        else:
            total = r.mean_total_ms or 1
            lay.addWidget(QLabel(
                f"전처리: {r.mean_pre_ms:.2f}ms ({r.mean_pre_ms/total*100:.1f}%)  "
                f"추론: {r.mean_infer_ms:.2f}ms ({r.mean_infer_ms/total*100:.1f}%)  "
                f"후처리: {r.mean_post_ms:.2f}ms ({r.mean_post_ms/total*100:.1f}%)"
            ))
        self._charts_lay.addWidget(group)

    def _add_ops_chart(self, r: BottleneckReport):
        if not r.top_ops:
            return
        group = QGroupBox(f"상위 연산자 실행 시간 (ORT 프로파일, 총 {r.profile_total_ms:.1f}ms)")
        lay = QVBoxLayout(group)
        ops = r.top_ops[:12]
        if _MPL:
            n = len(ops)
            fig = Figure(figsize=(5, max(n * 0.28 + 0.4, 1.5)), dpi=80, tight_layout=True)
            ax = fig.add_subplot(111)
            names_list = [op[0] for op in ops]
            times_list = [op[1] for op in ops]
            y = range(n)
            ax.barh(y, times_list, color="#2196F3")
            ax.set_yticks(y)
            ax.set_yticklabels(names_list, fontsize=7)
            ax.set_xlabel("ms", fontsize=8)
            ax.invert_yaxis()
            ax.tick_params(labelsize=7)
            canvas = FigureCanvasQTAgg(fig)
            canvas.setFixedHeight(max(n * 22 + 40, 120))
            lay.addWidget(canvas)
        else:
            tbl = QTableWidget(len(ops), 3)
            tbl.setHorizontalHeaderLabels(["연산자", "ms", "%"])
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            tbl.setEditTriggers(QTableWidget.NoEditTriggers)
            for row, (name, ms, pct) in enumerate(ops):
                tbl.setItem(row, 0, QTableWidgetItem(name))
                tbl.setItem(row, 1, QTableWidgetItem(f"{ms:.2f}"))
                tbl.setItem(row, 2, QTableWidgetItem(f"{pct:.1f}"))
            tbl.setFixedHeight(min(len(ops) * 26 + 30, 300))
            lay.addWidget(tbl)
        # #8 인사이트: 상위 연산자 분석
        insight = self._generate_ops_insight(r)
        if insight:
            lbl = QLabel(insight)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size: 11px; color: #333; padding: 4px; background: #F5F5F5; border-radius: 4px;")
            lay.addWidget(lbl)
        self._charts_lay.addWidget(group)

    def _generate_ops_insight(self, r: BottleneckReport) -> str:
        if not r.top_ops:
            return ""
        top = r.top_ops[0]
        name, ms, pct = top[0], top[1], top[2]
        lines = [f"인사이트: 상위 연산자 '{name}'이 전체의 {pct:.1f}%({ms:.1f}ms)를 차지합니다."]
        if "Conv" in name:
            lines.append("→ Conv 연산 최적화: 채널 Pruning, Depthwise Separable Conv 전환, INT8 양자화 검토")
        elif "MatMul" in name or "Gemm" in name:
            lines.append("→ 행렬 곱 최적화: FP16 변환, TensorRT/cuBLAS 가속, 배치 크기 조정 검토")
        elif "Resize" in name or "Upsample" in name:
            lines.append("→ 리사이즈 최적화: nearest 보간법 사용, 해상도 축소, 디코더 경량화 검토")
        elif "Softmax" in name or "Sigmoid" in name:
            lines.append("→ 활성화 함수 최적화: 연산 융합(Fusion) 확인, 근사 함수 사용 검토")
        elif "Concat" in name:
            lines.append("→ Concat 최적화: 메모리 사전 할당, Skip Connection 구조 단순화 검토")
        else:
            lines.append("→ 해당 연산의 입력 크기 축소, 그래프 최적화 레벨 상향, 양자화 적용 검토")
        if len(r.top_ops) >= 3:
            top3_pct = sum(op[2] for op in r.top_ops[:3])
            lines.append(f"→ 상위 3개 연산이 전체의 {top3_pct:.1f}%를 차지 — 이 연산들을 집중 최적화하면 효과적")
        return "\n".join(lines)

    def _add_onnx_bottleneck_chart(self, r: BottleneckReport):
        """#7 ONNX 그래프 병목/비효율 연산자 표시"""
        if not hasattr(r, 'onnx_bottleneck_ops') or not r.onnx_bottleneck_ops:
            return
        group = QGroupBox("ONNX 그래프 병목/비효율 연산자")
        lay = QVBoxLayout(group)
        tbl = QTableWidget(len(r.onnx_bottleneck_ops), 3)
        tbl.setHorizontalHeaderLabels(["연산자 유형", "개수", "설명"])
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.verticalHeader().setVisible(False)
        for row, (op_type, count, _names) in enumerate(r.onnx_bottleneck_ops):
            tbl.setItem(row, 0, QTableWidgetItem(op_type))
            item_cnt = QTableWidgetItem(str(count))
            item_cnt.setTextAlignment(Qt.AlignCenter)
            tbl.setItem(row, 1, item_cnt)
            desc = _ONNX_BOTTLENECK_OPS.get(op_type, "")
            tbl.setItem(row, 2, QTableWidgetItem(desc))
        tbl.setFixedHeight(min(len(r.onnx_bottleneck_ops) * 28 + 30, 250))
        lay.addWidget(tbl)
        total_ops = sum(x[1] for x in r.onnx_bottleneck_ops)
        summary = QLabel(
            f"총 {total_ops}개의 비효율 연산자 발견. "
            "onnxsim(모델 단순화) 또는 ORT graph_optimization_level=ALL 적용으로 "
            "Identity/Gather/Reshape 등을 자동 제거할 수 있습니다."
        )
        summary.setWordWrap(True)
        summary.setStyleSheet("font-size: 11px; color: #333; padding: 4px; background: #FFF3E0; border-radius: 4px;")
        lay.addWidget(summary)
        self._charts_lay.addWidget(group)

    def _add_cpu_chart(self, r: BottleneckReport):
        if not r.cpu_per_core:
            return
        group = QGroupBox("CPU 코어별 평균 점유율")
        lay = QVBoxLayout(group)
        if _MPL:
            n = len(r.cpu_per_core)
            fig = Figure(figsize=(max(n * 0.4, 4), 1.8), dpi=80, tight_layout=True)
            ax = fig.add_subplot(111)
            ax.bar(range(n), r.cpu_per_core, color="#4CAF50")
            ax.set_xticks(range(n))
            ax.set_xticklabels([f"#{i}" for i in range(n)], fontsize=7)
            ax.set_ylabel("%", fontsize=8)
            ax.set_ylim(0, 105)
            ax.axhline(100, color="red", linestyle="--", linewidth=0.8)
            ax.tick_params(labelsize=7)
            canvas = FigureCanvasQTAgg(fig)
            canvas.setFixedHeight(140)
            lay.addWidget(canvas)
        else:
            parts = [f"#{i}: {v:.0f}%" for i, v in enumerate(r.cpu_per_core)]
            lay.addWidget(QLabel("  ".join(parts)))
        # #9 CPU 인사이트
        insight = self._generate_cpu_insight(r)
        if insight:
            lbl = QLabel(insight)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size: 11px; color: #333; padding: 4px; background: #E8F5E9; border-radius: 4px;")
            lay.addWidget(lbl)
        self._charts_lay.addWidget(group)

    def _generate_cpu_insight(self, r: BottleneckReport) -> str:
        if not r.cpu_per_core or len(r.cpu_per_core) < 2:
            return ""
        avg = np.mean(r.cpu_per_core)
        std = np.std(r.cpu_per_core)
        mx = max(r.cpu_per_core)
        mn = min(r.cpu_per_core)
        lines = []
        if std > 15:
            lines.append(f"코어 간 점유율 편차가 큼 (표준편차 {std:.1f}%, 최대 {mx:.0f}%, 최소 {mn:.0f}%)")
            lines.append("→ 특정 코어에 부하 집중 — ORT intra_op_num_threads 조정으로 스레드 분산 검토")
            lines.append("→ 전처리/후처리가 단일 스레드로 실행 중일 수 있음 — 멀티스레드 병렬화 검토")
            if mx > 90:
                lines.append("→ 핫 코어가 90% 이상 — CPU 바운드 가능성 높음, GPU 오프로딩 또는 모델 경량화 권장")
        elif avg > 70:
            lines.append(f"전체 코어 평균 {avg:.0f}%로 고르게 높음 — CPU 전반적 과부하")
            lines.append("→ 모델 입력 해상도 축소, 배치 크기 감소, 또는 GPU EP 전환 검토")
        else:
            lines.append(f"CPU 부하 균형 양호 (평균 {avg:.0f}%, 편차 {std:.1f}%)")
        return "\n".join(lines)

    @staticmethod
    def _parse_src_hw(text: str) -> "tuple[int, int]":
        nums = [int(n) for n in re.findall(r"\d+", text) if int(n) > 0]
        if len(nums) == 1:
            return (nums[0], nums[0])
        if len(nums) >= 2:
            return (nums[1], nums[0])   # W×H → (H, W)
        return (1080, 1920)


# ── 분석 탭 (최상위) ─────────────────────────────────────────────────────────

class AnalysisTab(QWidget):
    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self._config = config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._bottleneck_widget   = BottleneckAnalysisWidget()
        self._inference_widget    = InferenceAnalysisWidget(config)
        self._tabs.addTab(self._bottleneck_widget, "병목 분석")
        self._tabs.addTab(self._inference_widget,  "추론 결과 분석")
        layout.addWidget(self._tabs)

    def set_model_info(self, model_info: "ModelInfo | None"):
        """main_window에서 모델 로드 이벤트 수신"""
        self._inference_widget.set_model_info(model_info)
