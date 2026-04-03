"""모델 A/B 비교 뷰: 동일 이미지에 2개 모델 결과를 Side-by-Side 비교"""
import os
import glob

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QComboBox, QDoubleSpinBox,
    QGroupBox, QProgressBar, QMessageBox, QScrollArea, QSlider,
    QSplitter,
)

from core.model_loader import load_model
from core.inference import run_inference, convert_darknet_to_unified, UNIFIED_NAMES


# ------------------------------------------------------------------ #
# 비교 워커
# ------------------------------------------------------------------ #
class _CompareWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(list)  # [(img_path, result_a, result_b)]
    error = Signal(str)

    def __init__(self, img_dir, model_a_path, model_a_type,
                 model_b_path, model_b_type, conf):
        super().__init__()
        self.img_dir = img_dir
        self.model_a = (model_a_path, model_a_type)
        self.model_b = (model_b_path, model_b_type)
        self.conf = conf
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            files = []
            for e in exts:
                files.extend(glob.glob(os.path.join(self.img_dir, e)))
            files.sort()
            if not files:
                self.error.emit("이미지가 없습니다.")
                return

            mi_a = load_model(self.model_a[0], model_type=self.model_a[1])
            mi_b = load_model(self.model_b[0], model_type=self.model_b[1])

            results = []
            for i, fp in enumerate(files):
                if self._stop:
                    break
                frame = cv2.imread(fp)
                if frame is None:
                    continue
                res_a = run_inference(mi_a, frame, self.conf)
                if self.model_a[1] == "darknet":
                    res_a = convert_darknet_to_unified(res_a)
                res_b = run_inference(mi_b, frame, self.conf)
                if self.model_b[1] == "darknet":
                    res_b = convert_darknet_to_unified(res_b)
                results.append((fp, res_a, res_b))
                self.progress.emit(i + 1, len(files))

            self.finished_ok.emit(results)
        except Exception as e:
            self.error.emit(str(e))


# ------------------------------------------------------------------ #
# 비교 뷰 위젯
# ------------------------------------------------------------------ #
_COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
           (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]


def _draw_boxes(img, result, names=None):
    """Detection 결과를 이미지에 그리기"""
    vis = img.copy()
    for box, score, cid in zip(result.boxes, result.scores, result.class_ids):
        cid = int(cid)
        color = _COLORS[cid % len(_COLORS)]
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = names.get(cid, str(cid)) if names else str(cid)
        cv2.putText(vis, f"{label} {score:.2f}", (x1, max(y1 - 4, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


def _bgr_to_pixmap(img, max_w=640, max_h=480):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class ModelCompareView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._results = []
        self._index = 0
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        # 설정
        grp = QGroupBox("비교 설정")
        g = QVBoxLayout(grp)

        row_a = QHBoxLayout()
        row_a.addWidget(QLabel("모델 A:"))
        self._le_a = QLineEdit()
        self._le_a.setReadOnly(True)
        row_a.addWidget(self._le_a, 1)
        btn_a = QPushButton("찾아보기")
        btn_a.clicked.connect(lambda: self._browse_model(self._le_a))
        row_a.addWidget(btn_a)
        self._combo_a = QComboBox()
        self._combo_a.addItems(["YOLO", "CenterNet"])
        row_a.addWidget(self._combo_a)
        g.addLayout(row_a)

        row_b = QHBoxLayout()
        row_b.addWidget(QLabel("모델 B:"))
        self._le_b = QLineEdit()
        self._le_b.setReadOnly(True)
        row_b.addWidget(self._le_b, 1)
        btn_b = QPushButton("찾아보기")
        btn_b.clicked.connect(lambda: self._browse_model(self._le_b))
        row_b.addWidget(btn_b)
        self._combo_b = QComboBox()
        self._combo_b.addItems(["YOLO", "CenterNet"])
        row_b.addWidget(self._combo_b)
        g.addLayout(row_b)

        row_img = QHBoxLayout()
        row_img.addWidget(QLabel("이미지 폴더:"))
        self._le_img = QLineEdit()
        row_img.addWidget(self._le_img, 1)
        btn_img = QPushButton("찾아보기")
        btn_img.clicked.connect(self._browse_img)
        row_img.addWidget(btn_img)
        row_img.addWidget(QLabel("Conf:"))
        self._spin_conf = QDoubleSpinBox()
        self._spin_conf.setRange(0.01, 1.0)
        self._spin_conf.setValue(0.25)
        self._spin_conf.setSingleStep(0.05)
        row_img.addWidget(self._spin_conf)
        self._btn_run = QPushButton("비교 실행")
        self._btn_run.clicked.connect(self._run)
        row_img.addWidget(self._btn_run)
        g.addLayout(row_img)

        self._prog = QProgressBar()
        self._prog.setMaximumHeight(18)
        g.addWidget(self._prog)
        root.addWidget(grp)

        # 네비게이션
        nav = QHBoxLayout()
        self._btn_prev = QPushButton("< 이전")
        self._btn_prev.clicked.connect(self._prev)
        nav.addWidget(self._btn_prev)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.valueChanged.connect(self._on_slider)
        nav.addWidget(self._slider, 1)
        self._lbl_nav = QLabel("0 / 0")
        self._lbl_nav.setAlignment(Qt.AlignCenter)
        self._lbl_nav.setFixedWidth(100)
        nav.addWidget(self._lbl_nav)
        self._btn_next = QPushButton("다음 >")
        self._btn_next.clicked.connect(self._next)
        nav.addWidget(self._btn_next)
        root.addLayout(nav)

        # Side-by-Side 이미지
        splitter = QSplitter(Qt.Horizontal)
        self._lbl_a = QLabel("모델 A")
        self._lbl_a.setAlignment(Qt.AlignCenter)
        self._lbl_a.setStyleSheet("background: #1a1a1a;")
        self._lbl_a.setMinimumHeight(400)
        self._lbl_b = QLabel("모델 B")
        self._lbl_b.setAlignment(Qt.AlignCenter)
        self._lbl_b.setStyleSheet("background: #1a1a1a;")
        self._lbl_b.setMinimumHeight(400)
        splitter.addWidget(self._lbl_a)
        splitter.addWidget(self._lbl_b)
        root.addWidget(splitter, 1)

        # 하단 정보
        self._lbl_info = QLabel("")
        self._lbl_info.setWordWrap(True)
        root.addWidget(self._lbl_info)

    def _browse_model(self, le):
        path, _ = QFileDialog.getOpenFileName(self, "모델 선택", "", "ONNX (*.onnx)")
        if path:
            le.setText(path)

    def _browse_img(self):
        d = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택")
        if d:
            self._le_img.setText(d)

    def _run(self):
        a_path, b_path = self._le_a.text(), self._le_b.text()
        img_dir = self._le_img.text()
        if not a_path or not b_path:
            QMessageBox.warning(self, "알림", "모델 A, B를 모두 선택하세요.")
            return
        if not os.path.isdir(img_dir):
            QMessageBox.warning(self, "알림", "이미지 폴더를 선택하세요.")
            return

        a_type = "yolo" if self._combo_a.currentIndex() == 0 else "darknet"
        b_type = "yolo" if self._combo_b.currentIndex() == 0 else "darknet"

        self._btn_run.setEnabled(False)
        self._prog.setValue(0)
        self._worker = _CompareWorker(img_dir, a_path, a_type, b_path, b_type,
                                      self._spin_conf.value())
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(lambda e: (QMessageBox.critical(self, "오류", e),
                                              self._btn_run.setEnabled(True)))
        self._worker.start()

    def _on_done(self, results):
        self._btn_run.setEnabled(True)
        self._prog.setValue(100)
        self._results = results
        self._index = 0
        self._slider.setRange(0, max(0, len(results) - 1))
        self._slider.setValue(0)
        self._show_current()

    def _show_current(self):
        if not self._results:
            return
        fp, res_a, res_b = self._results[self._index]
        self._lbl_nav.setText(f"{self._index + 1} / {len(self._results)}")

        frame = cv2.imread(fp)
        if frame is None:
            return

        vis_a = _draw_boxes(frame, res_a, UNIFIED_NAMES)
        vis_b = _draw_boxes(frame, res_b, UNIFIED_NAMES)

        w = max(self._lbl_a.width(), 400)
        h = max(self._lbl_a.height(), 300)
        self._lbl_a.setPixmap(_bgr_to_pixmap(vis_a, w, h))
        self._lbl_b.setPixmap(_bgr_to_pixmap(vis_b, w, h))

        name_a = os.path.basename(self._le_a.text())
        name_b = os.path.basename(self._le_b.text())
        info = (f"{os.path.basename(fp)}  |  "
                f"A({name_a}): {len(res_a.boxes)}개 {res_a.infer_ms:.1f}ms  |  "
                f"B({name_b}): {len(res_b.boxes)}개 {res_b.infer_ms:.1f}ms")
        self._lbl_info.setText(info)

    def _prev(self):
        if self._index > 0:
            self._index -= 1
            self._slider.setValue(self._index)

    def _next(self):
        if self._index < len(self._results) - 1:
            self._index += 1
            self._slider.setValue(self._index)

    def _on_slider(self, val):
        if val != self._index and 0 <= val < len(self._results):
            self._index = val
            self._show_current()
