"""유사 이미지 검색: 코사인 유사도 기반 Top-K 검색
- 인덱스 구축: 이미지를 64x64 그레이스케일로 리사이즈 후 벡터화
- 쿼리: 선택한 이미지와 가장 유사한 Top-K 이미지를 검색
- 더블클릭: 이미지 미리보기 / 우클릭: 파일 경로 복사
"""
import os, glob, cv2, numpy as np
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QProgressBar, QMessageBox, QSlider, QGridLayout,
    QScrollArea, QDialog, QMenu, QApplication,
)
from PySide6.QtGui import QImage, QPixmap


class _IndexWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(object)
    error = Signal(str)

    def __init__(self, img_dir, size=64):
        super().__init__()
        self.img_dir, self.size = img_dir, size

    def run(self):
        try:
            files = []
            for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                files.extend(glob.glob(os.path.join(self.img_dir, "**", e), recursive=True))
            files = sorted(set(files))
            if not files:
                self.error.emit("이미지가 없습니다.")
                return
            vecs, valid = [], []
            for i, fp in enumerate(files):
                img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                r = cv2.resize(img, (self.size, self.size)).flatten().astype(np.float32)
                norm = np.linalg.norm(r)
                if norm > 0: r /= norm
                vecs.append(r); valid.append(fp)
                self.progress.emit(i + 1, len(files))
            self.finished_ok.emit((valid, np.array(vecs)))
        except Exception as e:
            self.error.emit(str(e))


class _ClickableThumb(QLabel):
    """Thumbnail label with double-click preview and right-click copy path."""
    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.filepath = filepath

    def mouseDoubleClickEvent(self, ev):
        img = cv2.imread(self.filepath)
        if img is None: return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        dlg = QDialog(self)
        dlg.setWindowTitle(os.path.basename(self.filepath))
        dlg.resize(900, 700)
        lay = QVBoxLayout(dlg)
        lbl = QLabel()
        lbl.setPixmap(QPixmap.fromImage(qimg).scaled(880, 650, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)
        lay.addWidget(QLabel(self.filepath))
        dlg.exec()

    def contextMenuEvent(self, ev):
        menu = QMenu(self)
        act = menu.addAction("경로 복사")
        if menu.exec(ev.globalPos()) == act:
            QApplication.clipboard().setText(self.filepath)


class SimilaritySearch(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._files, self._vecs = [], None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        desc = QLabel("이미지 폴더를 인덱싱한 후, 쿼리 이미지와 유사한 이미지를 검색합니다.\n"
                       "• 더블클릭: 이미지 미리보기  • 우클릭: 파일 경로 복사")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: gray; font-size: 11px; margin-bottom: 4px;")
        root.addWidget(desc)

        top = QHBoxLayout()
        top.addWidget(QLabel("이미지 폴더:"))
        self._le = QLineEdit()
        self._le.setPlaceholderText("root 폴더 (하위 재귀 탐색)")
        top.addWidget(self._le, 1)
        btn = QPushButton("찾아보기")
        btn.clicked.connect(lambda: self._browse(self._le))
        top.addWidget(btn)
        self._btn_index = QPushButton("인덱스 구축")
        self._btn_index.clicked.connect(self._build_index)
        top.addWidget(self._btn_index)
        self._btn_query = QPushButton("쿼리 이미지")
        self._btn_query.clicked.connect(self._select_query)
        self._btn_query.setEnabled(False)
        top.addWidget(self._btn_query)
        top.addWidget(QLabel("Top-K:"))
        self._sl_k = QSlider(Qt.Horizontal)
        self._sl_k.setRange(5, 100)
        self._sl_k.setValue(20)
        self._sl_k.setFixedWidth(120)
        self._lbl_k = QLabel("20")
        self._sl_k.valueChanged.connect(lambda v: self._lbl_k.setText(str(v)))
        top.addWidget(self._sl_k); top.addWidget(self._lbl_k)
        root.addLayout(top)

        self._prog = QProgressBar(); self._prog.setMaximumHeight(18)
        root.addWidget(self._prog)
        self._lbl_status = QLabel("")
        root.addWidget(self._lbl_status)

        self._lbl_query = QLabel("쿼리 이미지를 선택하세요")
        self._lbl_query.setAlignment(Qt.AlignCenter)
        self._lbl_query.setFixedHeight(160)
        root.addWidget(self._lbl_query)

        self._scroll = QScrollArea(); self._scroll.setWidgetResizable(True)
        self._result_widget = QWidget()
        self._result_layout = QGridLayout(self._result_widget)
        self._result_layout.setSpacing(4)
        self._scroll.setWidget(self._result_widget)
        root.addWidget(self._scroll, 1)

    def _browse(self, le):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if d: le.setText(d)

    def _build_index(self):
        d = self._le.text()
        if not os.path.isdir(d): QMessageBox.warning(self, "알림", "폴더를 선택하세요."); return
        self._btn_index.setEnabled(False)
        self._worker = _IndexWorker(d)
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.finished_ok.connect(self._on_indexed)
        self._worker.error.connect(lambda e: (QMessageBox.critical(self, "오류", e), self._btn_index.setEnabled(True)))
        self._worker.start()

    def _on_indexed(self, data):
        self._files, self._vecs = data
        self._btn_index.setEnabled(True); self._btn_query.setEnabled(True)
        self._prog.setValue(100)
        self._lbl_status.setText(f"{len(self._files)}장 인덱싱 완료")

    def _select_query(self):
        f, _ = QFileDialog.getOpenFileName(self, "쿼리 이미지", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if f: self._search(f)

    def _search(self, query_path):
        img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return
        r = cv2.resize(img, (64, 64)).flatten().astype(np.float32)
        norm = np.linalg.norm(r)
        if norm > 0: r /= norm
        q_img = cv2.imread(query_path)
        if q_img is not None: self._show_on_label(self._lbl_query, q_img, 150)
        sims = self._vecs @ r
        k = self._sl_k.value()
        top_idx = np.argsort(sims)[::-1][:k]
        while self._result_layout.count():
            w = self._result_layout.takeAt(0).widget()
            if w: w.deleteLater()
        cols = 6
        for i, idx in enumerate(top_idx):
            fp = self._files[idx]
            thumb = cv2.imread(fp)
            if thumb is None: continue
            lbl = _ClickableThumb(fp)
            self._show_on_label(lbl, thumb, 130)
            lbl.setToolTip(f"{os.path.basename(fp)}\n유사도: {sims[idx]:.3f}\n더블클릭: 미리보기 | 우클릭: 경로복사")
            container = QWidget()
            cl = QVBoxLayout(container); cl.setContentsMargins(2, 2, 2, 2)
            cl.addWidget(lbl)
            cl.addWidget(QLabel(f"{sims[idx]:.1%}"))
            self._result_layout.addWidget(container, i // cols, i % cols)

    def _show_on_label(self, lbl, img, size):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = size / max(h, w)
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        lbl.setPixmap(QPixmap.fromImage(qimg))
