"""근사 중복 이미지 탐지: dHash 기반 perceptual hashing"""
import os, glob, shutil, cv2, numpy as np
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QProgressBar, QMessageBox, QGroupBox, QSlider,
    QSplitter, QListWidget, QListWidgetItem, QGridLayout, QScrollArea,
)
from PySide6.QtGui import QImage, QPixmap


def _compute_dhash(path, size=8):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    r = cv2.resize(img, (size + 1, size))
    return (r[:, 1:] > r[:, :-1]).flatten()


def _hamming(h1, h2):
    return int(np.count_nonzero(h1 != h2))


class _DuplicateWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(object)
    error = Signal(str)

    def __init__(self, img_dir, threshold):
        super().__init__()
        self.img_dir = img_dir
        self.threshold = threshold

    def run(self):
        try:
            files = []
            for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                files.extend(glob.glob(os.path.join(self.img_dir, "**", e), recursive=True))
            files.sort()
            if not files:
                self.error.emit("이미지가 없습니다.")
                return
            # compute hashes
            hashes = []
            for i, fp in enumerate(files):
                h = _compute_dhash(fp)
                hashes.append((fp, h))
                self.progress.emit(i + 1, len(files) * 2)
            # find groups via union-find
            n = len(hashes)
            parent = list(range(n))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                pa, pb = find(a), find(b)
                if pa != pb:
                    parent[pa] = pb

            step = len(files)
            for i in range(n):
                if hashes[i][1] is None:
                    continue
                for j in range(i + 1, n):
                    if hashes[j][1] is None:
                        continue
                    if _hamming(hashes[i][1], hashes[j][1]) <= self.threshold:
                        union(i, j)
                if i % 100 == 0:
                    self.progress.emit(step + i, step + n)

            groups = {}
            for i in range(n):
                r = find(i)
                groups.setdefault(r, []).append(hashes[i][0])
            dup_groups = [g for g in groups.values() if len(g) > 1]
            self.finished_ok.emit(dup_groups)
        except Exception as e:
            self.error.emit(str(e))


class NearDuplicateDetector(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._groups = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("이미지 폴더:"))
        self._le = QLineEdit()
        top.addWidget(self._le, 1)
        btn = QPushButton("찾아보기")
        btn.clicked.connect(lambda: self._browse(self._le))
        top.addWidget(btn)
        top.addWidget(QLabel("거리 임계값:"))
        self._sl_thr = QSlider(Qt.Horizontal)
        self._sl_thr.setRange(0, 20)
        self._sl_thr.setValue(10)
        self._sl_thr.setFixedWidth(120)
        self._lbl_thr = QLabel("10")
        self._sl_thr.valueChanged.connect(lambda v: self._lbl_thr.setText(str(v)))
        top.addWidget(self._sl_thr)
        top.addWidget(self._lbl_thr)
        self._btn_run = QPushButton("탐지")
        self._btn_run.clicked.connect(self._run)
        top.addWidget(self._btn_run)
        root.addLayout(top)

        self._prog = QProgressBar()
        self._prog.setMaximumHeight(18)
        root.addWidget(self._prog)
        self._lbl_summary = QLabel("")
        root.addWidget(self._lbl_summary)

        sp = QSplitter(Qt.Horizontal)
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._show_group)
        sp.addWidget(self._list)

        right = QWidget()
        self._grid_lay = QVBoxLayout(right)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._thumb_widget = QWidget()
        self._thumb_layout = QGridLayout(self._thumb_widget)
        self._scroll.setWidget(self._thumb_widget)
        self._grid_lay.addWidget(self._scroll, 1)
        btn_move = QPushButton("중복 이동 (_duplicates 폴더)")
        btn_move.clicked.connect(self._move_dupes)
        self._grid_lay.addWidget(btn_move)
        sp.addWidget(right)
        sp.setStretchFactor(1, 1)
        root.addWidget(sp, 1)

    def _browse(self, le):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if d:
            le.setText(d)

    def _run(self):
        d = self._le.text()
        if not os.path.isdir(d):
            QMessageBox.warning(self, "알림", "폴더를 선택하세요.")
            return
        self._btn_run.setEnabled(False)
        self._worker = _DuplicateWorker(d, self._sl_thr.value())
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(lambda e: (QMessageBox.critical(self, "오류", e), self._btn_run.setEnabled(True)))
        self._worker.start()

    def _on_done(self, groups):
        self._btn_run.setEnabled(True)
        self._prog.setValue(100)
        self._groups = groups
        total_dupes = sum(len(g) - 1 for g in groups)
        self._lbl_summary.setText(f"중복 그룹: {len(groups)}개 | 중복 이미지: {total_dupes}장")
        self._list.clear()
        for i, g in enumerate(groups):
            self._list.addItem(f"그룹 {i + 1} ({len(g)}장)")

    def _show_group(self, row):
        while self._thumb_layout.count():
            w = self._thumb_layout.takeAt(0).widget()
            if w:
                w.deleteLater()
        if row < 0 or row >= len(self._groups):
            return
        for i, fp in enumerate(self._groups[row]):
            img = cv2.imread(fp)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w_ = img.shape[:2]
            scale = 200 / max(h, w_)
            img = cv2.resize(img, (int(w_ * scale), int(h * scale)))
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
            lbl = QLabel()
            lbl.setPixmap(QPixmap.fromImage(qimg))
            lbl.setToolTip(os.path.basename(fp))
            self._thumb_layout.addWidget(lbl, i // 4, i % 4)

    def _move_dupes(self):
        row = self._list.currentRow()
        if row < 0 or row >= len(self._groups):
            return
        group = self._groups[row]
        dup_dir = os.path.join(os.path.dirname(group[0]), "_duplicates")
        os.makedirs(dup_dir, exist_ok=True)
        moved = 0
        for fp in group[1:]:  # keep first
            try:
                shutil.move(fp, os.path.join(dup_dir, os.path.basename(fp)))
                moved += 1
            except Exception:
                pass
        QMessageBox.information(self, "완료", f"{moved}장 이동 완료 → {dup_dir}")
