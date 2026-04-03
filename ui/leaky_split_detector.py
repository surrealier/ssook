"""Split 누수 탐지: Train/Val/Test 간 중복 이미지 탐지"""
import os, glob, shutil, cv2, numpy as np
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QProgressBar, QMessageBox, QGroupBox, QSlider,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog,
)
from PySide6.QtGui import QImage, QPixmap


def _compute_dhash(path, size=8):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    r = cv2.resize(img, (size + 1, size))
    return (r[:, 1:] > r[:, :-1]).flatten()


class _LeakWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(object)
    error = Signal(str)

    def __init__(self, splits, threshold):
        super().__init__()
        self.splits = splits  # [(name, dir)]
        self.threshold = threshold

    def run(self):
        try:
            # compute hashes per split
            split_data = {}  # {name: [(path, hash)]}
            total_files = 0
            for name, d in self.splits:
                files = []
                for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                    files.extend(glob.glob(os.path.join(d, "**", e), recursive=True))
                total_files += len(files)
                split_data[name] = []
                for fp in sorted(files):
                    h = _compute_dhash(fp)
                    if h is not None:
                        split_data[name].append((fp, h))

            # cross-split comparison
            names = list(split_data.keys())
            leaks = []
            done = 0
            total_cmp = sum(len(split_data[names[i]]) * len(split_data[names[j]])
                           for i in range(len(names)) for j in range(i + 1, len(names)))
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    for fp_a, h_a in split_data[names[i]]:
                        for fp_b, h_b in split_data[names[j]]:
                            dist = int(np.count_nonzero(h_a != h_b))
                            if dist <= self.threshold:
                                leaks.append(dict(
                                    path_a=fp_a, split_a=names[i],
                                    path_b=fp_b, split_b=names[j],
                                    distance=dist))
                            done += 1
                        if done % 5000 == 0:
                            self.progress.emit(min(done, total_cmp), max(total_cmp, 1))
            self.finished_ok.emit(leaks)
        except Exception as e:
            self.error.emit(str(e))


class LeakySplitDetector(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._leaks = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        desc = QLabel("Split 누수 탐지: Train/Val/Test 분할 간 중복·유사 이미지를 탐지합니다.\n"
                       "• dHash(perceptual hash)로 이미지 지문 생성 → 해밍 거리로 유사도 측정\n"
                       "• 누수가 있으면 검증 성능이 과대평가될 수 있으므로 제거 권장\n"
                       "• 더블클릭: 누수 이미지 쌍 나란히 비교")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: gray; font-size: 11px; margin-bottom: 4px;")
        root.addWidget(desc)
        # 폴더 입력
        g = QGroupBox("Split 폴더")
        gl = QVBoxLayout(g)
        self._les = {}
        for name in ["Train", "Val", "Test"]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{name}:"))
            le = QLineEdit()
            self._les[name] = le
            row.addWidget(le, 1)
            btn = QPushButton("찾아보기")
            btn.clicked.connect(lambda _, l=le: self._browse(l))
            row.addWidget(btn)
            gl.addLayout(row)
        root.addWidget(g)

        row_opt = QHBoxLayout()
        row_opt.addWidget(QLabel("거리 임계값:"))
        self._sl_thr = QSlider(Qt.Horizontal)
        self._sl_thr.setRange(0, 20)
        self._sl_thr.setValue(10)
        self._sl_thr.setFixedWidth(120)
        self._lbl_thr = QLabel("10")
        self._sl_thr.valueChanged.connect(lambda v: self._lbl_thr.setText(str(v)))
        row_opt.addWidget(self._sl_thr)
        row_opt.addWidget(self._lbl_thr)
        row_opt.addStretch()
        self._btn_run = QPushButton("탐지")
        self._btn_run.clicked.connect(self._run)
        row_opt.addWidget(self._btn_run)
        root.addLayout(row_opt)

        self._prog = QProgressBar()
        self._prog.setMaximumHeight(18)
        root.addWidget(self._prog)
        self._lbl_summary = QLabel("")
        root.addWidget(self._lbl_summary)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["이미지 A", "Split A", "이미지 B", "Split B", "거리"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.cellDoubleClicked.connect(self._preview)
        root.addWidget(self._table, 1)

    def _browse(self, le):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if d:
            le.setText(d)

    def _run(self):
        splits = []
        for name, le in self._les.items():
            d = le.text()
            if os.path.isdir(d):
                splits.append((name, d))
        if len(splits) < 2:
            QMessageBox.warning(self, "알림", "최소 2개 split 폴더를 지정하세요.")
            return
        self._btn_run.setEnabled(False)
        self._worker = _LeakWorker(splits, self._sl_thr.value())
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(lambda e: (QMessageBox.critical(self, "오류", e), self._btn_run.setEnabled(True)))
        self._worker.start()

    def _on_done(self, leaks):
        self._btn_run.setEnabled(True)
        self._prog.setValue(100)
        self._leaks = leaks
        exact = sum(1 for l in leaks if l["distance"] == 0)
        self._lbl_summary.setText(f"누수: {len(leaks)}쌍 (완전 동일: {exact})")
        self._table.setRowCount(len(leaks))
        for i, l in enumerate(leaks):
            self._table.setItem(i, 0, QTableWidgetItem(os.path.basename(l["path_a"])))
            self._table.setItem(i, 1, QTableWidgetItem(l["split_a"]))
            self._table.setItem(i, 2, QTableWidgetItem(os.path.basename(l["path_b"])))
            self._table.setItem(i, 3, QTableWidgetItem(l["split_b"]))
            self._table.setItem(i, 4, QTableWidgetItem(str(l["distance"])))

    def _preview(self, row, _col):
        if row >= len(self._leaks):
            return
        l = self._leaks[row]
        dlg = QDialog(self)
        dlg.setWindowTitle(f"누수 비교 (거리: {l['distance']})")
        dlg.resize(900, 450)
        lay = QHBoxLayout(dlg)
        for fp, split in [(l["path_a"], l["split_a"]), (l["path_b"], l["split_b"])]:
            col = QVBoxLayout()
            col.addWidget(QLabel(f"[{split}] {os.path.basename(fp)}"))
            img = cv2.imread(fp)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
                lbl = QLabel()
                lbl.setPixmap(QPixmap.fromImage(qimg).scaled(420, 380, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                lbl.setAlignment(Qt.AlignCenter)
                col.addWidget(lbl)
            lay.addLayout(col)
        dlg.exec()
