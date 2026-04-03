"""이미지 품질 검사: 흐림, 어두움, 과노출, 저정보, 이상 종횡비"""
import os, glob, cv2, numpy as np
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QProgressBar, QMessageBox, QGroupBox, QTableWidget,
    QTableWidgetItem, QDialog, QCheckBox, QSlider, QHeaderView, QSplitter,
)
from PySide6.QtGui import QImage, QPixmap


class _QualityCheckWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(object)
    error = Signal(str)

    def __init__(self, img_dir, blur_thr, bright_lo, bright_hi, overexp_pct, entropy_min):
        super().__init__()
        self.img_dir = img_dir
        self.blur_thr = blur_thr
        self.bright_lo = bright_lo
        self.bright_hi = bright_hi
        self.overexp_pct = overexp_pct
        self.entropy_min = entropy_min

    def run(self):
        try:
            files = []
            for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                files.extend(glob.glob(os.path.join(self.img_dir, "**", e), recursive=True))
            files.sort()
            if not files:
                self.error.emit("이미지가 없습니다.")
                return
            results = []
            for i, fp in enumerate(files):
                img = cv2.imread(fp)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                h, w = img.shape[:2]
                blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                brightness = float(gray.mean())
                overexp = float(np.count_nonzero(gray > 240) / gray.size * 100)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
                hist = hist / hist.sum()
                hist = hist[hist > 0]
                entropy = float(-np.sum(hist * np.log2(hist)))
                ar = w / h
                issues = []
                if blur < self.blur_thr:
                    issues.append("흐림")
                if brightness < self.bright_lo:
                    issues.append("어두움")
                if brightness > self.bright_hi:
                    issues.append("밝음")
                if overexp > self.overexp_pct:
                    issues.append("과노출")
                if entropy < self.entropy_min:
                    issues.append("저정보")
                if ar > 3.0 or ar < 0.33:
                    issues.append("이상비율")
                results.append(dict(path=fp, blur=blur, brightness=brightness,
                                    overexp=overexp, entropy=entropy, ar=ar, issues=issues))
                self.progress.emit(i + 1, len(files))
            self.finished_ok.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class ImageQualityChecker(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._results = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        # 상단
        top = QHBoxLayout()
        top.addWidget(QLabel("이미지 폴더:"))
        self._le = QLineEdit()
        top.addWidget(self._le, 1)
        btn = QPushButton("찾아보기")
        btn.clicked.connect(lambda: self._browse(self._le))
        top.addWidget(btn)
        self._btn_run = QPushButton("검사")
        self._btn_run.clicked.connect(self._run)
        top.addWidget(self._btn_run)
        root.addLayout(top)

        # 임계값
        thr = QGroupBox("임계값")
        tl = QHBoxLayout(thr)
        self._sliders = {}
        for name, lo, hi, default in [("흐림", 10, 500, 100), ("밝기(하)", 0, 100, 30),
                                       ("밝기(상)", 150, 255, 220), ("과노출%", 5, 80, 30),
                                       ("엔트로피", 10, 70, 40)]:
            tl.addWidget(QLabel(name))
            s = QSlider(Qt.Horizontal)
            s.setRange(lo, hi)
            s.setValue(default)
            lbl = QLabel(str(default if name != "엔트로피" else default / 10))
            s.valueChanged.connect(lambda v, l=lbl, n=name: l.setText(str(v / 10) if n == "엔트로피" else str(v)))
            tl.addWidget(s)
            tl.addWidget(lbl)
            self._sliders[name] = s
        root.addWidget(thr)

        self._prog = QProgressBar()
        self._prog.setMaximumHeight(18)
        root.addWidget(self._prog)

        # 필터
        frow = QHBoxLayout()
        self._chk_issues = QCheckBox("문제 이미지만")
        self._chk_issues.stateChanged.connect(self._filter)
        frow.addWidget(self._chk_issues)
        self._lbl_summary = QLabel("")
        frow.addWidget(self._lbl_summary, 1)
        root.addLayout(frow)

        # 테이블
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(["파일", "흐림", "밝기", "과노출%", "엔트로피", "문제"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.cellDoubleClicked.connect(self._preview)
        root.addWidget(self._table, 1)

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
        s = self._sliders
        self._worker = _QualityCheckWorker(
            d, s["흐림"].value(), s["밝기(하)"].value(), s["밝기(상)"].value(),
            s["과노출%"].value(), s["엔트로피"].value() / 10)
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(lambda e: (QMessageBox.critical(self, "오류", e), self._btn_run.setEnabled(True)))
        self._worker.start()

    def _on_done(self, results):
        self._btn_run.setEnabled(True)
        self._prog.setValue(100)
        self._results = results
        issues_count = sum(1 for r in results if r["issues"])
        cats = {}
        for r in results:
            for iss in r["issues"]:
                cats[iss] = cats.get(iss, 0) + 1
        summary = f"전체: {len(results)}장 | 문제: {issues_count}장"
        for k, v in cats.items():
            summary += f" | {k}: {v}"
        self._lbl_summary.setText(summary)
        self._filter()

    def _filter(self):
        only = self._chk_issues.isChecked()
        items = [r for r in self._results if (not only or r["issues"])]
        self._table.setRowCount(len(items))
        for i, r in enumerate(items):
            self._table.setItem(i, 0, QTableWidgetItem(os.path.basename(r["path"])))
            self._table.setItem(i, 1, QTableWidgetItem(f"{r['blur']:.1f}"))
            self._table.setItem(i, 2, QTableWidgetItem(f"{r['brightness']:.1f}"))
            self._table.setItem(i, 3, QTableWidgetItem(f"{r['overexp']:.1f}"))
            self._table.setItem(i, 4, QTableWidgetItem(f"{r['entropy']:.2f}"))
            self._table.setItem(i, 5, QTableWidgetItem(", ".join(r["issues"]) if r["issues"] else "✓"))
            self._table.item(i, 0).setData(Qt.UserRole, r["path"])

    def _preview(self, row, _col):
        fp = self._table.item(row, 0).data(Qt.UserRole)
        if not fp:
            return
        img = cv2.imread(fp)
        if img is None:
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        dlg = QDialog(self)
        dlg.setWindowTitle(os.path.basename(fp))
        dlg.resize(800, 600)
        lay = QVBoxLayout(dlg)
        lbl = QLabel()
        lbl.setPixmap(QPixmap.fromImage(qimg).scaled(780, 560, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)
        dlg.exec()
