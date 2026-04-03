"""라벨 이상 탐지: bbox 범위, 크기, 클래스별 이상치, 과다 겹침"""
import os, glob, cv2, numpy as np
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QProgressBar, QMessageBox, QGroupBox, QTableWidget,
    QTableWidgetItem, QDialog, QCheckBox, QHeaderView,
)
from PySide6.QtGui import QImage, QPixmap


def _iou(b1, b2):
    x1 = max(b1[0] - b1[2] / 2, b2[0] - b2[2] / 2)
    y1 = max(b1[1] - b1[3] / 2, b2[1] - b2[3] / 2)
    x2 = min(b1[0] + b1[2] / 2, b2[0] + b2[2] / 2)
    y2 = min(b1[1] + b1[3] / 2, b2[1] + b2[3] / 2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = b1[2] * b1[3]
    a2 = b2[2] * b2[3]
    return inter / (a1 + a2 - inter + 1e-9)


class _AnomalyWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(object)
    error = Signal(str)

    def __init__(self, img_dir, lbl_dir):
        super().__init__()
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir

    def run(self):
        try:
            files = []
            for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                files.extend(glob.glob(os.path.join(self.img_dir, "**", e), recursive=True))
            files.sort()
            if not files:
                self.error.emit("이미지가 없습니다.")
                return
            # first pass: collect per-class stats
            class_ws, class_hs = {}, {}
            all_boxes = []  # (img_path, lbl_path, boxes, issues_per_box)
            for i, fp in enumerate(files):
                stem = os.path.splitext(os.path.basename(fp))[0]
                txt = os.path.join(self.lbl_dir, stem + ".txt")
                boxes = []
                if os.path.isfile(txt):
                    with open(txt) as f:
                        for line in f:
                            p = line.strip().split()
                            if len(p) >= 5:
                                cid, cx, cy, bw, bh = int(p[0]), *[float(x) for x in p[1:5]]
                                boxes.append((cid, cx, cy, bw, bh))
                                class_ws.setdefault(cid, []).append(bw)
                                class_hs.setdefault(cid, []).append(bh)
                all_boxes.append((fp, txt, boxes))
                self.progress.emit(i + 1, len(files) * 2)

            # per-class mean/std
            stats = {}
            for cid in class_ws:
                ws, hs = np.array(class_ws[cid]), np.array(class_hs[cid])
                stats[cid] = dict(w_mean=ws.mean(), w_std=ws.std() + 1e-9,
                                  h_mean=hs.mean(), h_std=hs.std() + 1e-9)

            # second pass: detect anomalies
            results = []
            for idx, (fp, txt, boxes) in enumerate(all_boxes):
                for bi, (cid, cx, cy, bw, bh) in enumerate(boxes):
                    issues = []
                    if cx - bw / 2 < -0.01 or cy - bh / 2 < -0.01 or cx + bw / 2 > 1.01 or cy + bh / 2 > 1.01:
                        issues.append("범위초과")
                    if bw < 0.005 or bh < 0.005:
                        issues.append("극소")
                    if bw > 0.95 or bh > 0.95:
                        issues.append("극대")
                    if cid in stats:
                        s = stats[cid]
                        if abs(bw - s["w_mean"]) / s["w_std"] > 3:
                            issues.append("W이상치")
                        if abs(bh - s["h_mean"]) / s["h_std"] > 3:
                            issues.append("H이상치")
                    # overlap
                    for bj, (_, cx2, cy2, bw2, bh2) in enumerate(boxes):
                        if bj <= bi:
                            continue
                        if _iou((cx, cy, bw, bh), (cx2, cy2, bw2, bh2)) > 0.9:
                            issues.append("과다겹침")
                            break
                    if issues:
                        results.append(dict(path=fp, box_idx=bi, cid=cid,
                                            cx=cx, cy=cy, bw=bw, bh=bh,
                                            issues=issues, all_boxes=boxes))
                self.progress.emit(len(files) + idx + 1, len(files) * 2)
            self.finished_ok.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class LabelAnomalyDetector(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._results = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("이미지:"))
        self._le_img = QLineEdit()
        top.addWidget(self._le_img, 1)
        b1 = QPushButton("찾아보기")
        b1.clicked.connect(lambda: self._browse(self._le_img))
        top.addWidget(b1)
        top.addWidget(QLabel("라벨:"))
        self._le_lbl = QLineEdit()
        top.addWidget(self._le_lbl, 1)
        b2 = QPushButton("찾아보기")
        b2.clicked.connect(lambda: self._browse(self._le_lbl))
        top.addWidget(b2)
        self._btn_run = QPushButton("검사")
        self._btn_run.clicked.connect(self._run)
        top.addWidget(self._btn_run)
        root.addLayout(top)

        self._prog = QProgressBar()
        self._prog.setMaximumHeight(18)
        root.addWidget(self._prog)

        # 필터
        frow = QHBoxLayout()
        self._filters = {}
        for name in ["범위초과", "극소", "극대", "W이상치", "H이상치", "과다겹침"]:
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.stateChanged.connect(self._filter)
            frow.addWidget(cb)
            self._filters[name] = cb
        self._lbl_summary = QLabel("")
        frow.addWidget(self._lbl_summary, 1)
        root.addLayout(frow)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["파일", "클래스", "박스", "크기", "문제"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.cellDoubleClicked.connect(self._preview)
        root.addWidget(self._table, 1)

    def _browse(self, le):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if d:
            le.setText(d)

    def _run(self):
        img = self._le_img.text()
        lbl = self._le_lbl.text() or img
        if not os.path.isdir(img):
            QMessageBox.warning(self, "알림", "이미지 폴더를 선택하세요.")
            return
        self._btn_run.setEnabled(False)
        self._worker = _AnomalyWorker(img, lbl)
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(lambda e: (QMessageBox.critical(self, "오류", e), self._btn_run.setEnabled(True)))
        self._worker.start()

    def _on_done(self, results):
        self._btn_run.setEnabled(True)
        self._prog.setValue(100)
        self._results = results
        cats = {}
        for r in results:
            for iss in r["issues"]:
                cats[iss] = cats.get(iss, 0) + 1
        self._lbl_summary.setText(f"이상 박스: {len(results)}개 | " + " | ".join(f"{k}:{v}" for k, v in cats.items()))
        self._filter()

    def _filter(self):
        active = {n for n, cb in self._filters.items() if cb.isChecked()}
        items = [r for r in self._results if any(i in active for i in r["issues"])]
        self._table.setRowCount(len(items))
        for i, r in enumerate(items):
            self._table.setItem(i, 0, QTableWidgetItem(os.path.basename(r["path"])))
            self._table.setItem(i, 1, QTableWidgetItem(str(r["cid"])))
            self._table.setItem(i, 2, QTableWidgetItem(f"({r['cx']:.3f},{r['cy']:.3f})"))
            self._table.setItem(i, 3, QTableWidgetItem(f"{r['bw']:.3f}×{r['bh']:.3f}"))
            self._table.setItem(i, 4, QTableWidgetItem(", ".join(r["issues"])))
            self._table.item(i, 0).setData(Qt.UserRole, r)

    def _preview(self, row, _col):
        r = self._table.item(row, 0).data(Qt.UserRole)
        if not r:
            return
        img = cv2.imread(r["path"])
        if img is None:
            return
        h, w = img.shape[:2]
        for bi, (cid, cx, cy, bw, bh) in enumerate(r["all_boxes"]):
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            color = (0, 0, 255) if bi == r["box_idx"] else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, str(cid), (x1, max(y1 - 4, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        dlg = QDialog(self)
        dlg.setWindowTitle(os.path.basename(r["path"]))
        dlg.resize(800, 600)
        lay = QVBoxLayout(dlg)
        lbl = QLabel()
        lbl.setPixmap(QPixmap.fromImage(qimg).scaled(780, 560, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)
        lay.addWidget(QLabel(f"이상 박스 (빨강): class={r['cid']} | {', '.join(r['issues'])}"))
        dlg.exec()
