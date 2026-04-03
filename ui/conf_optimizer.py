"""Confidence 최적화 도구: PR 커브, F1 최적 threshold 탐색"""
import os, glob

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QComboBox, QGroupBox, QProgressBar,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
)

from core.model_loader import load_model
from core.inference import run_inference, convert_darknet_to_unified


def _compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def _yolo_to_xyxy(cx, cy, bw, bh, w, h):
    return ((cx - bw/2)*w, (cy - bh/2)*h, (cx + bw/2)*w, (cy + bh/2)*h)


class _ConfOptWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(dict)  # {class_id: [(conf, tp, fp, fn), ...]}
    error = Signal(str)

    def __init__(self, img_dir, gt_dir, model_path, model_type, iou_thres=0.5):
        super().__init__()
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.model_path = model_path
        self.model_type = model_type
        self.iou_thres = iou_thres

    def run(self):
        try:
            mi = load_model(self.model_path, model_type=self.model_type)
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            files = []
            for e in exts:
                files.extend(glob.glob(os.path.join(self.img_dir, e)))
            files.sort()
            if not files:
                self.error.emit("이미지가 없습니다."); return

            # 모든 이미지에 대해 낮은 conf로 추론, (score, class_id, matched) 수집
            all_preds = []  # (score, cid, img_idx, box)
            all_gt = []     # (cid, img_idx, box)

            for i, fp in enumerate(files):
                frame = cv2.imread(fp)
                if frame is None: continue
                h, w = frame.shape[:2]
                stem = os.path.splitext(os.path.basename(fp))[0]

                # GT
                txt = os.path.join(self.gt_dir, stem + ".txt")
                if os.path.isfile(txt):
                    with open(txt) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cid = int(parts[0])
                                cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                                all_gt.append((cid, i, _yolo_to_xyxy(cx, cy, bw, bh, w, h)))

                # 추론 (매우 낮은 conf)
                res = run_inference(mi, frame, 0.01)
                if self.model_type == "darknet":
                    res = convert_darknet_to_unified(res)
                for box, score, cid in zip(res.boxes, res.scores, res.class_ids):
                    all_preds.append((float(score), int(cid), i, tuple(box)))

                self.progress.emit(i + 1, len(files))

            # 클래스별 PR 커브 계산
            classes = sorted(set(c for c, _, _ in all_gt) | set(c for _, c, _, _ in all_preds))
            result = {}

            for cid in classes:
                preds_c = sorted([(s, idx, b) for s, c, idx, b in all_preds if c == cid], reverse=True)
                gt_c = [(idx, b) for c, idx, b in all_gt if c == cid]
                n_gt = len(gt_c)
                if n_gt == 0 and not preds_c:
                    continue

                # threshold sweep
                thresholds = np.arange(0.05, 1.0, 0.05)
                curve = []
                for thr in thresholds:
                    preds_t = [(s, idx, b) for s, idx, b in preds_c if s >= thr]
                    gt_matched = [False] * n_gt
                    tp = 0
                    for _, pidx, pbox in preds_t:
                        best_iou, best_j = 0, -1
                        for j, (gidx, gbox) in enumerate(gt_c):
                            if gidx != pidx or gt_matched[j]:
                                continue
                            iou = _compute_iou(pbox, gbox)
                            if iou > best_iou:
                                best_iou, best_j = iou, j
                        if best_iou >= self.iou_thres and best_j >= 0:
                            gt_matched[best_j] = True
                            tp += 1
                    fp = len(preds_t) - tp
                    fn = n_gt - tp
                    prec = tp / (tp + fp + 1e-9)
                    rec = tp / (tp + fn + 1e-9)
                    f1 = 2 * prec * rec / (prec + rec + 1e-9)
                    curve.append((float(thr), prec, rec, f1, tp, fp, fn))

                result[cid] = curve

            self.finished_ok.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ConfOptimizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        grp = QGroupBox("Confidence 최적화 설정")
        g = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("모델:"))
        self._le_model = QLineEdit(); self._le_model.setReadOnly(True)
        row1.addWidget(self._le_model, 1)
        btn_m = QPushButton("찾아보기")
        btn_m.clicked.connect(self._browse_model)
        row1.addWidget(btn_m)
        self._combo_type = QComboBox()
        self._combo_type.addItems(["YOLO", "CenterNet"])
        row1.addWidget(self._combo_type)
        g.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("이미지 폴더:"))
        self._le_img = QLineEdit()
        row2.addWidget(self._le_img, 1)
        btn_i = QPushButton("찾아보기")
        btn_i.clicked.connect(lambda: self._browse_dir(self._le_img))
        row2.addWidget(btn_i)
        row2.addWidget(QLabel("GT 라벨:"))
        self._le_gt = QLineEdit()
        row2.addWidget(self._le_gt, 1)
        btn_g = QPushButton("찾아보기")
        btn_g.clicked.connect(lambda: self._browse_dir(self._le_gt))
        row2.addWidget(btn_g)
        g.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addStretch()
        self._btn_run = QPushButton("분석 실행")
        self._btn_run.clicked.connect(self._run)
        row3.addWidget(self._btn_run)
        g.addLayout(row3)

        self._prog = QProgressBar(); self._prog.setMaximumHeight(18)
        g.addWidget(self._prog)
        root.addWidget(grp)

        # 결과: 최적 threshold 테이블
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["클래스", "최적 Conf", "F1", "Precision", "Recall"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        root.addWidget(self._table)

        # 차트 영역
        self._chart_container = QVBoxLayout()
        root.addLayout(self._chart_container, 1)

    def _browse_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "모델 선택", "", "ONNX (*.onnx)")
        if p: self._le_model.setText(p)

    def _browse_dir(self, le):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if d: le.setText(d)

    def _run(self):
        if not self._le_model.text() or not os.path.isdir(self._le_img.text()) or not os.path.isdir(self._le_gt.text()):
            QMessageBox.warning(self, "알림", "모델, 이미지, GT 폴더를 모두 지정하세요.")
            return
        mtype = "yolo" if self._combo_type.currentIndex() == 0 else "darknet"
        self._btn_run.setEnabled(False)
        self._prog.setValue(0)
        self._worker = _ConfOptWorker(self._le_img.text(), self._le_gt.text(),
                                      self._le_model.text(), mtype)
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(lambda e: (QMessageBox.critical(self, "오류", e),
                                              self._btn_run.setEnabled(True)))
        self._worker.start()

    def _on_done(self, result):
        self._btn_run.setEnabled(True)
        self._prog.setValue(100)
        self._result = result

        # 최적 threshold 테이블
        self._table.setRowCount(len(result))
        for r, (cid, curve) in enumerate(sorted(result.items())):
            best = max(curve, key=lambda x: x[3])  # max F1
            thr, prec, rec, f1, tp, fp, fn = best
            self._table.setItem(r, 0, QTableWidgetItem(str(cid)))
            self._table.setItem(r, 1, QTableWidgetItem(f"{thr:.2f}"))
            self._table.setItem(r, 2, QTableWidgetItem(f"{f1:.4f}"))
            self._table.setItem(r, 3, QTableWidgetItem(f"{prec:.4f}"))
            self._table.setItem(r, 4, QTableWidgetItem(f"{rec:.4f}"))

        self._show_charts()

    def _show_charts(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            QMessageBox.warning(self, "알림", "matplotlib이 필요합니다.")
            return

        for i in reversed(range(self._chart_container.count())):
            w = self._chart_container.itemAt(i).widget()
            if w: w.deleteLater()

        n = len(self._result)
        if n == 0: return

        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig = Figure(figsize=(5 * cols, 3.5 * rows))

        for idx, (cid, curve) in enumerate(sorted(self._result.items())):
            ax = fig.add_subplot(rows, cols, idx + 1)
            thrs = [c[0] for c in curve]
            precs = [c[1] for c in curve]
            recs = [c[2] for c in curve]
            f1s = [c[3] for c in curve]

            ax.plot(thrs, precs, 'b-', label='Precision', linewidth=1.5)
            ax.plot(thrs, recs, 'g-', label='Recall', linewidth=1.5)
            ax.plot(thrs, f1s, 'r-', label='F1', linewidth=2)

            best_idx = max(range(len(f1s)), key=lambda i: f1s[i])
            ax.axvline(thrs[best_idx], color='red', linestyle='--', alpha=0.5)
            ax.set_title(f"Class {cid} (best={thrs[best_idx]:.2f}, F1={f1s[best_idx]:.3f})", fontsize=9)
            ax.set_xlabel("Confidence"); ax.set_ylabel("Score")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        canvas = FigureCanvasQTAgg(fig)
        self._chart_container.addWidget(canvas)
