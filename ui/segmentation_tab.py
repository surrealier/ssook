"""Segmentation 평가 탭: mIoU/Dice, 마스크 오버레이 시각화"""
import os, glob, math

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QComboBox, QGroupBox, QProgressBar,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea, QGridLayout, QDialog, QSplitter,
)

from core.model_loader import load_model
from core.inference import preprocess, letterbox


def _run_seg_inference(mi, frame):
    """Segmentation 모델 추론 → (mask_logits, infer_ms)"""
    import time
    bs = mi.batch_size
    if mi.task_type == "classification":
        from core.inference import preprocess_classification
        tensor = preprocess_classification(frame, mi.input_size)
    else:
        tensor = preprocess(frame, mi.input_size)
    if bs > 1:
        tensor = np.repeat(tensor, bs, axis=0)
    t0 = time.perf_counter()
    out = mi.session.run(None, {mi.input_name: tensor})
    infer_ms = (time.perf_counter() - t0) * 1000.0
    return out[0][0], infer_ms  # (C, H, W) or (H, W)


def compute_seg_metrics(pred_mask, gt_mask, num_classes):
    """per-class IoU, Dice 계산"""
    results = {}
    for c in range(num_classes):
        pred_c = (pred_mask == c)
        gt_c = (gt_mask == c)
        inter = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        iou = float(inter) / (float(union) + 1e-9) if union > 0 else float('nan')
        dice = 2.0 * float(inter) / (float(pred_c.sum() + gt_c.sum()) + 1e-9)
        results[c] = {"iou": iou, "dice": dice,
                       "pred_px": int(pred_c.sum()), "gt_px": int(gt_c.sum())}
    # mIoU (NaN 제외)
    valid = [v["iou"] for v in results.values() if not np.isnan(v["iou"]) and v["gt_px"] > 0]
    results["__overall__"] = {"mIoU": float(np.mean(valid)) if valid else 0.0,
                              "mDice": float(np.mean([v["dice"] for v in results.values() if v["gt_px"] > 0])) if valid else 0.0}
    return results


class _SegEvalWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(dict, list)  # metrics, vis_samples
    error = Signal(str)

    def __init__(self, model_path, model_type, img_dir, gt_mask_dir, num_classes):
        super().__init__()
        self.model_path = model_path
        self.model_type = model_type
        self.img_dir = img_dir
        self.gt_mask_dir = gt_mask_dir
        self.num_classes = num_classes

    def run(self):
        try:
            mi = load_model(self.model_path, model_type=self.model_type)
            files = []
            for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                files.extend(glob.glob(os.path.join(self.img_dir, e)))
            files.sort()
            if not files:
                self.error.emit("이미지가 없습니다."); return

            all_ious = {c: [] for c in range(self.num_classes)}
            all_dices = {c: [] for c in range(self.num_classes)}
            vis_samples = []  # [(img_path, pred_mask, gt_mask)] 최대 20개

            for i, fp in enumerate(files):
                frame = cv2.imread(fp)
                if frame is None: continue
                h, w = frame.shape[:2]
                stem = os.path.splitext(os.path.basename(fp))[0]

                # GT 마스크 로드 (PNG grayscale, 픽셀값=클래스ID)
                gt_path = os.path.join(self.gt_mask_dir, stem + ".png")
                if not os.path.isfile(gt_path):
                    continue
                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is None: continue

                # 추론
                logits, _ = _run_seg_inference(mi, frame)
                if logits.ndim == 3:  # (C, H, W)
                    pred_mask = np.argmax(logits, axis=0).astype(np.uint8)
                else:  # (H, W) — binary
                    pred_mask = (logits > 0).astype(np.uint8)

                # 원본 크기로 리사이즈
                pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                metrics = compute_seg_metrics(pred_mask, gt_mask, self.num_classes)
                for c in range(self.num_classes):
                    if not np.isnan(metrics[c]["iou"]) and metrics[c]["gt_px"] > 0:
                        all_ious[c].append(metrics[c]["iou"])
                        all_dices[c].append(metrics[c]["dice"])

                if len(vis_samples) < 20:
                    vis_samples.append((fp, pred_mask, gt_mask))

                self.progress.emit(i + 1, len(files))

            # 전체 평균
            result = {}
            for c in range(self.num_classes):
                if all_ious[c]:
                    result[c] = {"iou": float(np.mean(all_ious[c])),
                                 "dice": float(np.mean(all_dices[c])),
                                 "count": len(all_ious[c])}
            valid_ious = [v["iou"] for v in result.values()]
            result["__overall__"] = {
                "mIoU": float(np.mean(valid_ious)) if valid_ious else 0.0,
                "mDice": float(np.mean([v["dice"] for v in result.values()])) if valid_ious else 0.0,
                "images": len(files),
            }
            self.finished_ok.emit(result, vis_samples)
        except Exception as e:
            self.error.emit(str(e))


# 마스크 색상 팔레트
_SEG_COLORS = np.array([
    [0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],
    [0,128,128],[128,128,128],[64,0,0],[192,0,0],[64,128,0],[192,128,0],
    [64,0,128],[192,0,128],[64,128,128],[192,128,128],[0,64,0],[128,64,0],
    [0,192,0],[128,192,0],[0,64,128],
], dtype=np.uint8)


def _mask_to_color(mask, alpha=0.5):
    """클래스 마스크 → 컬러 오버레이"""
    h, w = mask.shape
    color = _SEG_COLORS[mask % len(_SEG_COLORS)]
    return color


class SegmentationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._vis_samples = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        grp = QGroupBox("Segmentation 평가 설정")
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
        row2.addWidget(QLabel("GT 마스크 폴더 (PNG):"))
        self._le_gt = QLineEdit()
        row2.addWidget(self._le_gt, 1)
        btn_g = QPushButton("찾아보기")
        btn_g.clicked.connect(lambda: self._browse_dir(self._le_gt))
        row2.addWidget(btn_g)
        g.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("클래스 수:"))
        from PySide6.QtWidgets import QSpinBox
        self._spin_nc = QSpinBox(); self._spin_nc.setRange(2, 256); self._spin_nc.setValue(21)
        row3.addWidget(self._spin_nc)
        row3.addStretch()
        self._btn_run = QPushButton("평가 실행")
        self._btn_run.clicked.connect(self._run)
        row3.addWidget(self._btn_run)
        g.addLayout(row3)

        self._prog = QProgressBar(); self._prog.setMaximumHeight(18)
        g.addWidget(self._prog)
        root.addWidget(grp)

        # 결과
        splitter = QSplitter(Qt.Vertical)

        # 테이블
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["클래스", "IoU", "Dice", "이미지 수"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        splitter.addWidget(self._table)

        # 시각화 갤러리
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._gallery = QWidget()
        self._grid = QGridLayout(self._gallery)
        self._grid.setSpacing(4)
        self._scroll.setWidget(self._gallery)
        splitter.addWidget(self._scroll)
        root.addWidget(splitter, 1)

    def _browse_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "모델 선택", "", "ONNX (*.onnx)")
        if p: self._le_model.setText(p)

    def _browse_dir(self, le):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if d: le.setText(d)

    def _run(self):
        if not self._le_model.text() or not os.path.isdir(self._le_img.text()) or not os.path.isdir(self._le_gt.text()):
            QMessageBox.warning(self, "알림", "모델, 이미지, GT 마스크 폴더를 모두 지정하세요."); return
        mtype = "yolo" if self._combo_type.currentIndex() == 0 else "darknet"
        self._btn_run.setEnabled(False)
        self._prog.setValue(0)
        self._worker = _SegEvalWorker(self._le_model.text(), mtype,
                                      self._le_img.text(), self._le_gt.text(), self._spin_nc.value())
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(lambda e: (QMessageBox.critical(self, "오류", e),
                                              self._btn_run.setEnabled(True)))
        self._worker.start()

    def _on_done(self, result, vis_samples):
        self._btn_run.setEnabled(True)
        self._prog.setValue(100)
        self._vis_samples = vis_samples

        # 테이블
        class_keys = sorted(k for k in result if k != "__overall__")
        self._table.setRowCount(len(class_keys) + 1)

        # overall row
        ov = result["__overall__"]
        self._table.setItem(0, 0, QTableWidgetItem("전체 (mIoU/mDice)"))
        self._table.setItem(0, 1, QTableWidgetItem(f"{ov['mIoU']:.4f}"))
        self._table.setItem(0, 2, QTableWidgetItem(f"{ov['mDice']:.4f}"))
        self._table.setItem(0, 3, QTableWidgetItem(str(ov.get("images", 0))))
        from PySide6.QtGui import QColor
        for c in range(4):
            self._table.item(0, c).setBackground(QColor(46, 80, 46))
            self._table.item(0, c).setForeground(QColor(166, 227, 161))

        for r, cid in enumerate(class_keys, 1):
            v = result[cid]
            self._table.setItem(r, 0, QTableWidgetItem(str(cid)))
            self._table.setItem(r, 1, QTableWidgetItem(f"{v['iou']:.4f}"))
            self._table.setItem(r, 2, QTableWidgetItem(f"{v['dice']:.4f}"))
            self._table.setItem(r, 3, QTableWidgetItem(str(v["count"])))

        # 시각화 갤러리
        while self._grid.count():
            w = self._grid.takeAt(0).widget()
            if w: w.deleteLater()

        cols = 3
        for idx, (fp, pred_mask, gt_mask) in enumerate(vis_samples[:12]):
            img = cv2.imread(fp)
            if img is None: continue
            h, w_ = img.shape[:2]
            size = 200
            scale = size / max(h, w_)

            # 원본 + pred overlay + gt overlay
            pred_color = _mask_to_color(pred_mask)
            gt_color = _mask_to_color(gt_mask)
            overlay_pred = cv2.addWeighted(img, 0.6, pred_color, 0.4, 0)
            overlay_gt = cv2.addWeighted(img, 0.6, gt_color, 0.4, 0)
            combined = np.hstack([overlay_gt, overlay_pred])
            combined = cv2.resize(combined, (int(combined.shape[1]*scale), int(combined.shape[0]*scale)))
            rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
            lbl = QLabel()
            lbl.setPixmap(QPixmap.fromImage(qimg))
            lbl.setToolTip(f"{os.path.basename(fp)}\n좌: GT | 우: Pred")
            self._grid.addWidget(lbl, idx // cols, idx % cols)
