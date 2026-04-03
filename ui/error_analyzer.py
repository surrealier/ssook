"""오탐/미탐 분석기: 평가 결과에서 FP/FN 자동 분류, 통계, 갤러리"""
import os
import glob
import math

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QComboBox, QDoubleSpinBox,
    QGroupBox, QProgressBar, QMessageBox, QScrollArea, QGridLayout,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QDialog, QSplitter,
)

from core.model_loader import load_model
from core.inference import run_inference, convert_darknet_to_unified, UNIFIED_NAMES


def _compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def _yolo_to_xyxy_abs(cx, cy, bw, bh, img_w, img_h):
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h
    return (x1, y1, x2, y2)


# ------------------------------------------------------------------ #
# 분석 워커
# ------------------------------------------------------------------ #
class _ErrorAnalysisWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(dict)
    error = Signal(str)

    def __init__(self, img_dir, gt_dir, model_path, model_type, conf, iou_thres=0.5):
        super().__init__()
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.model_path = model_path
        self.model_type = model_type
        self.conf = conf
        self.iou_thres = iou_thres
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

            mi = load_model(self.model_path, model_type=self.model_type)

            fp_list = []  # (img_path, pred_box_xyxy, pred_cid, pred_score, img_w, img_h)
            fn_list = []  # (img_path, gt_box_xyxy, gt_cid, img_w, img_h)
            tp_count = 0

            for i, fp in enumerate(files):
                if self._stop:
                    break
                frame = cv2.imread(fp)
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                stem = os.path.splitext(os.path.basename(fp))[0]

                # GT 로드
                gt_boxes = []
                txt = os.path.join(self.gt_dir, stem + ".txt")
                if os.path.isfile(txt):
                    with open(txt) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cid = int(parts[0])
                                cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                                gt_boxes.append((cid, _yolo_to_xyxy_abs(cx, cy, bw, bh, w, h)))

                # 추론
                res = run_inference(mi, frame, self.conf)
                if self.model_type == "darknet":
                    res = convert_darknet_to_unified(res)

                pred_boxes = list(zip(res.boxes, res.scores, res.class_ids))
                gt_matched = [False] * len(gt_boxes)
                pred_matched = [False] * len(pred_boxes)

                # 매칭
                pairs = []
                for pi, (pbox, pscore, pcid) in enumerate(pred_boxes):
                    for gi, (gcid, gbox) in enumerate(gt_boxes):
                        iou = _compute_iou(pbox, gbox)
                        if iou > 0.1:
                            pairs.append((iou, pi, gi, int(pcid) == gcid))
                pairs.sort(key=lambda x: (-x[3], -x[0]))  # 같은 클래스 우선

                for iou, pi, gi, same_cls in pairs:
                    if pred_matched[pi] or gt_matched[gi]:
                        continue
                    if iou >= self.iou_thres and same_cls:
                        pred_matched[pi] = True
                        gt_matched[gi] = True
                        tp_count += 1

                # FP: 매칭 안 된 pred
                for pi, matched in enumerate(pred_matched):
                    if not matched:
                        pbox, pscore, pcid = pred_boxes[pi]
                        fp_list.append((fp, tuple(pbox), int(pcid), float(pscore), w, h))

                # FN: 매칭 안 된 GT
                for gi, matched in enumerate(gt_matched):
                    if not matched:
                        gcid, gbox = gt_boxes[gi]
                        fn_list.append((fp, gbox, gcid, w, h))

                self.progress.emit(i + 1, len(files))

            self.finished_ok.emit({
                "fp_list": fp_list,
                "fn_list": fn_list,
                "tp_count": tp_count,
                "total_images": len(files),
            })
        except Exception as e:
            self.error.emit(str(e))


# ------------------------------------------------------------------ #
# 통계 계산
# ------------------------------------------------------------------ #
def _compute_stats(error_list, is_fp=True):
    """FP/FN 리스트에서 크기별/클래스별/위치별 통계"""
    if not error_list:
        return {"by_class": {}, "by_size": {"small": 0, "medium": 0, "large": 0},
                "by_position": {"top": 0, "center": 0, "bottom": 0}}

    by_class = {}
    by_size = {"small": 0, "medium": 0, "large": 0}
    by_position = {"top": 0, "center": 0, "bottom": 0}

    for entry in error_list:
        if is_fp:
            _, box, cid, score, img_w, img_h = entry
        else:
            _, box, cid, img_w, img_h = entry

        by_class[cid] = by_class.get(cid, 0) + 1

        # 크기 분류 (면적 비율)
        bw = (box[2] - box[0]) / img_w
        bh = (box[3] - box[1]) / img_h
        area = bw * bh
        if area < 0.01:
            by_size["small"] += 1
        elif area < 0.1:
            by_size["medium"] += 1
        else:
            by_size["large"] += 1

        # 위치 분류 (y 중심)
        cy = (box[1] + box[3]) / 2 / img_h
        if cy < 0.33:
            by_position["top"] += 1
        elif cy < 0.66:
            by_position["center"] += 1
        else:
            by_position["bottom"] += 1

    return {"by_class": by_class, "by_size": by_size, "by_position": by_position}


# ------------------------------------------------------------------ #
# 갤러리 위젯
# ------------------------------------------------------------------ #
class _ErrorGallery(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._title = title
        self._items = []
        self._page = 0
        self._per_page = 20

        nav = QHBoxLayout()
        self._btn_prev = QPushButton("<")
        self._btn_prev.clicked.connect(self._prev)
        nav.addWidget(self._btn_prev)
        self._lbl_page = QLabel("0/0")
        self._lbl_page.setAlignment(Qt.AlignCenter)
        nav.addWidget(self._lbl_page, 1)
        self._lbl_count = QLabel("0건")
        nav.addWidget(self._lbl_count)
        self._btn_next = QPushButton(">")
        self._btn_next.clicked.connect(self._next)
        nav.addWidget(self._btn_next)
        lay.addLayout(nav)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._gallery = QWidget()
        self._grid = QGridLayout(self._gallery)
        self._grid.setSpacing(4)
        self._scroll.setWidget(self._gallery)
        lay.addWidget(self._scroll, 1)

    def set_items(self, items):
        self._items = items
        self._page = 0
        self._lbl_count.setText(f"{len(items)}건")
        self._refresh()

    def _refresh(self):
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        total_pages = max(1, math.ceil(len(self._items) / self._per_page))
        self._lbl_page.setText(f"{self._page + 1}/{total_pages}")

        start = self._page * self._per_page
        end = min(start + self._per_page, len(self._items))
        cols = 5

        for idx, i in enumerate(range(start, end)):
            entry = self._items[i]
            fp = entry[0]
            box = entry[1]
            cid = entry[2]

            img = cv2.imread(fp)
            if img is None:
                continue
            # 박스 영역 crop + 여백
            x1, y1, x2, y2 = [int(v) for v in box]
            h, w = img.shape[:2]
            pad = max(int((x2 - x1) * 0.2), int((y2 - y1) * 0.2), 20)
            cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad); cy2 = min(h, y2 + pad)
            crop = img[cy1:cy2, cx1:cx2]

            # 박스 그리기
            color = (0, 0, 255) if self._title == "FP" else (0, 200, 0)
            cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), color, 2)

            # 썸네일
            size = 140
            scale = size / max(crop.shape[:2])
            thumb = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
            rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)

            lbl = QLabel()
            lbl.setPixmap(QPixmap.fromImage(qimg))
            lbl.setFixedSize(size, size)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("border: 2px solid #444; background: #222;")
            name = UNIFIED_NAMES.get(cid, str(cid))
            if len(entry) == 6:  # FP (has score)
                lbl.setToolTip(f"{os.path.basename(fp)}\nclass: {name}\nscore: {entry[3]:.3f}")
            else:
                lbl.setToolTip(f"{os.path.basename(fp)}\nclass: {name}")
            self._grid.addWidget(lbl, idx // cols, idx % cols)

    def _prev(self):
        if self._page > 0:
            self._page -= 1
            self._refresh()

    def _next(self):
        total_pages = max(1, math.ceil(len(self._items) / self._per_page))
        if self._page < total_pages - 1:
            self._page += 1
            self._refresh()


# ------------------------------------------------------------------ #
# 메인 분석기 위젯
# ------------------------------------------------------------------ #
class ErrorAnalyzer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        # 설정
        grp = QGroupBox("오탐/미탐 분석 설정")
        g = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("모델:"))
        self._le_model = QLineEdit()
        self._le_model.setReadOnly(True)
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
        row3.addWidget(QLabel("Confidence:"))
        self._spin_conf = QDoubleSpinBox()
        self._spin_conf.setRange(0.01, 1.0)
        self._spin_conf.setValue(0.25)
        self._spin_conf.setSingleStep(0.05)
        row3.addWidget(self._spin_conf)
        row3.addWidget(QLabel("IoU Threshold:"))
        self._spin_iou = QDoubleSpinBox()
        self._spin_iou.setRange(0.1, 1.0)
        self._spin_iou.setValue(0.5)
        self._spin_iou.setSingleStep(0.05)
        row3.addWidget(self._spin_iou)
        row3.addStretch()
        self._btn_run = QPushButton("분석 실행")
        self._btn_run.clicked.connect(self._run)
        row3.addWidget(self._btn_run)
        g.addLayout(row3)

        self._prog = QProgressBar()
        self._prog.setMaximumHeight(18)
        g.addWidget(self._prog)
        root.addWidget(grp)

        # 결과 영역
        self._result_tabs = QTabWidget()

        # 요약 탭
        self._summary_widget = QWidget()
        sum_lay = QVBoxLayout(self._summary_widget)
        self._lbl_summary = QLabel("분석을 실행하세요.")
        self._lbl_summary.setWordWrap(True)
        sum_lay.addWidget(self._lbl_summary)
        self._stats_table = QTableWidget()
        self._stats_table.setColumnCount(5)
        self._stats_table.setHorizontalHeaderLabels(["구분", "Small", "Medium", "Large", "합계"])
        self._stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        sum_lay.addWidget(self._stats_table)
        self._result_tabs.addTab(self._summary_widget, "요약")

        # FP 갤러리
        self._fp_gallery = _ErrorGallery("FP")
        self._result_tabs.addTab(self._fp_gallery, "오탐 (FP)")

        # FN 갤러리
        self._fn_gallery = _ErrorGallery("FN")
        self._result_tabs.addTab(self._fn_gallery, "미탐 (FN)")

        root.addWidget(self._result_tabs, 1)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "모델 선택", "", "ONNX (*.onnx)")
        if path:
            self._le_model.setText(path)

    def _browse_dir(self, le):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if d:
            le.setText(d)

    def _run(self):
        model_path = self._le_model.text()
        img_dir = self._le_img.text()
        gt_dir = self._le_gt.text()
        if not model_path or not os.path.isdir(img_dir) or not os.path.isdir(gt_dir):
            QMessageBox.warning(self, "알림", "모델, 이미지 폴더, GT 폴더를 모두 선택하세요.")
            return

        mtype = "yolo" if self._combo_type.currentIndex() == 0 else "darknet"
        self._btn_run.setEnabled(False)
        self._prog.setValue(0)

        self._worker = _ErrorAnalysisWorker(
            img_dir, gt_dir, model_path, mtype,
            self._spin_conf.value(), self._spin_iou.value())
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(lambda e: (QMessageBox.critical(self, "오류", e),
                                              self._btn_run.setEnabled(True)))
        self._worker.start()

    def _on_done(self, result):
        self._btn_run.setEnabled(True)
        self._prog.setValue(100)

        fp_list = result["fp_list"]
        fn_list = result["fn_list"]
        tp = result["tp_count"]
        total_pred = tp + len(fp_list)
        total_gt = tp + len(fn_list)

        prec = tp / (total_pred + 1e-9)
        rec = tp / (total_gt + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)

        summary = (f"이미지: {result['total_images']}장\n"
                   f"TP: {tp}  |  FP: {len(fp_list)}  |  FN: {len(fn_list)}\n"
                   f"Precision: {prec:.4f}  |  Recall: {rec:.4f}  |  F1: {f1:.4f}")
        self._lbl_summary.setText(summary)

        # 통계 테이블
        fp_stats = _compute_stats(fp_list, is_fp=True)
        fn_stats = _compute_stats(fn_list, is_fp=False)

        self._stats_table.setRowCount(2)
        for r, (label, stats) in enumerate([("FP (오탐)", fp_stats), ("FN (미탐)", fn_stats)]):
            self._stats_table.setItem(r, 0, QTableWidgetItem(label))
            self._stats_table.setItem(r, 1, QTableWidgetItem(str(stats["by_size"]["small"])))
            self._stats_table.setItem(r, 2, QTableWidgetItem(str(stats["by_size"]["medium"])))
            self._stats_table.setItem(r, 3, QTableWidgetItem(str(stats["by_size"]["large"])))
            total = sum(stats["by_size"].values())
            self._stats_table.setItem(r, 4, QTableWidgetItem(str(total)))

        # 갤러리
        self._fp_gallery.set_items(fp_list)
        self._fn_gallery.set_items(fn_list)

        self._result_tabs.setTabText(1, f"오탐 FP ({len(fp_list)})")
        self._result_tabs.setTabText(2, f"미탐 FN ({len(fn_list)})")
