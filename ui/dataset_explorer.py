"""FiftyOne-style 데이터셋 탐색기: 이미지/라벨 브라우징, 통계, 필터링, 품질 검사"""
import os
import glob
import math

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QThread, QSize
from PySide6.QtGui import QImage, QPixmap, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QPushButton,
    QLineEdit, QFileDialog, QScrollArea, QGridLayout, QComboBox,
    QGroupBox, QSpinBox, QCheckBox, QListWidget, QListWidgetItem,
    QProgressBar, QMessageBox, QDialog, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QSlider,
)


# ------------------------------------------------------------------ #
# 데이터 로드 워커
# ------------------------------------------------------------------ #
class _DatasetLoadWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(dict)  # dataset_info
    error = Signal(str)

    def __init__(self, img_dir, label_dir):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir

    def run(self):
        try:
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            files = []
            for e in exts:
                files.extend(glob.glob(os.path.join(self.img_dir, "**", e), recursive=True))
            files = sorted(set(files))
            if not files:
                self.error.emit("이미지가 없습니다.")
                return

            samples = []  # [(img_path, boxes, img_w, img_h)]
            class_counts = {}
            box_sizes = []  # (w_ratio, h_ratio)
            box_aspects = []
            total_boxes = 0
            no_label_count = 0
            empty_label_count = 0
            duplicate_box_count = 0

            for i, fp in enumerate(files):
                stem = os.path.splitext(os.path.basename(fp))[0]
                img_dir = os.path.dirname(fp)
                # 라벨: 이미지와 같은 폴더 우선, 없으면 지정된 label_dir에서 탐색
                txt = os.path.join(img_dir, stem + ".txt")
                if not os.path.isfile(txt) and self.label_dir != self.img_dir:
                    # label_dir 내에서 동일 상대경로 시도
                    rel = os.path.relpath(img_dir, self.img_dir)
                    txt_alt = os.path.join(self.label_dir, rel, stem + ".txt")
                    if os.path.isfile(txt_alt):
                        txt = txt_alt

                # 이미지 크기 (빠르게 읽기)
                img = cv2.imread(fp)
                if img is None:
                    continue
                h, w = img.shape[:2]

                boxes = []
                if os.path.isfile(txt):
                    with open(txt) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cid = int(parts[0])
                                cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                                boxes.append((cid, cx, cy, bw, bh))
                                class_counts[cid] = class_counts.get(cid, 0) + 1
                                box_sizes.append((bw, bh))
                                box_aspects.append(bw / (bh + 1e-9))
                                total_boxes += 1
                    if not boxes:
                        empty_label_count += 1
                    # 중복 박스 검사
                    seen = set()
                    for b in boxes:
                        key = (b[0], round(b[1], 4), round(b[2], 4), round(b[3], 4), round(b[4], 4))
                        if key in seen:
                            duplicate_box_count += 1
                        seen.add(key)
                else:
                    no_label_count += 1

                samples.append((fp, boxes, w, h))
                self.progress.emit(i + 1, len(files))

            info = {
                "samples": samples,
                "class_counts": class_counts,
                "box_sizes": np.array(box_sizes) if box_sizes else np.zeros((0, 2)),
                "box_aspects": np.array(box_aspects) if box_aspects else np.zeros(0),
                "total_images": len(samples),
                "total_boxes": total_boxes,
                "no_label_count": no_label_count,
                "empty_label_count": empty_label_count,
                "duplicate_box_count": duplicate_box_count,
            }
            self.finished_ok.emit(info)
        except Exception as e:
            self.error.emit(str(e))


# ------------------------------------------------------------------ #
# 썸네일 위젯
# ------------------------------------------------------------------ #
class _ThumbLabel(QLabel):
    clicked = Signal(int)

    def __init__(self, index, parent=None):
        super().__init__(parent)
        self._index = index
        self.setFixedSize(160, 160)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid #444; background: #222;")

    def mousePressEvent(self, ev):
        self.clicked.emit(self._index)


# ------------------------------------------------------------------ #
# 메인 탐색기 위젯
# ------------------------------------------------------------------ #
class DatasetExplorer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples = []
        self._filtered = []
        self._info = {}
        self._page = 0
        self._per_page = 50
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        # 상단: 폴더 선택
        top = QHBoxLayout()
        top.addWidget(QLabel("이미지 폴더:"))
        self._le_img = QLineEdit()
        self._le_img.textChanged.connect(lambda t: self._le_lbl.setText(t))
        top.addWidget(self._le_img, 1)
        btn_img = QPushButton("찾아보기")
        btn_img.clicked.connect(lambda: self._browse(self._le_img))
        top.addWidget(btn_img)
        top.addWidget(QLabel("라벨 폴더:"))
        self._le_lbl = QLineEdit()
        self._le_lbl.setPlaceholderText("(이미지와 동일 폴더 — 다르면 직접 지정)")
        top.addWidget(self._le_lbl, 1)
        btn_lbl = QPushButton("찾아보기")
        btn_lbl.clicked.connect(lambda: self._browse(self._le_lbl))
        top.addWidget(btn_lbl)
        self._btn_load = QPushButton("로드")
        self._btn_load.clicked.connect(self._load_dataset)
        top.addWidget(self._btn_load)
        root.addLayout(top)

        self._prog = QProgressBar()
        self._prog.setMaximumHeight(18)
        root.addWidget(self._prog)

        # 메인 영역: 좌측 필터/통계 + 우측 갤러리
        splitter = QSplitter(Qt.Horizontal)

        # 좌측 패널
        left = QWidget()
        left.setFixedWidth(280)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)

        # 필터
        fgrp = QGroupBox("필터")
        flay = QVBoxLayout(fgrp)
        flay.addWidget(QLabel("클래스 필터:"))
        self._class_list = QListWidget()
        self._class_list.setSelectionMode(QListWidget.MultiSelection)
        self._class_list.itemSelectionChanged.connect(self._apply_filter)
        flay.addWidget(self._class_list)

        row_lbl = QHBoxLayout()
        self._chk_no_label = QCheckBox("라벨 없음만")
        self._chk_no_label.stateChanged.connect(self._apply_filter)
        row_lbl.addWidget(self._chk_no_label)
        self._chk_empty = QCheckBox("빈 라벨만")
        self._chk_empty.stateChanged.connect(self._apply_filter)
        row_lbl.addWidget(self._chk_empty)
        flay.addLayout(row_lbl)

        row_box = QHBoxLayout()
        row_box.addWidget(QLabel("최소 박스 수:"))
        self._spin_min_box = QSpinBox()
        self._spin_min_box.setRange(0, 9999)
        self._spin_min_box.valueChanged.connect(self._apply_filter)
        row_box.addWidget(self._spin_min_box)
        flay.addLayout(row_box)

        left_lay.addWidget(fgrp)

        # 통계
        sgrp = QGroupBox("통계")
        slay = QVBoxLayout(sgrp)
        self._lbl_stats = QLabel("데이터셋을 로드하세요.")
        self._lbl_stats.setWordWrap(True)
        slay.addWidget(self._lbl_stats)
        self._btn_charts = QPushButton("분포 차트 보기")
        self._btn_charts.clicked.connect(self._show_charts)
        self._btn_charts.setEnabled(False)
        slay.addWidget(self._btn_charts)
        left_lay.addWidget(sgrp)

        # 품질 검사
        qgrp = QGroupBox("라벨 품질")
        qlay = QVBoxLayout(qgrp)
        self._lbl_quality = QLabel("-")
        self._lbl_quality.setWordWrap(True)
        qlay.addWidget(self._lbl_quality)
        left_lay.addWidget(qgrp)

        left_lay.addStretch()
        splitter.addWidget(left)

        # 우측: 갤러리
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)

        # 페이지 네비게이션
        nav = QHBoxLayout()
        self._btn_prev = QPushButton("< 이전")
        self._btn_prev.clicked.connect(self._prev_page)
        nav.addWidget(self._btn_prev)
        self._lbl_page = QLabel("0 / 0")
        self._lbl_page.setAlignment(Qt.AlignCenter)
        nav.addWidget(self._lbl_page, 1)
        self._lbl_count = QLabel("0장")
        nav.addWidget(self._lbl_count)
        self._btn_next = QPushButton("다음 >")
        self._btn_next.clicked.connect(self._next_page)
        nav.addWidget(self._btn_next)
        right_lay.addLayout(nav)

        # 갤러리 스크롤
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._gallery_widget = QWidget()
        self._gallery_layout = QGridLayout(self._gallery_widget)
        self._gallery_layout.setSpacing(4)
        self._scroll.setWidget(self._gallery_widget)
        right_lay.addWidget(self._scroll, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

    # ---- helpers ----
    def _browse(self, le):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if d:
            le.setText(d)

    def _load_dataset(self):
        img_dir = self._le_img.text()
        lbl_dir = self._le_lbl.text()
        if not os.path.isdir(img_dir):
            QMessageBox.warning(self, "알림", "이미지 폴더를 선택하세요.")
            return
        if not lbl_dir:
            lbl_dir = img_dir  # 같은 폴더에 라벨이 있을 수 있음
        self._btn_load.setEnabled(False)
        self._prog.setValue(0)
        self._worker = _DatasetLoadWorker(img_dir, lbl_dir)
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.finished_ok.connect(self._on_loaded)
        self._worker.error.connect(lambda e: (QMessageBox.critical(self, "오류", e),
                                              self._btn_load.setEnabled(True)))
        self._worker.start()

    def _on_loaded(self, info):
        self._btn_load.setEnabled(True)
        self._prog.setValue(100)
        self._info = info
        self._samples = info["samples"]

        # 클래스 필터 업데이트
        self._class_list.clear()
        for cid in sorted(info["class_counts"].keys()):
            cnt = info["class_counts"][cid]
            item = QListWidgetItem(f"{cid} ({cnt})")
            item.setData(Qt.UserRole, cid)
            item.setSelected(True)
            self._class_list.addItem(item)

        # 통계
        stats = (f"이미지: {info['total_images']}장\n"
                 f"박스: {info['total_boxes']}개\n"
                 f"클래스: {len(info['class_counts'])}종\n"
                 f"이미지당 평균 박스: {info['total_boxes'] / max(info['total_images'], 1):.1f}")
        self._lbl_stats.setText(stats)

        # 품질
        issues = []
        if info["no_label_count"]:
            issues.append(f"라벨 없음: {info['no_label_count']}장")
        if info["empty_label_count"]:
            issues.append(f"빈 라벨: {info['empty_label_count']}장")
        if info["duplicate_box_count"]:
            issues.append(f"중복 박스: {info['duplicate_box_count']}개")
        self._lbl_quality.setText("\n".join(issues) if issues else "이상 없음")

        self._btn_charts.setEnabled(True)
        self._apply_filter()

    def _apply_filter(self):
        if not self._samples:
            return
        selected_classes = set()
        for item in self._class_list.selectedItems():
            selected_classes.add(item.data(Qt.UserRole))

        no_label_only = self._chk_no_label.isChecked()
        empty_only = self._chk_empty.isChecked()
        min_boxes = self._spin_min_box.value()

        filtered = []
        for fp, boxes, w, h in self._samples:
            stem = os.path.splitext(os.path.basename(fp))[0]
            img_dir = os.path.dirname(fp)
            # 이미지와 같은 폴더에서 라벨 탐색
            has_label_file = os.path.isfile(os.path.join(img_dir, stem + ".txt"))
            if not has_label_file:
                lbl_root = self._le_lbl.text() or self._le_img.text()
                if lbl_root and lbl_root != self._le_img.text():
                    rel = os.path.relpath(img_dir, self._le_img.text())
                    has_label_file = os.path.isfile(os.path.join(lbl_root, rel, stem + ".txt"))
            if no_label_only and has_label_file:
                continue
            if empty_only and boxes:
                continue
            if min_boxes > 0 and len(boxes) < min_boxes:
                continue
            if selected_classes:
                class_match = any(b[0] in selected_classes for b in boxes)
                if boxes and not class_match:
                    continue
            filtered.append((fp, boxes, w, h))

        self._filtered = filtered
        self._page = 0
        self._lbl_count.setText(f"{len(filtered)}장")
        self._refresh_gallery()

    def _refresh_gallery(self):
        # 기존 위젯 제거
        while self._gallery_layout.count():
            item = self._gallery_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        total_pages = max(1, math.ceil(len(self._filtered) / self._per_page))
        self._lbl_page.setText(f"{self._page + 1} / {total_pages}")

        start = self._page * self._per_page
        end = min(start + self._per_page, len(self._filtered))
        cols = 5

        for idx, i in enumerate(range(start, end)):
            fp, boxes, w, h = self._filtered[i]
            thumb = self._make_thumbnail(fp, boxes, w, h)
            lbl = _ThumbLabel(i)
            lbl.setPixmap(thumb)
            lbl.setToolTip(f"{os.path.basename(fp)}\n박스: {len(boxes)}개")
            lbl.clicked.connect(self._on_thumb_clicked)
            self._gallery_layout.addWidget(lbl, idx // cols, idx % cols)

    def _make_thumbnail(self, fp, boxes, w, h, size=156):
        img = cv2.imread(fp)
        if img is None:
            pix = QPixmap(size, size)
            pix.fill(QColor(40, 40, 40))
            return pix
        # 박스 그리기
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255)]
        for cid, cx, cy, bw, bh in boxes:
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            color = colors[cid % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # 리사이즈
        scale = size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _on_thumb_clicked(self, index):
        if index >= len(self._filtered):
            return
        fp, boxes, w, h = self._filtered[index]
        dlg = QDialog(self)
        dlg.setWindowTitle(os.path.basename(fp))
        dlg.resize(900, 700)
        lay = QVBoxLayout(dlg)

        # 이미지 표시
        img = cv2.imread(fp)
        if img is not None:
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            for cid, cx, cy, bw, bh in boxes:
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                color = colors[cid % len(colors)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, str(cid), (x1, max(y1 - 4, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(860, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_lbl = QLabel()
            img_lbl.setPixmap(pix)
            img_lbl.setAlignment(Qt.AlignCenter)
            lay.addWidget(img_lbl)

        # 박스 정보
        info_text = f"파일: {fp}\n크기: {w}×{h}\n박스: {len(boxes)}개"
        for cid, cx, cy, bw, bh in boxes:
            info_text += f"\n  class={cid}  center=({cx:.3f},{cy:.3f})  size=({bw:.3f}×{bh:.3f})"
        info_lbl = QLabel(info_text)
        info_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(info_lbl)
        dlg.exec()

    def _prev_page(self):
        if self._page > 0:
            self._page -= 1
            self._refresh_gallery()

    def _next_page(self):
        total_pages = max(1, math.ceil(len(self._filtered) / self._per_page))
        if self._page < total_pages - 1:
            self._page += 1
            self._refresh_gallery()

    def _show_charts(self):
        if not self._info:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            QMessageBox.warning(self, "알림", "matplotlib이 필요합니다.\npip install matplotlib")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("데이터셋 분포")
        dlg.resize(900, 700)
        lay = QVBoxLayout(dlg)

        fig = Figure(figsize=(10, 8))

        # 1) 클래스 분포
        ax1 = fig.add_subplot(2, 2, 1)
        cc = self._info["class_counts"]
        classes = sorted(cc.keys())
        counts = [cc[c] for c in classes]
        ax1.barh([str(c) for c in classes], counts, color="#4CAF50")
        ax1.set_title("클래스 분포")
        ax1.set_xlabel("박스 수")

        # 2) 박스 크기 분포 (w vs h)
        ax2 = fig.add_subplot(2, 2, 2)
        bs = self._info["box_sizes"]
        if len(bs) > 0:
            ax2.scatter(bs[:, 0], bs[:, 1], alpha=0.3, s=5, c="#2196F3")
            ax2.set_xlabel("Width (ratio)")
            ax2.set_ylabel("Height (ratio)")
        ax2.set_title("박스 크기 분포")

        # 3) 이미지당 박스 수 히스토그램
        ax3 = fig.add_subplot(2, 2, 3)
        box_per_img = [len(b) for _, b, _, _ in self._samples]
        ax3.hist(box_per_img, bins=30, color="#FF9800", edgecolor="white")
        ax3.set_title("이미지당 박스 수")
        ax3.set_xlabel("박스 수")
        ax3.set_ylabel("이미지 수")

        # 4) 종횡비 분포
        ax4 = fig.add_subplot(2, 2, 4)
        ba = self._info["box_aspects"]
        if len(ba) > 0:
            ax4.hist(ba, bins=50, color="#9C27B0", edgecolor="white")
        ax4.set_title("박스 종횡비 (W/H)")
        ax4.set_xlabel("종횡비")

        fig.tight_layout()
        canvas = FigureCanvasQTAgg(fig)
        lay.addWidget(canvas)
        dlg.exec()
