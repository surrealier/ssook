"""스마트 샘플링: 클래스 균형, 랜덤, 층화 샘플링"""
import os, glob, shutil, random, math
from collections import Counter
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QProgressBar, QMessageBox, QGroupBox, QRadioButton,
    QButtonGroup, QSpinBox, QCheckBox, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView,
)


class _SampleWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(object)
    error = Signal(str)

    def __init__(self, img_dir, lbl_dir, out_dir, strategy, target, seed, include_labels):
        super().__init__()
        self.img_dir, self.lbl_dir, self.out_dir = img_dir, lbl_dir, out_dir
        self.strategy, self.target, self.seed = strategy, target, seed
        self.include_labels = include_labels

    def run(self):
        try:
            random.seed(self.seed)
            files = []
            for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                files.extend(glob.glob(os.path.join(self.img_dir, "**", e), recursive=True))
            files.sort()
            if not files:
                self.error.emit("이미지가 없습니다.")
                return

            # load class info per image
            img_classes = {}  # {img_path: set of class_ids}
            class_images = {}  # {class_id: [img_paths]}
            for fp in files:
                stem = os.path.splitext(os.path.basename(fp))[0]
                txt = os.path.join(self.lbl_dir, stem + ".txt")
                classes = set()
                if os.path.isfile(txt):
                    with open(txt) as f:
                        for line in f:
                            p = line.strip().split()
                            if p:
                                classes.add(int(p[0]))
                img_classes[fp] = classes
                for c in classes:
                    class_images.setdefault(c, []).append(fp)

            selected = set()
            if self.strategy == "random":
                selected = set(random.sample(files, min(self.target, len(files))))
            elif self.strategy == "stratified":
                # proportional sampling per class
                total_boxes = sum(len(v) for v in class_images.values())
                for cid, imgs in class_images.items():
                    n = max(1, int(self.target * len(imgs) / max(total_boxes, 1)))
                    selected.update(random.sample(imgs, min(n, len(imgs))))
                # fill remaining
                remaining = [f for f in files if f not in selected]
                need = self.target - len(selected)
                if need > 0 and remaining:
                    selected.update(random.sample(remaining, min(need, len(remaining))))
            elif self.strategy == "balance":
                if not class_images:
                    selected = set(random.sample(files, min(self.target, len(files))))
                else:
                    per_class = max(1, self.target // len(class_images))
                    for cid, imgs in class_images.items():
                        unique = list(set(imgs))
                        if len(unique) >= per_class:
                            selected.update(random.sample(unique, per_class))
                        else:
                            selected.update(unique)
                            # oversample
                            extra = per_class - len(unique)
                            selected.update(random.choices(unique, k=extra))

            # copy
            os.makedirs(os.path.join(self.out_dir, "images"), exist_ok=True)
            if self.include_labels:
                os.makedirs(os.path.join(self.out_dir, "labels"), exist_ok=True)
            selected = list(selected)
            for i, fp in enumerate(selected):
                shutil.copy2(fp, os.path.join(self.out_dir, "images", os.path.basename(fp)))
                if self.include_labels:
                    stem = os.path.splitext(os.path.basename(fp))[0]
                    txt = os.path.join(self.lbl_dir, stem + ".txt")
                    if os.path.isfile(txt):
                        shutil.copy2(txt, os.path.join(self.out_dir, "labels", stem + ".txt"))
                self.progress.emit(i + 1, len(selected))

            # stats
            after_classes = Counter()
            for fp in selected:
                for c in img_classes.get(fp, set()):
                    after_classes[c] += 1
            before_classes = {c: len(imgs) for c, imgs in class_images.items()}
            self.finished_ok.emit(dict(total=len(selected), before=before_classes, after=dict(after_classes)))
        except Exception as e:
            self.error.emit(str(e))


class SmartSampler(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        desc = QLabel("스마트 샘플링: 대규모 데이터셋에서 목적에 맞는 부분집합을 추출합니다.\n"
                       "• 클래스 균형: 소수 클래스를 오버샘플링하여 균등 분포 생성\n"
                       "• 랜덤: 지정 수만큼 무작위 추출 (시드 고정 가능)\n"
                       "• 층화: 원본 클래스 비율을 유지하면서 축소 샘플링")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: gray; font-size: 11px; margin-bottom: 4px;")
        root.addWidget(desc)
        # 입력
        for label, attr in [("이미지 폴더:", "_le_img"), ("라벨 폴더:", "_le_lbl"), ("출력 폴더:", "_le_out")]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            le = QLineEdit()
            setattr(self, attr, le)
            row.addWidget(le, 1)
            btn = QPushButton("찾아보기")
            btn.clicked.connect(lambda _, l=le: self._browse(l))
            row.addWidget(btn)
            root.addLayout(row)

        # 전략
        g = QGroupBox("샘플링 전략")
        gl = QHBoxLayout(g)
        self._bg = QButtonGroup(self)
        for i, (text, key) in enumerate([("클래스 균형", "balance"), ("랜덤", "random"), ("층화", "stratified")]):
            rb = QRadioButton(text)
            rb.setProperty("key", key)
            if i == 0:
                rb.setChecked(True)
            self._bg.addButton(rb)
            gl.addWidget(rb)
        gl.addWidget(QLabel("목표 수:"))
        self._spin_target = QSpinBox()
        self._spin_target.setRange(1, 999999)
        self._spin_target.setValue(1000)
        gl.addWidget(self._spin_target)
        gl.addWidget(QLabel("시드:"))
        self._spin_seed = QSpinBox()
        self._spin_seed.setRange(0, 99999)
        self._spin_seed.setValue(42)
        gl.addWidget(self._spin_seed)
        self._chk_labels = QCheckBox("라벨 포함")
        self._chk_labels.setChecked(True)
        gl.addWidget(self._chk_labels)
        self._btn_run = QPushButton("실행")
        self._btn_run.clicked.connect(self._run)
        gl.addWidget(self._btn_run)
        root.addWidget(g)

        self._prog = QProgressBar()
        self._prog.setMaximumHeight(18)
        root.addWidget(self._prog)

        # 결과
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["클래스", "변환 전", "변환 후"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        root.addWidget(self._table, 1)
        self._lbl_summary = QLabel("")
        root.addWidget(self._lbl_summary)

    def _browse(self, le):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if d:
            le.setText(d)

    def _run(self):
        img = self._le_img.text()
        lbl = self._le_lbl.text() or img
        out = self._le_out.text()
        if not os.path.isdir(img) or not out:
            QMessageBox.warning(self, "알림", "폴더를 지정하세요.")
            return
        strategy = self._bg.checkedButton().property("key")
        self._btn_run.setEnabled(False)
        self._worker = _SampleWorker(img, lbl, out, strategy,
                                      self._spin_target.value(), self._spin_seed.value(),
                                      self._chk_labels.isChecked())
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(lambda e: (QMessageBox.critical(self, "오류", e), self._btn_run.setEnabled(True)))
        self._worker.start()

    def _on_done(self, stats):
        self._btn_run.setEnabled(True)
        self._prog.setValue(100)
        all_classes = sorted(set(list(stats["before"].keys()) + list(stats["after"].keys())))
        self._table.setRowCount(len(all_classes))
        for i, c in enumerate(all_classes):
            self._table.setItem(i, 0, QTableWidgetItem(str(c)))
            self._table.setItem(i, 1, QTableWidgetItem(str(stats["before"].get(c, 0))))
            self._table.setItem(i, 2, QTableWidgetItem(str(stats["after"].get(c, 0))))
        self._lbl_summary.setText(f"샘플링 완료: {stats['total']}장")
