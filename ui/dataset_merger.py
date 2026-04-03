"""데이터셋 병합: 여러 데이터셋 통합 + dHash 중복 제거"""
import os, glob, shutil, cv2, numpy as np
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QProgressBar, QMessageBox, QGroupBox, QSlider,
    QTextEdit, QListWidget, QListWidgetItem,
)


def _compute_dhash(path, size=8):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    r = cv2.resize(img, (size + 1, size))
    return (r[:, 1:] > r[:, :-1]).flatten()


class _MergeWorker(QThread):
    progress = Signal(int, int)
    log = Signal(str)
    finished_ok = Signal(object)
    error = Signal(str)

    def __init__(self, datasets, out_dir, dedup, threshold):
        super().__init__()
        self.datasets = datasets  # [(img_dir, lbl_dir, prefix)]
        self.out_dir = out_dir
        self.dedup = dedup
        self.threshold = threshold

    def run(self):
        try:
            os.makedirs(os.path.join(self.out_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "labels"), exist_ok=True)
            # collect all files
            all_files = []
            for di, (img_dir, lbl_dir, prefix) in enumerate(self.datasets):
                for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                    for fp in sorted(glob.glob(os.path.join(img_dir, "**", e), recursive=True)):
                        all_files.append((fp, lbl_dir, prefix))
            self.log.emit(f"전체 이미지: {len(all_files)}장")

            # dedup
            hashes = {}
            skip = set()
            if self.dedup:
                self.log.emit("중복 검사 중...")
                for i, (fp, _, _) in enumerate(all_files):
                    h = _compute_dhash(fp)
                    if h is not None:
                        for prev_fp, prev_h in hashes.values():
                            if int(np.count_nonzero(h != prev_h)) <= self.threshold:
                                skip.add(fp)
                                break
                        if fp not in skip:
                            hashes[fp] = (fp, h)
                    self.progress.emit(i + 1, len(all_files) * 2)
                self.log.emit(f"중복 제거: {len(skip)}장")

            # copy
            copied = 0
            total = len(all_files)
            for i, (fp, lbl_dir, prefix) in enumerate(all_files):
                if fp in skip:
                    continue
                stem = os.path.splitext(os.path.basename(fp))[0]
                ext = os.path.splitext(fp)[1]
                out_name = f"{prefix}_{stem}" if prefix else stem
                # handle conflict
                out_img = os.path.join(self.out_dir, "images", out_name + ext)
                n = 0
                while os.path.exists(out_img):
                    n += 1
                    out_img = os.path.join(self.out_dir, "images", f"{out_name}_{n}{ext}")
                    out_name_final = f"{out_name}_{n}"
                else:
                    out_name_final = out_name if n == 0 else f"{out_name}_{n}"
                shutil.copy2(fp, out_img)
                # label
                txt = os.path.join(lbl_dir, stem + ".txt")
                if os.path.isfile(txt):
                    shutil.copy2(txt, os.path.join(self.out_dir, "labels", out_name_final + ".txt"))
                copied += 1
                offset = len(all_files) if self.dedup else 0
                self.progress.emit(offset + i + 1, offset + total)
            self.finished_ok.emit(dict(total=len(all_files), copied=copied, skipped=len(skip)))
        except Exception as e:
            self.error.emit(str(e))


class DatasetMerger(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._datasets = []  # [(img_dir, lbl_dir)]
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        # 데이터셋 목록
        g = QGroupBox("데이터셋 목록")
        gl = QVBoxLayout(g)
        self._ds_list = QListWidget()
        gl.addWidget(self._ds_list)
        row_btn = QHBoxLayout()
        self._btn_add = QPushButton("데이터셋 추가")
        self._btn_add.clicked.connect(self._add_dataset)
        row_btn.addWidget(self._btn_add)
        self._btn_remove = QPushButton("선택 제거")
        self._btn_remove.clicked.connect(self._remove_dataset)
        row_btn.addWidget(self._btn_remove)
        row_btn.addStretch()
        gl.addLayout(row_btn)
        root.addWidget(g)

        # 설정
        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("출력 폴더:"))
        self._le_out = QLineEdit()
        row_out.addWidget(self._le_out, 1)
        bo = QPushButton("찾아보기")
        bo.clicked.connect(lambda: self._browse(self._le_out))
        row_out.addWidget(bo)
        root.addLayout(row_out)

        row_opt = QHBoxLayout()
        from PySide6.QtWidgets import QCheckBox
        self._chk_dedup = QCheckBox("중복 제거")
        self._chk_dedup.setChecked(True)
        row_opt.addWidget(self._chk_dedup)
        row_opt.addWidget(QLabel("거리 임계값:"))
        self._sl_thr = QSlider(Qt.Horizontal)
        self._sl_thr.setRange(0, 20)
        self._sl_thr.setValue(10)
        self._sl_thr.setFixedWidth(100)
        self._lbl_thr = QLabel("10")
        self._sl_thr.valueChanged.connect(lambda v: self._lbl_thr.setText(str(v)))
        row_opt.addWidget(self._sl_thr)
        row_opt.addWidget(self._lbl_thr)
        row_opt.addStretch()
        self._btn_run = QPushButton("병합")
        self._btn_run.clicked.connect(self._run)
        row_opt.addWidget(self._btn_run)
        root.addLayout(row_opt)

        self._prog = QProgressBar()
        self._prog.setMaximumHeight(18)
        root.addWidget(self._prog)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        root.addWidget(self._log, 1)

    def _browse(self, le):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if d:
            le.setText(d)

    def _add_dataset(self):
        img_dir = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택")
        if not img_dir:
            return
        lbl_dir = QFileDialog.getExistingDirectory(self, "라벨 폴더 선택 (취소=이미지와 동일)")
        if not lbl_dir:
            lbl_dir = img_dir
        idx = len(self._datasets)
        self._datasets.append((img_dir, lbl_dir))
        self._ds_list.addItem(f"[d{idx}] {img_dir} | {lbl_dir}")

    def _remove_dataset(self):
        row = self._ds_list.currentRow()
        if row >= 0:
            self._datasets.pop(row)
            self._ds_list.takeItem(row)

    def _run(self):
        if not self._datasets or not self._le_out.text():
            QMessageBox.warning(self, "알림", "데이터셋과 출력 폴더를 지정하세요.")
            return
        self._btn_run.setEnabled(False)
        self._log.clear()
        ds = [(img, lbl, f"d{i}") for i, (img, lbl) in enumerate(self._datasets)]
        self._worker = _MergeWorker(ds, self._le_out.text(),
                                     self._chk_dedup.isChecked(), self._sl_thr.value())
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.log.connect(self._log.append)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(lambda e: (self._log.append(f"오류: {e}"), self._btn_run.setEnabled(True)))
        self._worker.start()

    def _on_done(self, stats):
        self._btn_run.setEnabled(True)
        self._prog.setValue(100)
        self._log.append(f"병합 완료: {stats['copied']}장 복사 (전체 {stats['total']}장, 중복 제거 {stats['skipped']}장)")
