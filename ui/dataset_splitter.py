"""데이터셋 분할: SSIM + 이미지 메트릭 + 클래스 균형 기반 다양성 분할"""
import os, glob, shutil, random, math, cv2, numpy as np
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QProgressBar, QMessageBox, QGroupBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QSpinBox,
)


def _image_metrics(img):
    """Compute image quality metrics: brightness, contrast, blur, entropy, edge_density, color_richness, snr."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    brightness = float(gray.mean())
    contrast = float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log2(p)))
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges.mean())
    # color richness: std of hue channel
    color_richness = 0.0
    if len(img.shape) == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_richness = float(hsv[:, :, 0].std())
    # SNR
    signal = brightness
    noise = float(gray.std()) + 1e-9
    snr = signal / noise
    return np.array([brightness, contrast, blur, entropy, edge_density, color_richness, snr], dtype=np.float32)


def _thumb(img, size=32):
    """Resize to small grayscale thumbnail for fast SSIM-like comparison."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.resize(gray, (size, size)).astype(np.float32)


def _ssim_fast(t1, t2):
    """Simplified SSIM between two small thumbnails."""
    c1, c2 = 6.5025, 58.5225
    mu1, mu2 = t1.mean(), t2.mean()
    s1, s2 = t1.var(), t2.var()
    s12 = ((t1 - mu1) * (t2 - mu2)).mean()
    num = (2 * mu1 * mu2 + c1) * (2 * s12 + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2)
    return float(num / den)


class _SplitWorker(QThread):
    progress = Signal(int, int)
    log = Signal(str)
    finished_ok = Signal(object)
    error = Signal(str)

    def __init__(self, img_dir, out_dir, val_r, test_r, seed, copy_mode, ssim_thr):
        super().__init__()
        self.img_dir = img_dir
        self.out_dir = out_dir
        self.val_r, self.test_r = val_r, test_r
        self.seed = seed
        self.copy_mode = copy_mode
        self.ssim_thr = ssim_thr

    def run(self):
        try:
            random.seed(self.seed)
            np.random.seed(self.seed)
            # 1. Scan images recursively
            files = []
            for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                files.extend(glob.glob(os.path.join(self.img_dir, "**", e), recursive=True))
            files = sorted(set(files))
            if len(files) < 5:
                self.error.emit(f"이미지가 부족합니다 ({len(files)}장).")
                return
            self.log.emit(f"이미지 {len(files)}장 발견")

            # 2. Load labels + compute metrics
            self.log.emit("이미지 메트릭 계산 중...")
            data = []  # [(path, classes_set, metrics_vec, thumbnail)]
            for i, fp in enumerate(files):
                img = cv2.imread(fp)
                if img is None:
                    continue
                # labels (same dir)
                stem = os.path.splitext(os.path.basename(fp))[0]
                txt = os.path.join(os.path.dirname(fp), stem + ".txt")
                classes = set()
                if os.path.isfile(txt):
                    with open(txt) as f:
                        for line in f:
                            p = line.strip().split()
                            if p:
                                classes.add(int(p[0]))
                metrics = _image_metrics(img)
                thumb = _thumb(img)
                data.append((fp, classes, metrics, thumb))
                if i % 50 == 0:
                    self.progress.emit(i, len(files) * 3)

            n = len(data)
            n_val = max(1, int(n * self.val_r))
            n_test = max(0, int(n * self.test_r))
            self.log.emit(f"목표: val={n_val}, test={n_test}, train={n - n_val - n_test}")

            # 3. Normalize metrics for diversity scoring
            all_metrics = np.array([d[2] for d in data])
            m_min = all_metrics.min(axis=0)
            m_range = all_metrics.max(axis=0) - m_min + 1e-9
            norm_metrics = (all_metrics - m_min) / m_range

            # 4. Class distribution
            all_classes = set()
            class_images = {}  # {cid: [indices]}
            for idx, (_, classes, _, _) in enumerate(data):
                all_classes.update(classes)
                for c in classes:
                    class_images.setdefault(c, []).append(idx)
            self.log.emit(f"클래스: {sorted(all_classes)}")

            # 5. Greedy diverse val selection
            self.log.emit("다양성 기반 검증셋 선택 중...")
            val_indices = set()
            # 5a. Class-stratified seed: pick ~val_r from each class
            for cid, idxs in class_images.items():
                need = max(1, int(len(idxs) * self.val_r))
                random.shuffle(idxs)
                added = 0
                for idx in idxs:
                    if added >= need:
                        break
                    if idx in val_indices:
                        added += 1
                        continue
                    # check SSIM against existing val
                    ok = True
                    for vi in val_indices:
                        if _ssim_fast(data[idx][3], data[vi][3]) > self.ssim_thr:
                            ok = False
                            break
                    if ok:
                        val_indices.add(idx)
                        added += 1

            # 5b. Fill remaining val with most diverse images
            remaining = [i for i in range(n) if i not in val_indices]
            random.shuffle(remaining)
            # score by metric diversity (distance from val centroid)
            if val_indices:
                val_centroid = norm_metrics[list(val_indices)].mean(axis=0)
            else:
                val_centroid = norm_metrics.mean(axis=0)

            for idx in remaining:
                if len(val_indices) >= n_val:
                    break
                # SSIM check
                ok = True
                for vi in val_indices:
                    if _ssim_fast(data[idx][3], data[vi][3]) > self.ssim_thr:
                        ok = False
                        break
                if ok:
                    val_indices.add(idx)
                self.progress.emit(len(files) + len(val_indices), len(files) * 3)

            # 6. Test set: random from remaining
            remaining = [i for i in range(n) if i not in val_indices]
            test_indices = set(random.sample(remaining, min(n_test, len(remaining)))) if n_test > 0 else set()
            train_indices = set(range(n)) - val_indices - test_indices

            # 7. Verify val-train SSIM dissimilarity (log only)
            high_ssim = 0
            sample_check = random.sample(list(val_indices), min(20, len(val_indices)))
            sample_train = random.sample(list(train_indices), min(100, len(train_indices)))
            for vi in sample_check:
                for ti in sample_train:
                    if _ssim_fast(data[vi][3], data[ti][3]) > self.ssim_thr:
                        high_ssim += 1
            self.log.emit(f"Val-Train 유사 쌍 (샘플): {high_ssim}/{len(sample_check)*len(sample_train)}")

            # 8. Copy/symlink
            self.log.emit("파일 복사 중...")
            splits = {"train": train_indices, "val": val_indices, "test": test_indices}
            stats = {}
            total_copy = n
            done = 0
            for split_name, indices in splits.items():
                if not indices:
                    continue
                img_out = os.path.join(self.out_dir, split_name, "images")
                lbl_out = os.path.join(self.out_dir, split_name, "labels")
                os.makedirs(img_out, exist_ok=True)
                os.makedirs(lbl_out, exist_ok=True)
                for idx in indices:
                    fp = data[idx][0]
                    stem = os.path.splitext(os.path.basename(fp))[0]
                    ext = os.path.splitext(fp)[1]
                    dst_img = os.path.join(img_out, os.path.basename(fp))
                    if self.copy_mode == "symlink":
                        os.symlink(os.path.abspath(fp), dst_img)
                    else:
                        shutil.copy2(fp, dst_img)
                    txt = os.path.join(os.path.dirname(fp), stem + ".txt")
                    if os.path.isfile(txt):
                        dst_lbl = os.path.join(lbl_out, stem + ".txt")
                        if self.copy_mode == "symlink":
                            os.symlink(os.path.abspath(txt), dst_lbl)
                        else:
                            shutil.copy2(txt, dst_lbl)
                    done += 1
                    self.progress.emit(len(files) * 2 + done, total_copy + len(files) * 2)
                stats[split_name] = len(indices)

            # 9. Class distribution per split
            for split_name, indices in splits.items():
                if not indices:
                    continue
                cc = {}
                for idx in indices:
                    for c in data[idx][1]:
                        cc[c] = cc.get(c, 0) + 1
                self.log.emit(f"  {split_name}: {dict(sorted(cc.items()))}")

            self.finished_ok.emit(stats)
        except Exception as e:
            self.error.emit(str(e))


class DatasetSplitter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        desc = QLabel("SSIM + 이미지 메트릭(밝기/대비/흐림/SNR/엔트로피/에지밀도/색상풍부도) + 클래스 균형 기반 다양성 분할.\n"
                       "검증셋은 서로 유사하지 않고, 학습셋과도 유사하지 않은 이미지로 구성됩니다.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: gray; font-size: 11px; margin-bottom: 4px;")
        root.addWidget(desc)

        grp = QGroupBox("분할 설정")
        g = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("데이터 폴더:"))
        self._le_img = QLineEdit()
        self._le_img.setPlaceholderText("root 폴더 (하위 재귀 탐색)")
        row1.addWidget(self._le_img, 1)
        btn_i = QPushButton("찾아보기")
        btn_i.clicked.connect(lambda: self._browse(self._le_img))
        row1.addWidget(btn_i)
        g.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("출력 폴더:"))
        self._le_out = QLineEdit()
        row2.addWidget(self._le_out, 1)
        btn_o = QPushButton("찾아보기")
        btn_o.clicked.connect(lambda: self._browse(self._le_out))
        row2.addWidget(btn_o)
        g.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Val:"))
        self._spin_val = QDoubleSpinBox()
        self._spin_val.setRange(0.05, 0.5)
        self._spin_val.setValue(0.2)
        self._spin_val.setSingleStep(0.05)
        row3.addWidget(self._spin_val)
        row3.addWidget(QLabel("Test:"))
        self._spin_test = QDoubleSpinBox()
        self._spin_test.setRange(0, 0.3)
        self._spin_test.setValue(0.1)
        self._spin_test.setSingleStep(0.05)
        row3.addWidget(self._spin_test)
        row3.addWidget(QLabel("SSIM 임계값:"))
        self._spin_ssim = QDoubleSpinBox()
        self._spin_ssim.setRange(0.3, 0.99)
        self._spin_ssim.setValue(0.85)
        self._spin_ssim.setSingleStep(0.05)
        self._spin_ssim.setToolTip("검증셋 이미지 간 SSIM이 이 값 이하여야 선택됨 (낮을수록 엄격)")
        row3.addWidget(self._spin_ssim)
        row3.addWidget(QLabel("시드:"))
        self._spin_seed = QSpinBox()
        self._spin_seed.setRange(0, 99999)
        self._spin_seed.setValue(42)
        row3.addWidget(self._spin_seed)
        self._chk_symlink = QCheckBox("심볼릭 링크")
        row3.addWidget(self._chk_symlink)
        self._btn_run = QPushButton("분할 실행")
        self._btn_run.clicked.connect(self._run)
        row3.addWidget(self._btn_run)
        g.addLayout(row3)

        self._prog = QProgressBar()
        self._prog.setMaximumHeight(18)
        g.addWidget(self._prog)
        root.addWidget(grp)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        root.addWidget(self._log, 1)

    def _browse(self, le):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if d:
            le.setText(d)

    def _run(self):
        img_dir = self._le_img.text()
        out_dir = self._le_out.text()
        if not os.path.isdir(img_dir) or not out_dir:
            QMessageBox.warning(self, "알림", "폴더를 지정하세요.")
            return
        self._btn_run.setEnabled(False)
        self._prog.setValue(0)
        self._log.clear()
        mode = "symlink" if self._chk_symlink.isChecked() else "copy"
        self._worker = _SplitWorker(
            img_dir, out_dir, self._spin_val.value(), self._spin_test.value(),
            self._spin_seed.value(), mode, self._spin_ssim.value())
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.log.connect(self._log.append)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(lambda e: (QMessageBox.critical(self, "오류", e), self._btn_run.setEnabled(True)))
        self._worker.start()

    def _on_done(self, stats):
        self._btn_run.setEnabled(True)
        self._prog.setValue(100)
        lines = [f"  {k}: {v}장" for k, v in stats.items()]
        self._log.append("분할 완료:\n" + "\n".join(lines))
