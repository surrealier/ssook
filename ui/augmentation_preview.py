"""Augmentation Preview + Batch Apply + Quick YOLO Training Test"""
import os, glob, math, random, cv2, numpy as np, shutil, subprocess, json
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QProgressBar, QMessageBox, QGroupBox, QCheckBox,
    QSlider, QSpinBox, QSplitter, QTextEdit, QTabWidget, QDoubleSpinBox,
)
from PySide6.QtGui import QImage, QPixmap

try:
    import albumentations as A
    HAS_ALB = True
except ImportError:
    HAS_ALB = False


def _mosaic4(images, labels_list, target_size=640):
    """Create a 2x2 mosaic from 4 images with label adjustment."""
    s = target_size
    out = np.full((s, s, 3), 114, dtype=np.uint8)
    cx, cy = s // 2, s // 2
    positions = [(0, 0, cx, cy), (cx, 0, s, cy), (0, cy, cx, s), (cx, cy, s, s)]
    out_boxes = []
    for i, (img, boxes) in enumerate(zip(images, labels_list)):
        x1, y1, x2, y2 = positions[i]
        pw, ph = x2 - x1, y2 - y1
        h, w = img.shape[:2]
        scale = min(pw / w, ph / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh))
        out[y1:y1 + nh, x1:x1 + nw] = resized
        for cid, bcx, bcy, bw, bh in boxes:
            abs_cx = x1 + bcx * nw
            abs_cy = y1 + bcy * nh
            abs_w = bw * nw
            abs_h = bh * nh
            out_boxes.append((cid, abs_cx / s, abs_cy / s, abs_w / s, abs_h / s))
    return out, out_boxes


def _apply_cv_aug(img, params, boxes=None):
    """Apply augmentations with OpenCV. boxes: [(cid, cx, cy, w, h)] normalized."""
    h, w = img.shape[:2]
    out = img.copy()
    out_boxes = list(boxes) if boxes else []
    if params.get("hflip"):
        out = cv2.flip(out, 1)
        out_boxes = [(c, 1 - cx, cy, bw, bh) for c, cx, cy, bw, bh in out_boxes]
    if params.get("vflip"):
        out = cv2.flip(out, 0)
        out_boxes = [(c, cx, 1 - cy, bw, bh) for c, cx, cy, bw, bh in out_boxes]
    angle = params.get("rotation", 0)
    if angle != 0:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += (nw - w) / 2; M[1, 2] += (nh - h) / 2
        out = cv2.warpAffine(out, M, (nw, nh))
        new_boxes = []
        for c, cx, cy, bw, bh in out_boxes:
            px, py = cx * w, cy * h
            nx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
            ny = M[1, 0] * px + M[1, 1] * py + M[1, 2]
            new_boxes.append((c, nx / nw, ny / nh, bw * w / nw, bh * h / nh))
        out_boxes = new_boxes
    bright = params.get("brightness", 0)
    if bright != 0:
        out = np.clip(out.astype(np.int16) + bright, 0, 255).astype(np.uint8)
    contrast = params.get("contrast", 1.0)
    if contrast != 1.0:
        out = np.clip(out.astype(np.float32) * contrast, 0, 255).astype(np.uint8)
    blur_k = params.get("blur", 0)
    if blur_k > 0:
        k = blur_k * 2 + 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    noise_s = params.get("noise", 0)
    if noise_s > 0:
        out = np.clip(out.astype(np.int16) + np.random.normal(0, noise_s, out.shape).astype(np.int16), 0, 255).astype(np.uint8)
    cutout_n = params.get("cutout_n", 0)
    cutout_pct = params.get("cutout_pct", 20)
    if cutout_n > 0:
        oh, ow = out.shape[:2]
        ch, cw = int(oh * cutout_pct / 100), int(ow * cutout_pct / 100)
        for _ in range(cutout_n):
            y, x = np.random.randint(0, max(oh - ch, 1)), np.random.randint(0, max(ow - cw, 1))
            out[y:y + ch, x:x + cw] = 0
    return out, out_boxes


def _apply_alb_aug(img, params):
    """Apply albumentations augmentations (no bbox transform)."""
    if not HAS_ALB:
        return img
    transforms = []
    if params.get("clahe"): transforms.append(A.CLAHE(p=1.0))
    if params.get("equalize"): transforms.append(A.Equalize(p=1.0))
    if params.get("sharpen"): transforms.append(A.Sharpen(p=1.0))
    if params.get("emboss"): transforms.append(A.Emboss(p=1.0))
    if params.get("hsv_shift"): transforms.append(A.HueSaturationValue(p=1.0))
    if params.get("channel_shuffle"): transforms.append(A.ChannelShuffle(p=1.0))
    if params.get("rgb_shift"): transforms.append(A.RGBShift(p=1.0))
    if params.get("posterize"): transforms.append(A.Posterize(p=1.0))
    if params.get("solarize"): transforms.append(A.Solarize(p=1.0))
    if params.get("iso_noise"): transforms.append(A.ISONoise(p=1.0))
    if params.get("motion_blur"):
        k = params.get("motion_blur_k", 7)
        transforms.append(A.MotionBlur(blur_limit=k, p=1.0))
    if params.get("median_blur"):
        transforms.append(A.MedianBlur(blur_limit=7, p=1.0))
    if not transforms:
        return img
    aug = A.Compose(transforms)
    return aug(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["image"]


class _AugWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(object)
    error = Signal(str)

    def __init__(self, img_dir, out_dir, params, multiplier, use_alb, alb_params, mosaic):
        super().__init__()
        self.img_dir, self.out_dir = img_dir, out_dir
        self.params, self.multiplier = params, multiplier
        self.use_alb, self.alb_params, self.mosaic = use_alb, alb_params, mosaic

    def run(self):
        try:
            files = []
            for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                files.extend(glob.glob(os.path.join(self.img_dir, "**", e), recursive=True))
            files = sorted(set(files))
            os.makedirs(os.path.join(self.out_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "labels"), exist_ok=True)
            count = 0
            total = len(files) * self.multiplier
            for fp in files:
                stem = os.path.splitext(os.path.basename(fp))[0]
                img = cv2.imread(fp)
                if img is None: continue
                boxes = []
                txt = os.path.join(os.path.dirname(fp), stem + ".txt")
                if os.path.isfile(txt):
                    with open(txt) as f:
                        for line in f:
                            p = line.strip().split()
                            if len(p) >= 5:
                                boxes.append((int(p[0]), *[float(x) for x in p[1:5]]))
                for m in range(self.multiplier):
                    aug_img, aug_boxes = _apply_cv_aug(img, self.params, boxes)
                    if self.use_alb and HAS_ALB:
                        aug_rgb = _apply_alb_aug(aug_img, self.alb_params)
                        aug_img = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR) if len(aug_rgb.shape) == 3 else aug_rgb
                    suffix = f"_aug{m}" if self.multiplier > 1 else "_aug"
                    out_name = stem + suffix
                    cv2.imwrite(os.path.join(self.out_dir, "images", out_name + ".jpg"), aug_img)
                    with open(os.path.join(self.out_dir, "labels", out_name + ".txt"), "w") as f:
                        for c, cx, cy, bw, bh in aug_boxes:
                            f.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    count += 1
                    self.progress.emit(count, total)
            # mosaic
            if self.mosaic and len(files) >= 4:
                random.shuffle(files)
                n_mosaic = len(files) // 4
                for mi in range(n_mosaic):
                    imgs, lbls = [], []
                    for fi in range(4):
                        fp = files[mi * 4 + fi]
                        im = cv2.imread(fp)
                        if im is None: im = np.full((640, 640, 3), 114, dtype=np.uint8)
                        imgs.append(im)
                        stem = os.path.splitext(os.path.basename(fp))[0]
                        txt = os.path.join(os.path.dirname(fp), stem + ".txt")
                        bx = []
                        if os.path.isfile(txt):
                            with open(txt) as f:
                                for line in f:
                                    p = line.strip().split()
                                    if len(p) >= 5:
                                        bx.append((int(p[0]), *[float(x) for x in p[1:5]]))
                        lbls.append(bx)
                    mos_img, mos_boxes = _mosaic4(imgs, lbls)
                    out_name = f"mosaic_{mi:04d}"
                    cv2.imwrite(os.path.join(self.out_dir, "images", out_name + ".jpg"), mos_img)
                    with open(os.path.join(self.out_dir, "labels", out_name + ".txt"), "w") as f:
                        for c, cx, cy, bw, bh in mos_boxes:
                            f.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    count += 1
            self.finished_ok.emit(count)
        except Exception as e:
            self.error.emit(str(e))


class _YoloTrainWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(object)
    error = Signal(str)

    def __init__(self, data_yaml, epochs, imgsz):
        super().__init__()
        self.data_yaml, self.epochs, self.imgsz = data_yaml, epochs, imgsz

    def run(self):
        try:
            from ultralytics import YOLO
            model = YOLO("yolo11n.pt")
            self.log.emit("YOLO 학습 시작...")
            results = model.train(data=self.data_yaml, epochs=self.epochs,
                                  imgsz=self.imgsz, batch=8, device="cpu",
                                  verbose=False, exist_ok=True)
            metrics = model.val()
            result = {
                "mAP50": float(metrics.box.map50),
                "mAP50_95": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
            }
            self.finished_ok.emit(result)
        except ImportError:
            self.error.emit("ultralytics가 설치되지 않았습니다.\npip install ultralytics")
        except Exception as e:
            self.error.emit(str(e))


class AugmentationPreview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._img = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        tabs = QTabWidget()

        # === Tab 1: Preview & Batch ===
        preview_tab = QWidget()
        pl = QVBoxLayout(preview_tab)
        top = QHBoxLayout()
        top.addWidget(QLabel("Image:"))
        self._le_img = QLineEdit()
        top.addWidget(self._le_img, 1)
        b1 = QPushButton("File")
        b1.clicked.connect(self._browse_file)
        top.addWidget(b1)
        b2 = QPushButton("Folder")
        b2.clicked.connect(lambda: self._browse_dir(self._le_img))
        top.addWidget(b2)
        pl.addLayout(top)

        sp = QSplitter(Qt.Horizontal)
        # Controls
        left = QWidget()
        left.setFixedWidth(300)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        # CV augmentations
        g1 = QGroupBox("Basic (OpenCV)")
        g1l = QVBoxLayout(g1)
        self._cv_ui = {}
        for name in ["H-Flip", "V-Flip"]:
            cb = QCheckBox(name)
            cb.stateChanged.connect(self._update_preview)
            g1l.addWidget(cb)
            self._cv_ui[name] = cb
        for name, lo, hi, default, div in [
            ("Rotation", -45, 45, 0, 1), ("Brightness", -50, 50, 0, 1),
            ("Contrast", 5, 20, 10, 10), ("Gaussian Blur", 0, 15, 0, 1),
            ("Gaussian Noise", 0, 50, 0, 1), ("Cutout Count", 0, 5, 0, 1),
            ("Cutout Size%", 5, 50, 20, 1)]:
            row = QHBoxLayout()
            row.addWidget(QLabel(name))
            s = QSlider(Qt.Horizontal); s.setRange(lo, hi); s.setValue(default)
            lbl = QLabel(str(default / div if div > 1 else default))
            s.valueChanged.connect(lambda v, l=lbl, d=div: l.setText(f"{v/d:.1f}" if d > 1 else str(v)))
            s.valueChanged.connect(self._update_preview)
            row.addWidget(s, 1); row.addWidget(lbl)
            g1l.addLayout(row)
            self._cv_ui[name] = s
        ll.addWidget(g1)

        # Albumentations
        g2 = QGroupBox(f"Advanced (Albumentations {'✓' if HAS_ALB else '✗ pip install albumentations'})")
        g2l = QVBoxLayout(g2)
        self._alb_ui = {}
        for name in ["CLAHE", "Equalize", "Sharpen", "Emboss", "HSV Shift",
                      "Channel Shuffle", "RGB Shift", "Posterize", "Solarize",
                      "ISO Noise", "Motion Blur", "Median Blur"]:
            cb = QCheckBox(name)
            cb.setEnabled(HAS_ALB)
            cb.stateChanged.connect(self._update_preview)
            g2l.addWidget(cb)
            self._alb_ui[name] = cb
        ll.addWidget(g2)

        # Mosaic + Batch
        g3 = QGroupBox("Batch")
        g3l = QVBoxLayout(g3)
        self._chk_mosaic = QCheckBox("Mosaic4")
        g3l.addWidget(self._chk_mosaic)
        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("Output:"))
        self._le_out = QLineEdit()
        row_out.addWidget(self._le_out, 1)
        bo = QPushButton("Browse")
        bo.clicked.connect(lambda: self._browse_dir(self._le_out))
        row_out.addWidget(bo)
        g3l.addLayout(row_out)
        row_m = QHBoxLayout()
        row_m.addWidget(QLabel("Multiplier:"))
        self._spin_mult = QSpinBox(); self._spin_mult.setRange(1, 10); self._spin_mult.setValue(1)
        row_m.addWidget(self._spin_mult)
        self._btn_apply = QPushButton("Apply Batch")
        self._btn_apply.clicked.connect(self._run_batch)
        row_m.addWidget(self._btn_apply)
        g3l.addLayout(row_m)
        self._prog = QProgressBar(); self._prog.setMaximumHeight(18)
        g3l.addWidget(self._prog)
        ll.addWidget(g3)
        ll.addStretch()
        sp.addWidget(left)

        # Preview
        right = QWidget()
        rl = QHBoxLayout(right)
        self._lbl_orig = QLabel("Original"); self._lbl_orig.setAlignment(Qt.AlignCenter); self._lbl_orig.setMinimumSize(300, 300)
        rl.addWidget(self._lbl_orig, 1)
        self._lbl_aug = QLabel("Augmented"); self._lbl_aug.setAlignment(Qt.AlignCenter); self._lbl_aug.setMinimumSize(300, 300)
        rl.addWidget(self._lbl_aug, 1)
        sp.addWidget(right); sp.setStretchFactor(1, 1)
        pl.addWidget(sp, 1)
        tabs.addTab(preview_tab, "Preview & Batch")

        # === Tab 2: Quick YOLO Test ===
        yolo_tab = QWidget()
        yl = QVBoxLayout(yolo_tab)
        yl.addWidget(QLabel("데이터셋에서 소량 추출 → 증강 적용 → YOLO 학습 → 성능 비교\n"
                            "(ultralytics, yolo11n.pt 기반)"))
        row_ds = QHBoxLayout()
        row_ds.addWidget(QLabel("Dataset:"))
        self._le_yolo_ds = QLineEdit(); self._le_yolo_ds.setPlaceholderText("root 폴더 (재귀 탐색)")
        row_ds.addWidget(self._le_yolo_ds, 1)
        bds = QPushButton("Browse"); bds.clicked.connect(lambda: self._browse_dir(self._le_yolo_ds))
        row_ds.addWidget(bds)
        yl.addLayout(row_ds)
        row_yp = QHBoxLayout()
        row_yp.addWidget(QLabel("Sample:"))
        self._spin_sample = QSpinBox(); self._spin_sample.setRange(10, 500); self._spin_sample.setValue(50)
        row_yp.addWidget(self._spin_sample)
        row_yp.addWidget(QLabel("Epochs:"))
        self._spin_epochs = QSpinBox(); self._spin_epochs.setRange(1, 50); self._spin_epochs.setValue(5)
        row_yp.addWidget(self._spin_epochs)
        row_yp.addWidget(QLabel("Seed:"))
        self._spin_yseed = QSpinBox(); self._spin_yseed.setRange(0, 99999); self._spin_yseed.setValue(42)
        row_yp.addWidget(self._spin_yseed)
        self._btn_yolo_base = QPushButton("Baseline (증강 없이)")
        self._btn_yolo_base.clicked.connect(lambda: self._run_yolo(False))
        row_yp.addWidget(self._btn_yolo_base)
        self._btn_yolo_aug = QPushButton("With Augmentation")
        self._btn_yolo_aug.clicked.connect(lambda: self._run_yolo(True))
        row_yp.addWidget(self._btn_yolo_aug)
        yl.addLayout(row_yp)
        self._yolo_prog = QProgressBar(); self._yolo_prog.setMaximumHeight(18)
        yl.addWidget(self._yolo_prog)
        self._yolo_log = QTextEdit(); self._yolo_log.setReadOnly(True)
        yl.addWidget(self._yolo_log, 1)
        tabs.addTab(yolo_tab, "Quick YOLO Test")

        root.addWidget(tabs)

    # --- helpers ---
    def _browse_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Image", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if f:
            self._le_img.setText(f); self._load_image(f)

    def _browse_dir(self, le):
        d = QFileDialog.getExistingDirectory(self, "Folder")
        if d:
            le.setText(d)
            if le is self._le_img:
                for e in ("*.jpg", "*.png"):
                    files = glob.glob(os.path.join(d, "**", e), recursive=True)
                    if files: self._load_image(sorted(files)[0]); break

    def _load_image(self, path):
        self._img = cv2.imread(path)
        if self._img is not None:
            self._show(self._lbl_orig, self._img); self._update_preview()

    def _get_cv_params(self):
        ui = self._cv_ui
        return dict(hflip=ui["H-Flip"].isChecked(), vflip=ui["V-Flip"].isChecked(),
                    rotation=ui["Rotation"].value(), brightness=ui["Brightness"].value(),
                    contrast=ui["Contrast"].value() / 10, blur=ui["Gaussian Blur"].value(),
                    noise=ui["Gaussian Noise"].value(), cutout_n=ui["Cutout Count"].value(),
                    cutout_pct=ui["Cutout Size%"].value())

    def _get_alb_params(self):
        key_map = {"CLAHE": "clahe", "Equalize": "equalize", "Sharpen": "sharpen",
                   "Emboss": "emboss", "HSV Shift": "hsv_shift", "Channel Shuffle": "channel_shuffle",
                   "RGB Shift": "rgb_shift", "Posterize": "posterize", "Solarize": "solarize",
                   "ISO Noise": "iso_noise", "Motion Blur": "motion_blur", "Median Blur": "median_blur"}
        return {v: self._alb_ui[k].isChecked() for k, v in key_map.items()}

    def _update_preview(self, _=None):
        if self._img is None: return
        aug, _ = _apply_cv_aug(self._img, self._get_cv_params())
        if HAS_ALB and any(self._get_alb_params().values()):
            aug_rgb = _apply_alb_aug(aug, self._get_alb_params())
            aug = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR) if len(aug_rgb.shape) == 3 else aug_rgb
        self._show(self._lbl_aug, aug)

    def _show(self, lbl, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        lbl.setPixmap(QPixmap.fromImage(qimg).scaled(
            max(lbl.width() - 10, 100), max(lbl.height() - 10, 100), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _run_batch(self):
        img_dir = self._le_img.text()
        out_dir = self._le_out.text()
        if not out_dir: QMessageBox.warning(self, "알림", "출력 폴더를 지정하세요."); return
        if os.path.isfile(img_dir): img_dir = os.path.dirname(img_dir)
        if not os.path.isdir(img_dir): QMessageBox.warning(self, "알림", "이미지 폴더를 확인하세요."); return
        self._btn_apply.setEnabled(False)
        self._worker = _AugWorker(img_dir, out_dir, self._get_cv_params(), self._spin_mult.value(),
                                   HAS_ALB and any(self._get_alb_params().values()),
                                   self._get_alb_params(), self._chk_mosaic.isChecked())
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.finished_ok.connect(lambda n: (self._btn_apply.setEnabled(True), self._prog.setValue(100),
                                                     QMessageBox.information(self, "Done", f"{n}장 증강 완료")))
        self._worker.error.connect(lambda e: (QMessageBox.critical(self, "Error", e), self._btn_apply.setEnabled(True)))
        self._worker.start()

    # --- Quick YOLO Test ---
    def _run_yolo(self, with_aug):
        ds_dir = self._le_yolo_ds.text()
        if not os.path.isdir(ds_dir): QMessageBox.warning(self, "알림", "데이터셋 폴더를 선택하세요."); return
        self._yolo_log.clear()
        seed = self._spin_yseed.value()
        n_sample = self._spin_sample.value()
        random.seed(seed)
        # collect images
        files = []
        for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files.extend(glob.glob(os.path.join(ds_dir, "**", e), recursive=True))
        files = sorted(set(files))
        if len(files) < 10: QMessageBox.warning(self, "알림", f"이미지 부족 ({len(files)}장)"); return
        sampled = random.sample(files, min(n_sample, len(files)))
        self._yolo_log.append(f"샘플: {len(sampled)}장 / {len(files)}장")

        # split 70/20/10
        random.shuffle(sampled)
        n_tr = int(len(sampled) * 0.7)
        n_va = int(len(sampled) * 0.2)
        splits = {"train": sampled[:n_tr], "val": sampled[n_tr:n_tr + n_va], "test": sampled[n_tr + n_va:]}

        # prepare temp dir
        import tempfile
        tmp = tempfile.mkdtemp(prefix="yolo_test_")
        class_names = set()
        for split_name, split_files in splits.items():
            img_out = os.path.join(tmp, split_name, "images")
            lbl_out = os.path.join(tmp, split_name, "labels")
            os.makedirs(img_out, exist_ok=True); os.makedirs(lbl_out, exist_ok=True)
            for fp in split_files:
                stem = os.path.splitext(os.path.basename(fp))[0]
                shutil.copy2(fp, os.path.join(img_out, os.path.basename(fp)))
                txt = os.path.join(os.path.dirname(fp), stem + ".txt")
                if os.path.isfile(txt):
                    shutil.copy2(txt, os.path.join(lbl_out, stem + ".txt"))
                    with open(txt) as f:
                        for line in f:
                            p = line.strip().split()
                            if p: class_names.add(int(p[0]))
                # augment train only
                if with_aug and split_name == "train":
                    img = cv2.imread(fp)
                    if img is not None:
                        aug_img, _ = _apply_cv_aug(img, self._get_cv_params())
                        cv2.imwrite(os.path.join(img_out, stem + "_aug.jpg"), aug_img)
                        if os.path.isfile(txt):
                            shutil.copy2(txt, os.path.join(lbl_out, stem + "_aug.txt"))

        # dataset.yaml
        names_dict = {i: str(i) for i in sorted(class_names)}
        yaml_content = f"path: {tmp}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        yaml_content += f"nc: {len(names_dict)}\nnames: {names_dict}\n"
        yaml_path = os.path.join(tmp, "dataset.yaml")
        with open(yaml_path, "w") as f: f.write(yaml_content)
        self._yolo_log.append(f"{'With Aug' if with_aug else 'Baseline'} | classes: {len(names_dict)} | dir: {tmp}")

        # train
        self._btn_yolo_base.setEnabled(False); self._btn_yolo_aug.setEnabled(False)
        self._yolo_worker = _YoloTrainWorker(yaml_path, self._spin_epochs.value(), 640)
        self._yolo_worker.log.connect(self._yolo_log.append)
        self._yolo_worker.finished_ok.connect(self._on_yolo_done)
        self._yolo_worker.error.connect(lambda e: (self._yolo_log.append(f"오류: {e}"),
                                                    self._btn_yolo_base.setEnabled(True),
                                                    self._btn_yolo_aug.setEnabled(True)))
        self._yolo_worker.start()

    def _on_yolo_done(self, result):
        self._btn_yolo_base.setEnabled(True); self._btn_yolo_aug.setEnabled(True)
        self._yolo_log.append(f"결과: mAP@50={result['mAP50']:.4f} | mAP@50:95={result['mAP50_95']:.4f} | "
                              f"P={result['precision']:.4f} | R={result['recall']:.4f}")
