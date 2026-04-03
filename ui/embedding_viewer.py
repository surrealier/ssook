"""Embedding 추출 & t-SNE/UMAP 2D 시각화 탭"""
import os
import glob

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QComboBox, QGroupBox, QProgressBar,
    QMessageBox, QSpinBox, QCheckBox,
)

from core.model_loader import load_model
from core.inference import preprocess, preprocess_classification, letterbox


# ------------------------------------------------------------------ #
# Embedding 추출 워커
# ------------------------------------------------------------------ #
class _EmbeddingWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(np.ndarray, list, list)  # embeddings, file_paths, labels
    error = Signal(str)

    def __init__(self, model_path, model_type, img_dir, layer_name, label_mode):
        super().__init__()
        self.model_path = model_path
        self.model_type = model_type
        self.img_dir = img_dir
        self.layer_name = layer_name
        self.label_mode = label_mode  # "folder" | "filename" | "none"

    def run(self):
        try:
            mi = load_model(self.model_path, model_type=self.model_type)
            if mi.session is None:
                self.error.emit("모델 세션이 없습니다.")
                return

            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            files = []
            for e in exts:
                files.extend(glob.glob(os.path.join(self.img_dir, "**", e), recursive=True))
            files.sort()
            if not files:
                self.error.emit("이미지가 없습니다.")
                return

            # 출력 레이어 결정
            out_names = [o.name for o in mi.session.get_outputs()]
            target = self.layer_name if self.layer_name in out_names else out_names[-1]

            embeddings = []
            labels = []
            bs = mi.batch_size

            for i, fp in enumerate(files):
                frame = cv2.imread(fp)
                if frame is None:
                    continue

                if mi.task_type == "classification":
                    tensor = preprocess_classification(frame, mi.input_size)
                else:
                    tensor = preprocess(frame, mi.input_size)

                if bs > 1:
                    tensor = np.repeat(tensor, bs, axis=0)

                out = mi.session.run([target], {mi.input_name: tensor})
                feat = out[0][0].flatten()
                embeddings.append(feat)

                # 라벨 결정
                if self.label_mode == "folder":
                    parent = os.path.basename(os.path.dirname(fp))
                    labels.append(parent)
                elif self.label_mode == "filename":
                    labels.append(os.path.splitext(os.path.basename(fp))[0].split("_")[0])
                else:
                    labels.append("")

                self.progress.emit(i + 1, len(files))

            self.finished_ok.emit(np.array(embeddings), files, labels)
        except Exception as e:
            self.error.emit(str(e))


# ------------------------------------------------------------------ #
# 시각화 탭
# ------------------------------------------------------------------ #
class EmbeddingViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._embeddings = None
        self._files = []
        self._labels = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        grp = QGroupBox("Embedding 추출 설정")
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
        g.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("출력 레이어:"))
        self._le_layer = QLineEdit()
        self._le_layer.setPlaceholderText("비워두면 마지막 출력 사용")
        row3.addWidget(self._le_layer, 1)
        row3.addWidget(QLabel("라벨 모드:"))
        self._combo_label = QComboBox()
        self._combo_label.addItems(["폴더명", "파일명 접두사", "없음"])
        row3.addWidget(self._combo_label)
        g.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("알고리즘:"))
        self._combo_algo = QComboBox()
        self._combo_algo.addItems(["t-SNE", "UMAP", "PCA"])
        row4.addWidget(self._combo_algo)
        row4.addWidget(QLabel("Perplexity:"))
        self._spin_perp = QSpinBox()
        self._spin_perp.setRange(5, 100)
        self._spin_perp.setValue(30)
        row4.addWidget(self._spin_perp)
        row4.addStretch()
        self._btn_run = QPushButton("추출 & 시각화")
        self._btn_run.clicked.connect(self._run)
        row4.addWidget(self._btn_run)
        g.addLayout(row4)

        self._prog = QProgressBar()
        self._prog.setMaximumHeight(18)
        g.addWidget(self._prog)
        root.addWidget(grp)

        # 차트 영역
        self._chart_container = QVBoxLayout()
        self._lbl_placeholder = QLabel("Embedding을 추출하면 여기에 시각화됩니다.")
        self._lbl_placeholder.setAlignment(Qt.AlignCenter)
        self._lbl_placeholder.setStyleSheet("color: #888; font-size: 14px;")
        self._chart_container.addWidget(self._lbl_placeholder)
        root.addLayout(self._chart_container, 1)

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
        if not model_path or not os.path.isdir(img_dir):
            QMessageBox.warning(self, "알림", "모델과 이미지 폴더를 선택하세요.")
            return

        mtype = "yolo" if self._combo_type.currentIndex() == 0 else "darknet"
        label_modes = ["folder", "filename", "none"]
        label_mode = label_modes[self._combo_label.currentIndex()]

        self._btn_run.setEnabled(False)
        self._prog.setValue(0)
        self._worker = _EmbeddingWorker(
            model_path, mtype, img_dir, self._le_layer.text().strip(), label_mode)
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(lambda e: (QMessageBox.critical(self, "오류", e),
                                              self._btn_run.setEnabled(True)))
        self._worker.start()

    def _on_done(self, embeddings, files, labels):
        self._btn_run.setEnabled(True)
        self._prog.setValue(100)
        self._embeddings = embeddings
        self._files = files
        self._labels = labels
        self._visualize()

    def _visualize(self):
        if self._embeddings is None or len(self._embeddings) < 2:
            QMessageBox.warning(self, "알림", "최소 2개 이상의 이미지가 필요합니다.")
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            QMessageBox.warning(self, "알림", "matplotlib이 필요합니다.")
            return

        algo = self._combo_algo.currentText()
        perp = self._spin_perp.value()

        # 차원 축소
        X = self._embeddings
        if algo == "t-SNE":
            try:
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, perplexity=min(perp, len(X) - 1), random_state=42)
                coords = reducer.fit_transform(X)
            except ImportError:
                QMessageBox.warning(self, "알림", "scikit-learn이 필요합니다.\npip install scikit-learn")
                return
        elif algo == "UMAP":
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                coords = reducer.fit_transform(X)
            except ImportError:
                QMessageBox.warning(self, "알림", "umap-learn이 필요합니다.\npip install umap-learn")
                return
        else:  # PCA
            try:
                from sklearn.decomposition import PCA
                coords = PCA(n_components=2, random_state=42).fit_transform(X)
            except ImportError:
                QMessageBox.warning(self, "알림", "scikit-learn이 필요합니다.")
                return

        # 기존 차트 제거
        for i in reversed(range(self._chart_container.count())):
            w = self._chart_container.itemAt(i).widget()
            if w:
                w.deleteLater()

        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        unique_labels = sorted(set(self._labels))
        if unique_labels and unique_labels != [""]:
            cmap = matplotlib.colormaps.get_cmap("tab20")
            label_to_idx = {l: i for i, l in enumerate(unique_labels)}
            colors = [cmap(label_to_idx[l] / max(len(unique_labels), 1)) for l in self._labels]
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=15, alpha=0.7)
            # 범례
            handles = []
            for l in unique_labels:
                idx = label_to_idx[l]
                h = ax.scatter([], [], c=[cmap(idx / max(len(unique_labels), 1))], s=30, label=l)
                handles.append(h)
            ax.legend(handles=handles, loc="best", fontsize=8, markerscale=1.5)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=15, alpha=0.7, c="#2196F3")

        ax.set_title(f"{algo} Embedding Visualization ({len(self._embeddings)} samples)")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        fig.tight_layout()

        canvas = FigureCanvasQTAgg(fig)
        self._chart_container.addWidget(canvas)
