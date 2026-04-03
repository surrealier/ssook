"""Embedder 평가: Retrieval accuracy, 유사 이미지 검색, cosine similarity 매트릭스"""
import os, glob

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QComboBox, QGroupBox, QProgressBar,
    QMessageBox, QSpinBox, QScrollArea, QGridLayout, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView,
)

from core.model_loader import load_model
from core.inference import preprocess, preprocess_classification, letterbox


class _EmbedWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(np.ndarray, list, list)  # embeddings, paths, labels
    error = Signal(str)

    def __init__(self, model_path, model_type, img_dir, layer_name):
        super().__init__()
        self.model_path = model_path
        self.model_type = model_type
        self.img_dir = img_dir
        self.layer_name = layer_name

    def run(self):
        try:
            mi = load_model(self.model_path, model_type=self.model_type)
            if mi.session is None:
                self.error.emit("모델 세션 없음"); return

            files = []
            for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                files.extend(glob.glob(os.path.join(self.img_dir, "**", e), recursive=True))
            files.sort()
            if not files:
                self.error.emit("이미지가 없습니다."); return

            out_names = [o.name for o in mi.session.get_outputs()]
            target = self.layer_name if self.layer_name in out_names else out_names[-1]
            bs = mi.batch_size

            embeddings, labels = [], []
            for i, fp in enumerate(files):
                frame = cv2.imread(fp)
                if frame is None: continue
                tensor = preprocess_classification(frame, mi.input_size) if mi.task_type == "classification" else preprocess(frame, mi.input_size)
                if bs > 1:
                    tensor = np.repeat(tensor, bs, axis=0)
                out = mi.session.run([target], {mi.input_name: tensor})
                emb = out[0][0].flatten().astype(np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-9)
                embeddings.append(emb)
                labels.append(os.path.basename(os.path.dirname(fp)))
                self.progress.emit(i + 1, len(files))

            self.finished_ok.emit(np.array(embeddings), files, labels)
        except Exception as e:
            self.error.emit(str(e))


class EmbedderEvalTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._embeddings = None
        self._files = []
        self._labels = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        grp = QGroupBox("Embedder 평가 설정")
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
        row2.addWidget(QLabel("이미지 폴더 (하위 폴더=클래스):"))
        self._le_img = QLineEdit()
        row2.addWidget(self._le_img, 1)
        btn_i = QPushButton("찾아보기")
        btn_i.clicked.connect(lambda: self._browse_dir(self._le_img))
        row2.addWidget(btn_i)
        row2.addWidget(QLabel("출력 레이어:"))
        self._le_layer = QLineEdit()
        self._le_layer.setPlaceholderText("비워두면 마지막")
        row2.addWidget(self._le_layer)
        g.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Top-K:"))
        self._spin_k = QSpinBox(); self._spin_k.setRange(1, 50); self._spin_k.setValue(5)
        row3.addWidget(self._spin_k)
        row3.addStretch()
        self._btn_run = QPushButton("평가 실행")
        self._btn_run.clicked.connect(self._run)
        row3.addWidget(self._btn_run)
        g.addLayout(row3)

        self._prog = QProgressBar(); self._prog.setMaximumHeight(18)
        g.addWidget(self._prog)
        root.addWidget(grp)

        # 결과
        splitter = QSplitter(Qt.Horizontal)

        # 좌: 메트릭 테이블
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        self._lbl_metrics = QLabel("평가를 실행하세요.")
        self._lbl_metrics.setWordWrap(True)
        left_lay.addWidget(self._lbl_metrics)
        self._btn_sim_matrix = QPushButton("유사도 매트릭스 보기")
        self._btn_sim_matrix.clicked.connect(self._show_sim_matrix)
        self._btn_sim_matrix.setEnabled(False)
        left_lay.addWidget(self._btn_sim_matrix)
        self._btn_search = QPushButton("유사 이미지 검색")
        self._btn_search.clicked.connect(self._search_similar)
        self._btn_search.setEnabled(False)
        left_lay.addWidget(self._btn_search)
        left_lay.addStretch()
        splitter.addWidget(left)

        # 우: 검색 결과 갤러리
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._gallery = QWidget()
        self._grid = QGridLayout(self._gallery)
        self._grid.setSpacing(4)
        self._scroll.setWidget(self._gallery)
        splitter.addWidget(self._scroll)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

    def _browse_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "모델 선택", "", "ONNX (*.onnx)")
        if p: self._le_model.setText(p)

    def _browse_dir(self, le):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if d: le.setText(d)

    def _run(self):
        if not self._le_model.text() or not os.path.isdir(self._le_img.text()):
            QMessageBox.warning(self, "알림", "모델과 이미지 폴더를 지정하세요."); return
        mtype = "yolo" if self._combo_type.currentIndex() == 0 else "darknet"
        self._btn_run.setEnabled(False)
        self._prog.setValue(0)
        self._worker = _EmbedWorker(self._le_model.text(), mtype,
                                    self._le_img.text(), self._le_layer.text().strip())
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
        self._btn_sim_matrix.setEnabled(True)
        self._btn_search.setEnabled(True)

        # Retrieval 평가
        k = self._spin_k.value()
        unique_labels = sorted(set(l for l in labels if l))
        if len(unique_labels) < 2:
            self._lbl_metrics.setText(f"임베딩 추출 완료: {len(embeddings)}개\n(Retrieval 평가에는 2개 이상 클래스 필요)")
            return

        sim_matrix = embeddings @ embeddings.T
        n = len(embeddings)
        top1_correct = 0
        topk_correct = 0
        for i in range(n):
            sims = sim_matrix[i].copy()
            sims[i] = -1  # 자기 자신 제외
            top_indices = np.argsort(sims)[::-1][:k]
            if labels[top_indices[0]] == labels[i]:
                top1_correct += 1
            if any(labels[j] == labels[i] for j in top_indices):
                topk_correct += 1

        r1 = top1_correct / n
        rk = topk_correct / n
        self._lbl_metrics.setText(
            f"임베딩: {n}개, 클래스: {len(unique_labels)}종\n"
            f"Retrieval@1: {r1:.4f} ({top1_correct}/{n})\n"
            f"Retrieval@{k}: {rk:.4f} ({topk_correct}/{n})")

    def _show_sim_matrix(self):
        if self._embeddings is None: return
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            QMessageBox.warning(self, "알림", "matplotlib 필요"); return

        from PySide6.QtWidgets import QDialog
        sim = self._embeddings @ self._embeddings.T
        dlg = QDialog(self)
        dlg.setWindowTitle("Cosine Similarity Matrix")
        dlg.resize(700, 600)
        lay = QVBoxLayout(dlg)
        fig = Figure(figsize=(8, 7))
        ax = fig.add_subplot(111)
        im = ax.imshow(sim, cmap="RdYlBu_r", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set_title(f"Cosine Similarity ({len(sim)}×{len(sim)})")
        fig.tight_layout()
        canvas = FigureCanvasQTAgg(fig)
        lay.addWidget(canvas)
        dlg.exec()

    def _search_similar(self):
        if self._embeddings is None: return
        p, _ = QFileDialog.getOpenFileName(self, "쿼리 이미지 선택", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if not p: return

        # 쿼리 이미지의 인덱스 찾기 또는 새로 추론
        idx = None
        for i, fp in enumerate(self._files):
            if os.path.abspath(fp) == os.path.abspath(p):
                idx = i; break

        if idx is not None:
            query_emb = self._embeddings[idx]
        else:
            QMessageBox.information(self, "알림", "데이터셋 내 이미지를 선택하세요.")
            return

        sims = self._embeddings @ query_emb
        sims[idx] = -2
        top_k = np.argsort(sims)[::-1][:20]

        while self._grid.count():
            w = self._grid.takeAt(0).widget()
            if w: w.deleteLater()

        cols = 5
        for r, j in enumerate(top_k):
            fp = self._files[j]
            img = cv2.imread(fp)
            if img is None: continue
            size = 130
            scale = size / max(img.shape[:2])
            thumb = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
            rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
            card = QLabel()
            card.setPixmap(QPixmap.fromImage(qimg))
            card.setToolTip(f"{os.path.basename(fp)}\nsim: {sims[j]:.4f}\nlabel: {self._labels[j]}")
            card.setFixedSize(size, size)
            card.setAlignment(Qt.AlignCenter)
            card.setStyleSheet("border: 2px solid #444; background: #222;")
            self._grid.addWidget(card, r // cols, r % cols)
