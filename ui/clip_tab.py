"""VLM/CLIP 탭: 이미지-텍스트 매칭, zero-shot classification"""
import os, glob

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QGroupBox, QProgressBar,
    QMessageBox, QTextEdit, QScrollArea, QGridLayout, QSplitter,
)

from core.clip_inference import CLIPModel, simple_tokenize


class _ZeroShotWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(list)  # [(img_path, [(label, score), ...])]
    error = Signal(str)

    def __init__(self, clip_model, img_dir, labels):
        super().__init__()
        self.clip = clip_model
        self.img_dir = img_dir
        self.labels = labels

    def run(self):
        try:
            # 텍스트 임베딩 사전 계산
            text_embs = []
            for label in self.labels:
                tokens = simple_tokenize(f"a photo of {label}")
                emb = self.clip.encode_text(tokens)
                text_embs.append(emb)

            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            files = []
            for e in exts:
                files.extend(glob.glob(os.path.join(self.img_dir, e)))
            files.sort()
            if not files:
                self.error.emit("이미지가 없습니다."); return

            results = []
            for i, fp in enumerate(files):
                frame = cv2.imread(fp)
                if frame is None: continue
                ranked = self.clip.zero_shot_classify(frame, text_embs, self.labels)
                results.append((fp, ranked))
                self.progress.emit(i + 1, len(files))

            self.finished_ok.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class CLIPTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._clip = None
        self._results = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        grp = QGroupBox("CLIP 모델 설정")
        g = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Image Encoder:"))
        self._le_img_enc = QLineEdit(); self._le_img_enc.setReadOnly(True)
        row1.addWidget(self._le_img_enc, 1)
        btn1 = QPushButton("찾아보기")
        btn1.clicked.connect(lambda: self._browse_onnx(self._le_img_enc))
        row1.addWidget(btn1)
        g.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Text Encoder:"))
        self._le_txt_enc = QLineEdit(); self._le_txt_enc.setReadOnly(True)
        row2.addWidget(self._le_txt_enc, 1)
        btn2 = QPushButton("찾아보기")
        btn2.clicked.connect(lambda: self._browse_onnx(self._le_txt_enc))
        row2.addWidget(btn2)
        self._btn_load = QPushButton("모델 로드")
        self._btn_load.clicked.connect(self._load_model)
        row2.addWidget(self._btn_load)
        g.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("이미지 폴더:"))
        self._le_img_dir = QLineEdit()
        row3.addWidget(self._le_img_dir, 1)
        btn3 = QPushButton("찾아보기")
        btn3.clicked.connect(self._browse_img_dir)
        row3.addWidget(btn3)
        g.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("클래스 라벨 (줄바꿈 구분):"))
        self._te_labels = QTextEdit()
        self._te_labels.setMaximumHeight(80)
        self._te_labels.setPlaceholderText("dog\ncat\nbird\ncar\nperson")
        row4.addWidget(self._te_labels, 1)
        g.addLayout(row4)

        row5 = QHBoxLayout()
        row5.addStretch()
        self._btn_run = QPushButton("Zero-Shot 분류 실행")
        self._btn_run.clicked.connect(self._run)
        self._btn_run.setEnabled(False)
        row5.addWidget(self._btn_run)
        g.addLayout(row5)

        self._prog = QProgressBar(); self._prog.setMaximumHeight(18)
        g.addWidget(self._prog)
        root.addWidget(grp)

        # 결과 갤러리
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._gallery = QWidget()
        self._grid = QGridLayout(self._gallery)
        self._grid.setSpacing(6)
        self._scroll.setWidget(self._gallery)
        root.addWidget(self._scroll, 1)

    def _browse_onnx(self, le):
        p, _ = QFileDialog.getOpenFileName(self, "ONNX 선택", "", "ONNX (*.onnx)")
        if p: le.setText(p)

    def _browse_img_dir(self):
        d = QFileDialog.getExistingDirectory(self, "이미지 폴더")
        if d: self._le_img_dir.setText(d)

    def _load_model(self):
        img_enc = self._le_img_enc.text()
        txt_enc = self._le_txt_enc.text()
        if not img_enc:
            QMessageBox.warning(self, "알림", "Image Encoder를 선택하세요."); return
        try:
            self._clip = CLIPModel(img_enc, txt_enc if txt_enc else None)
            self._btn_run.setEnabled(True)
            QMessageBox.information(self, "완료", "CLIP 모델 로드 완료")
        except Exception as e:
            QMessageBox.critical(self, "오류", str(e))

    def _run(self):
        if not self._clip:
            QMessageBox.warning(self, "알림", "모델을 먼저 로드하세요."); return
        img_dir = self._le_img_dir.text()
        if not os.path.isdir(img_dir):
            QMessageBox.warning(self, "알림", "이미지 폴더를 선택하세요."); return
        labels = [l.strip() for l in self._te_labels.toPlainText().strip().splitlines() if l.strip()]
        if not labels:
            QMessageBox.warning(self, "알림", "클래스 라벨을 입력하세요."); return

        self._btn_run.setEnabled(False)
        self._prog.setValue(0)
        self._worker = _ZeroShotWorker(self._clip, img_dir, labels)
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(lambda e: (QMessageBox.critical(self, "오류", e),
                                              self._btn_run.setEnabled(True)))
        self._worker.start()

    def _on_done(self, results):
        self._btn_run.setEnabled(True)
        self._prog.setValue(100)
        self._results = results

        # 갤러리 초기화
        while self._grid.count():
            w = self._grid.takeAt(0).widget()
            if w: w.deleteLater()

        cols = 4
        for idx, (fp, ranked) in enumerate(results[:100]):  # 최대 100장
            card = QWidget()
            card_lay = QVBoxLayout(card)
            card_lay.setContentsMargins(4, 4, 4, 4)

            # 썸네일
            img = cv2.imread(fp)
            if img is not None:
                size = 150
                scale = size / max(img.shape[:2])
                thumb = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
                rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
                lbl_img = QLabel()
                lbl_img.setPixmap(QPixmap.fromImage(qimg))
                lbl_img.setAlignment(Qt.AlignCenter)
                card_lay.addWidget(lbl_img)

            # top-3 결과
            text = "\n".join(f"{l}: {s:.1%}" for l, s in ranked[:3])
            lbl_txt = QLabel(text)
            lbl_txt.setStyleSheet("font-size: 11px;")
            lbl_txt.setAlignment(Qt.AlignCenter)
            card_lay.addWidget(lbl_txt)

            self._grid.addWidget(card, idx // cols, idx % cols)
