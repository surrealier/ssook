"""Videos / Models 폴더 파일 목록 패널"""
import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QFileDialog, QSizePolicy,
    QScrollBar,
)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".ts", ".m4v"}
MODEL_EXTS = {".onnx", ".pt"}


class FileBrowserWidget(QWidget):
    video_selected = Signal(str)   # 절대 경로
    model_selected = Signal(str)

    def __init__(self, videos_dir: str = "Videos", models_dir: str = "Models", parent=None):
        super().__init__(parent)
        self.videos_dir = videos_dir
        self.models_dir = models_dir

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- 모델 패널 ---
        layout.addWidget(QLabel("모델 (ONNX / PT)"))
        self._model_list = QListWidget()
        self._model_list.setMaximumHeight(160)
        self._model_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._model_list.itemDoubleClicked.connect(self._on_model_double_clicked)
        self._model_list.setHorizontalScrollMode(QListWidget.ScrollPerPixel)
        self._model_list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self._model_list.horizontalScrollBar().setSingleStep(15)
        self._model_list.verticalScrollBar().setSingleStep(15)
        layout.addWidget(self._model_list)

        model_btn_row = QHBoxLayout()
        btn_refresh_model = QPushButton("새로고침")
        btn_browse_model = QPushButton("탐색...")
        btn_refresh_model.clicked.connect(self._populate_models)
        btn_browse_model.clicked.connect(self._browse_model)
        model_btn_row.addWidget(btn_refresh_model)
        model_btn_row.addWidget(btn_browse_model)
        layout.addLayout(model_btn_row)

        # --- 비디오 패널 ---
        layout.addWidget(QLabel("비디오"))
        self._video_list = QListWidget()
        self._video_list.itemDoubleClicked.connect(self._on_video_double_clicked)
        self._video_list.setHorizontalScrollMode(QListWidget.ScrollPerPixel)
        self._video_list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self._video_list.horizontalScrollBar().setSingleStep(15)
        self._video_list.verticalScrollBar().setSingleStep(15)
        layout.addWidget(self._video_list)

        video_btn_row = QHBoxLayout()
        btn_refresh_video = QPushButton("새로고침")
        btn_browse_video = QPushButton("탐색...")
        btn_refresh_video.clicked.connect(self._populate_videos)
        btn_browse_video.clicked.connect(self._browse_video)
        video_btn_row.addWidget(btn_refresh_video)
        video_btn_row.addWidget(btn_browse_video)
        layout.addLayout(video_btn_row)

        self._populate_models()
        self._populate_videos()

    def _populate_list(self, list_widget: QListWidget, directory: str, exts: set):
        list_widget.clear()
        if not os.path.isdir(directory):
            return
        for fname in sorted(os.listdir(directory)):
            if os.path.splitext(fname)[1].lower() in exts:
                item = QListWidgetItem(fname)
                item.setData(256, os.path.abspath(os.path.join(directory, fname)))
                list_widget.addItem(item)

    def _populate_models(self):
        self._populate_list(self._model_list, self.models_dir, MODEL_EXTS)

    def _populate_videos(self):
        self._populate_list(self._video_list, self.videos_dir, VIDEO_EXTS)

    def _on_model_double_clicked(self, item: QListWidgetItem):
        self.model_selected.emit(item.data(256))

    def _on_video_double_clicked(self, item: QListWidgetItem):
        self.video_selected.emit(item.data(256))

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "모델 파일 선택", self.models_dir,
            "모델 파일 (*.onnx *.pt)"
        )
        if path:
            self.model_selected.emit(path)

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "비디오 파일 선택", self.videos_dir,
            "비디오 파일 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.ts *.m4v)"
        )
        if path:
            self.video_selected.emit(path)

    def add_external_file(self, path: str):
        """드래그앤드롭 등 외부에서 파일이 들어왔을 때 목록에 추가"""
        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXTS:
            self.video_selected.emit(path)
        elif ext in MODEL_EXTS:
            self.model_selected.emit(path)
