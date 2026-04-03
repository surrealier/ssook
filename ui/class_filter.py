"""클래스별 ON/OFF 필터 위젯"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QCheckBox, QPushButton,
)

from core.app_config import AppConfig


class ClassFilterWidget(QWidget):
    filter_changed = Signal()

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._checkboxes: dict[int, QCheckBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # 헤더 + 전체선택/해제
        header = QHBoxLayout()
        header.addWidget(QLabel("클래스 필터"))
        btn_all = QPushButton("전체 ON")
        btn_none = QPushButton("전체 OFF")
        btn_all.setFixedWidth(64)
        btn_none.setFixedWidth(64)
        btn_all.clicked.connect(self._select_all)
        btn_none.clicked.connect(self._deselect_all)
        header.addStretch()
        header.addWidget(btn_all)
        header.addWidget(btn_none)
        layout.addLayout(header)

        # 스크롤 영역
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(2, 2, 2, 2)
        self._container_layout.setSpacing(2)
        self._container_layout.addStretch()
        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll)

    def populate(self, names: dict):
        """모델 로드 시 클래스 목록으로 초기화"""
        # 기존 위젯 제거
        for cb in self._checkboxes.values():
            cb.setParent(None)
        self._checkboxes.clear()

        # stretch 제거 후 추가
        item = self._container_layout.takeAt(self._container_layout.count() - 1)
        del item

        for cls_id in sorted(names.keys()):
            name = names[cls_id]
            cb = QCheckBox(f"{cls_id}: {name}")
            style = self._config.get_class_style(cls_id)
            cb.setChecked(style.enabled)
            cb.stateChanged.connect(lambda state, cid=cls_id: self._on_changed(cid, state))
            self._checkboxes[cls_id] = cb
            self._container_layout.addWidget(cb)

        self._container_layout.addStretch()

    def _on_changed(self, cls_id: int, state: int):
        style = self._config.get_class_style(cls_id)
        style.enabled = bool(state)
        self._config.set_class_style(cls_id, style)
        self.filter_changed.emit()

    def _select_all(self):
        for cls_id, cb in self._checkboxes.items():
            cb.setChecked(True)

    def _deselect_all(self):
        for cls_id, cb in self._checkboxes.items():
            cb.setChecked(False)
