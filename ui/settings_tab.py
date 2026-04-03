"""설정 탭: conf/두께/라벨크기/클래스별 색상·두께"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QSlider, QSpinBox, QCheckBox, QLabel,
    QGroupBox, QTableWidget, QTableWidgetItem,
    QPushButton, QColorDialog, QHeaderView, QComboBox,
)

from core.app_config import AppConfig
from ui.video_widget import get_palette_color


class SettingsTab(QWidget):
    settings_changed = Signal()

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._names: dict = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- 전역 설정 ---
        global_group = QGroupBox("전역 설정")
        form = QFormLayout(global_group)
        form.setSpacing(6)

        # 모델 타입
        self._model_type_combo = QComboBox()
        self._model_type_combo.addItems(["YOLO (v5/v7/v8/v9/v11)", "CenterNet"])
        self._model_type_combo.setCurrentIndex(0 if config.model_type == "yolo" else 1)
        self._model_type_combo.currentIndexChanged.connect(self._on_model_type_changed)
        form.addRow("모델 타입:", self._model_type_combo)

        # 배치 크기
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 16)
        self._batch_spin.setValue(config.batch_size)
        self._batch_spin.setToolTip(
            "추론 배치 크기. dynamic 배치 모델에서 N프레임을 모아 한번에 추론합니다.\n"
            "고정 배치 모델은 이 설정과 무관하게 모델 배치 크기를 따릅니다."
        )
        self._batch_spin.valueChanged.connect(self._on_batch_changed)
        form.addRow("배치 크기:", self._batch_spin)

        # Confidence
        self._conf_slider = QSlider(Qt.Horizontal)
        self._conf_slider.setRange(1, 99)
        self._conf_slider.setValue(int(config.conf_threshold * 100))
        self._conf_label = QLabel(f"{config.conf_threshold:.2f}")
        self._conf_slider.valueChanged.connect(self._on_conf_changed)
        conf_row = QHBoxLayout()
        conf_row.addWidget(self._conf_slider)
        conf_row.addWidget(self._conf_label)
        form.addRow("Confidence:", conf_row)

        # 박스 두께
        self._thickness_spin = QSpinBox()
        self._thickness_spin.setRange(1, 10)
        self._thickness_spin.setValue(config.box_thickness)
        self._thickness_spin.valueChanged.connect(self._on_thickness_changed)
        form.addRow("박스 두께:", self._thickness_spin)

        # 라벨 크기
        self._label_size_slider = QSlider(Qt.Horizontal)
        self._label_size_slider.setRange(3, 15)
        self._label_size_slider.setValue(int(config.label_size * 10))
        self._label_size_label = QLabel(f"{config.label_size:.1f}")
        self._label_size_slider.valueChanged.connect(self._on_label_size_changed)
        label_size_row = QHBoxLayout()
        label_size_row.addWidget(self._label_size_slider)
        label_size_row.addWidget(self._label_size_label)
        form.addRow("라벨 크기:", label_size_row)

        # 라벨 / Confidence 표시
        self._show_labels_cb = QCheckBox("라벨 표시")
        self._show_labels_cb.setChecked(config.show_labels)
        self._show_labels_cb.stateChanged.connect(self._on_show_labels_changed)

        self._show_conf_cb = QCheckBox("Confidence 표시")
        self._show_conf_cb.setChecked(config.show_confidence)
        self._show_conf_cb.stateChanged.connect(self._on_show_conf_changed)

        toggle_row = QHBoxLayout()
        toggle_row.addWidget(self._show_labels_cb)
        toggle_row.addWidget(self._show_conf_cb)
        form.addRow("표시 옵션:", toggle_row)

        layout.addWidget(global_group)

        # 저장 버튼 (우측 상단)
        save_row = QHBoxLayout()
        save_row.addStretch()
        btn_save = QPushButton("설정 저장")
        btn_save.setFixedHeight(30)
        btn_save.setMinimumWidth(110)
        btn_save.clicked.connect(self._save)
        save_row.addWidget(btn_save)
        layout.addLayout(save_row)

        # --- 클래스별 설정 테이블 ---
        class_group = QGroupBox("클래스별 설정")
        class_layout = QVBoxLayout(class_group)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["클래스", "활성화", "색상", "두께"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        self._table.verticalHeader().setVisible(False)
        class_layout.addWidget(self._table)

        layout.addWidget(class_group)

    def populate_classes(self, names: dict):
        """모델 로드 시 클래스 테이블 채우기"""
        self._names = names
        self._table.setRowCount(0)
        total = len(names)

        for cls_id in sorted(names.keys()):
            name = names[cls_id]
            style = self._config.get_class_style(cls_id)
            row = self._table.rowCount()
            self._table.insertRow(row)

            # 클래스 이름
            name_item = QTableWidgetItem(f"{cls_id}: {name}")
            name_item.setFlags(Qt.ItemIsEnabled)
            self._table.setItem(row, 0, name_item)

            # 활성화 체크박스
            cb = QCheckBox()
            cb.setChecked(style.enabled)
            cb.setStyleSheet("margin-left: 8px;")
            cb.stateChanged.connect(lambda state, cid=cls_id: self._on_class_enabled(cid, bool(state)))
            cb_widget = QWidget()
            cb_layout = QHBoxLayout(cb_widget)
            cb_layout.addWidget(cb)
            cb_layout.setAlignment(Qt.AlignCenter)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            self._table.setCellWidget(row, 1, cb_widget)

            # 색상 버튼
            color = style.color if style.color else get_palette_color(cls_id, total)
            btn_color = QPushButton()
            btn_color.setFixedSize(40, 22)
            r, g, b = color[2], color[1], color[0]  # BGR→RGB
            btn_color.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #888;")
            btn_color.clicked.connect(lambda _, cid=cls_id, btn=btn_color: self._on_color_click(cid, btn))
            self._table.setCellWidget(row, 2, btn_color)

            # 두께 스핀박스
            spin = QSpinBox()
            spin.setRange(0, 10)
            spin.setSpecialValueText("기본")
            spin.setValue(style.thickness or 0)
            spin.valueChanged.connect(lambda val, cid=cls_id: self._on_class_thickness(cid, val))
            self._table.setCellWidget(row, 3, spin)

    # --- 핸들러 ---
    def _on_model_type_changed(self, idx: int):
        self._config.model_type = "yolo" if idx == 0 else "darknet"
        self.settings_changed.emit()

    def _on_batch_changed(self, val: int):
        self._config.batch_size = val

    def _on_conf_changed(self, val: int):
        self._config.conf_threshold = val / 100.0
        self._conf_label.setText(f"{self._config.conf_threshold:.2f}")
        self.settings_changed.emit()

    def _on_thickness_changed(self, val: int):
        self._config.box_thickness = val
        self.settings_changed.emit()

    def _on_label_size_changed(self, val: int):
        self._config.label_size = val / 10.0
        self._label_size_label.setText(f"{self._config.label_size:.1f}")
        self.settings_changed.emit()

    def _on_show_labels_changed(self, state: int):
        self._config.show_labels = bool(state)
        self.settings_changed.emit()

    def _on_show_conf_changed(self, state: int):
        self._config.show_confidence = bool(state)
        self.settings_changed.emit()

    def _on_class_enabled(self, cls_id: int, enabled: bool):
        style = self._config.get_class_style(cls_id)
        style.enabled = enabled
        self._config.set_class_style(cls_id, style)
        self.settings_changed.emit()

    def _on_color_click(self, cls_id: int, btn: QPushButton):
        style = self._config.get_class_style(cls_id)
        init_color = QColor(*reversed(style.color)) if style.color else QColor(0, 255, 0)
        color = QColorDialog.getColor(init_color, self, f"클래스 {cls_id} 색상 선택")
        if color.isValid():
            bgr = (color.blue(), color.green(), color.red())
            style.color = bgr
            self._config.set_class_style(cls_id, style)
            r, g, b = color.red(), color.green(), color.blue()
            btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #888;")
            self.settings_changed.emit()

    def _on_class_thickness(self, cls_id: int, val: int):
        style = self._config.get_class_style(cls_id)
        style.thickness = val if val > 0 else None
        self._config.set_class_style(cls_id, style)
        self.settings_changed.emit()

    def _save(self):
        self._config.save()
