"""재생 컨트롤 바: 재생/정지/탐색/속도/FPS/스냅샷/녹화"""
import os
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QMouseEvent, QIcon
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QSlider, QLabel, QComboBox, QSizePolicy, QStyle,
)

_ICON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "icons")

def _icon(name: str) -> QIcon:
    return QIcon(os.path.join(_ICON_DIR, f"{name}.svg"))


class ClickableSlider(QSlider):
    """클릭한 위치로 바로 이동하는 슬라이더"""
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            val = QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(), int(event.position().x()), self.width()
            )
            self.setValue(val)
            self.sliderMoved.emit(val)
        super().mousePressEvent(event)


class ControlBar(QWidget):
    play_clicked = Signal()
    pause_clicked = Signal()
    stop_clicked = Signal()
    seek_requested = Signal(int)
    snapshot_clicked = Signal()
    record_toggled = Signal(bool)
    speed_changed = Signal(float)
    step_forward = Signal()
    step_backward = Signal()
    csv_record_toggled = Signal(bool)
    csv_export_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._total_frames = 0
        self._is_playing = False
        self._is_recording = False
        self._slider_dragging = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        # 탐색 슬라이더
        self._slider = ClickableSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._slider.sliderMoved.connect(lambda v: self.seek_requested.emit(v))
        layout.addWidget(self._slider)

        # 버튼 행
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        _SZ = QSize(20, 20)

        self._btn_prev = QPushButton()
        self._btn_prev.setIcon(_icon("prev"))
        self._btn_prev.setIconSize(_SZ)
        self._btn_prev.setFixedWidth(30)
        self._btn_prev.setToolTip("1프레임 뒤로")
        self._btn_prev.clicked.connect(self.step_backward)

        self._btn_play = QPushButton()
        self._btn_play.setIcon(_icon("play"))
        self._btn_play.setIconSize(_SZ)
        self._btn_play.setFixedWidth(40)
        self._btn_play.clicked.connect(self._on_play_pause)

        self._btn_next = QPushButton()
        self._btn_next.setIcon(_icon("next"))
        self._btn_next.setIconSize(_SZ)
        self._btn_next.setFixedWidth(30)
        self._btn_next.setToolTip("1프레임 앞으로")
        self._btn_next.clicked.connect(self.step_forward)

        self._btn_stop = QPushButton()
        self._btn_stop.setIcon(_icon("stop"))
        self._btn_stop.setIconSize(_SZ)
        self._btn_stop.setFixedWidth(30)
        self._btn_stop.clicked.connect(self._on_stop)

        self._btn_snapshot = QPushButton("스냅샷")
        self._btn_snapshot.clicked.connect(self.snapshot_clicked)

        self._btn_record = QPushButton()
        self._btn_record.setIcon(_icon("record"))
        self._btn_record.setIconSize(_SZ)
        self._btn_record.setCheckable(True)
        self._btn_record.clicked.connect(self._on_record_toggled)

        # 속도
        speed_label = QLabel("속도:")
        self._speed_combo = QComboBox()
        for s in ["0.25x", "0.5x", "1.0x", "1.5x", "2.0x", "4.0x"]:
            self._speed_combo.addItem(s)
        self._speed_combo.setCurrentText("1.0x")
        self._speed_combo.currentTextChanged.connect(self._on_speed_changed)
        self._speed_combo.setFixedWidth(70)

        # FPS 정보
        self._fps_label = QLabel("FPS: --")
        self._fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._fps_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # 프레임 정보
        self._frame_label = QLabel("0 / 0")
        self._frame_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        btn_row.addWidget(self._btn_prev)
        btn_row.addWidget(self._btn_play)
        btn_row.addWidget(self._btn_next)
        btn_row.addWidget(self._btn_stop)
        btn_row.addSpacing(8)
        btn_row.addWidget(self._btn_snapshot)
        btn_row.addWidget(self._btn_record)
        btn_row.addSpacing(8)
        self._btn_csv_rec = QPushButton("CSV 기록")
        self._btn_csv_rec.setCheckable(True)
        self._btn_csv_rec.setToolTip("탐지 데이터 CSV 기록 시작/중지")
        self._btn_csv_rec.clicked.connect(self.csv_record_toggled)
        btn_row.addWidget(self._btn_csv_rec)
        self._btn_csv_export = QPushButton("CSV 저장")
        self._btn_csv_export.setToolTip("누적된 탐지 데이터를 CSV로 내보내기")
        self._btn_csv_export.clicked.connect(self.csv_export_clicked)
        btn_row.addWidget(self._btn_csv_export)
        btn_row.addSpacing(8)
        btn_row.addWidget(speed_label)
        btn_row.addWidget(self._speed_combo)
        btn_row.addStretch()
        btn_row.addWidget(self._fps_label)
        btn_row.addWidget(self._frame_label)
        layout.addLayout(btn_row)

    # --- 외부 업데이트 ---
    def set_total_frames(self, n: int):
        self._total_frames = n
        self._slider.setMaximum(max(0, n - 1))

    def update_position(self, frame_idx: int, total: int):
        if total > 0 and total - 1 != self._slider.maximum():
            self._slider.setMaximum(total - 1)
        if not self._slider_dragging:
            self._slider.setValue(frame_idx)
        self._frame_label.setText(f"{frame_idx} / {total}")

    def seek_to_end(self):
        if not self._slider_dragging:
            self._slider.setValue(self._slider.maximum())

    def update_fps(self, video_fps: float, infer_fps: float, infer_ms: float = 0.0):
        self._fps_label.setText(
            f"비디오  {video_fps:.1f} fps    │    추론  {infer_fps:.1f} fps  ({infer_ms:.1f} ms)"
        )

    def set_playing(self, playing: bool):
        self._is_playing = playing
        self._btn_play.setIcon(_icon("pause") if playing else _icon("play"))

    # --- 내부 핸들러 ---
    def _on_play_pause(self):
        if self._is_playing:
            self.pause_clicked.emit()
        else:
            self.play_clicked.emit()

    def _on_stop(self):
        self.stop_clicked.emit()

    def _on_slider_pressed(self):
        self._slider_dragging = True

    def _on_slider_released(self):
        self._slider_dragging = False
        self.seek_requested.emit(self._slider.value())

    def _on_record_toggled(self, checked: bool):
        self._is_recording = checked
        self._btn_record.setIcon(_icon("record_on") if checked else _icon("record"))
        self.record_toggled.emit(checked)

    def _on_speed_changed(self, text: str):
        try:
            speed = float(text.replace("x", ""))
            self.speed_changed.emit(speed)
        except ValueError:
            pass
