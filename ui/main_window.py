"""메인 윈도우: 레이아웃 조립 + 시그널 연결"""
import csv
import os
import time
from datetime import datetime

import cv2
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QTabWidget,
    QVBoxLayout, QHBoxLayout, QStatusBar, QMessageBox, QLabel,
    QMenuBar, QFileDialog,
)

from core.app_config import AppConfig
from core.inference import DetectionResult, ClassificationResult
from core.model_loader import ModelInfo, load_model
from ui.class_filter import ClassFilterWidget
from ui.control_bar import ControlBar
from ui.detect_thread import DetectThread
from ui.file_browser import FileBrowserWidget
from ui.analysis_tab import AnalysisTab
from ui.benchmark_tab import BenchmarkTab
from ui.evaluation_tab import EvaluationTab
from ui.dataset_explorer import DatasetExplorer
from ui.model_compare import ModelCompareView
from ui.error_analyzer import ErrorAnalyzer
from ui.embedding_viewer import EmbeddingViewer
from ui.conf_optimizer import ConfOptimizer
from ui.dataset_splitter import DatasetSplitter
from ui.image_quality_checker import ImageQualityChecker
from ui.near_duplicate_detector import NearDuplicateDetector
from ui.label_anomaly_detector import LabelAnomalyDetector
from ui.format_converter import FormatConverter
from ui.augmentation_preview import AugmentationPreview
from ui.class_remapper import ClassRemapper
from ui.similarity_search import SimilaritySearch
from ui.smart_sampler import SmartSampler
from ui.dataset_merger import DatasetMerger
from ui.leaky_split_detector import LeakySplitDetector
from ui.clip_tab import CLIPTab
from ui.embedder_eval import EmbedderEvalTab
from ui.segmentation_tab import SegmentationTab
from ui.settings_tab import SettingsTab
from ui.stats_widget import StatsWidget
from ui.video_widget import VideoWidget
from ui.i18n import t, set_language, get_language
from ui import theme


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ssook")
        self.resize(1280, 720)

        self._config = AppConfig()
        self._model_info: ModelInfo | None = None
        self._video_path: str | None = None
        self._thread: DetectThread | None = None
        self._recorder: cv2.VideoWriter | None = None
        self._last_frame = None
        self._last_result: DetectionResult | None = None
        self._csv_rows: list = []
        self._csv_recording = False
        self._last_render_time: float = 0.0
        self._MIN_RENDER_INTERVAL = 1.0 / 35.0  # ~35fps cap for GUI rendering

        self._build_ui()
        self._connect_signals()
        self._status(t("ready"))

        # 기본 모델 자동 로드 (#11)
        if self._config.default_model_path and os.path.isfile(self._config.default_model_path):
            try:
                self._on_model_selected(self._config.default_model_path)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # UI 구성
    # ------------------------------------------------------------------ #
    _DARK_STYLE = None  # replaced by theme system

    def _build_ui(self):
        # 라이트 모드 기본
        self._dark_mode = False
        theme.set_dark(False)
        self.setStyleSheet(theme.generate_qss(dark=False))

        # 메뉴바
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)
        view_menu = menubar.addMenu(t("view"))
        self._act_dark = view_menu.addAction(t("dark_mode"))
        self._act_dark.setCheckable(True)
        self._act_dark.setChecked(self._dark_mode)
        self._act_dark.triggered.connect(self._toggle_dark_mode)

        lang_menu = menubar.addMenu(t("language"))
        for code, label in [("en", "English"), ("ko", "한국어"), ("zh", "中文")]:
            act = lang_menu.addAction(label)
            act.triggered.connect(lambda _, c=code: self._change_language(c))

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # 탭
        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

        # --- 뷰어 탭 ---
        viewer_widget = QWidget()
        viewer_layout = QHBoxLayout(viewer_widget)
        viewer_layout.setContentsMargins(0, 0, 0, 0)

        # 좌측 패널 (파일 브라우저 + 클래스 필터)
        left_panel = QWidget()
        left_panel.setFixedWidth(220)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self._file_browser = FileBrowserWidget(
            self._config, self._config.videos_dir, self._config.models_dir
        )
        self._class_filter = ClassFilterWidget(self._config)
        left_layout.addWidget(self._file_browser, stretch=1)
        left_layout.addWidget(self._class_filter, stretch=1)

        # 중앙: 비디오 + 컨트롤바
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(2)

        self._video_widget = VideoWidget()
        self._control_bar = ControlBar()
        center_layout.addWidget(self._video_widget, stretch=1)
        center_layout.addWidget(self._control_bar)

        # 우측: 통계 패널
        self._stats_widget = StatsWidget()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(self._stats_widget)
        splitter.setStretchFactor(1, 1)
        viewer_layout.addWidget(splitter)

        self._tabs.addTab(viewer_widget, t("viewer"))

        # --- 설정 탭 ---
        self._settings_tab = SettingsTab(self._config)
        self._tabs.addTab(self._settings_tab, t("settings"))

        # === 평가 그룹 ===
        eval_group = QTabWidget()
        eval_group.setTabPosition(QTabWidget.South)
        self._evaluation_tab = EvaluationTab()
        eval_group.addTab(self._evaluation_tab, "Detection/Cls")
        self._segmentation_tab = SegmentationTab()
        eval_group.addTab(self._segmentation_tab, t("segmentation"))
        self._clip_tab = CLIPTab()
        eval_group.addTab(self._clip_tab, "CLIP")
        self._embedder_eval = EmbedderEvalTab()
        eval_group.addTab(self._embedder_eval, "Embedder")
        self._benchmark_tab = BenchmarkTab()
        eval_group.addTab(self._benchmark_tab, t("benchmark"))
        self._tabs.addTab(eval_group, t("evaluation"))

        # === 분석 그룹 ===
        analysis_group = QTabWidget()
        analysis_group.setTabPosition(QTabWidget.South)
        self._analysis_tab = AnalysisTab(self._config)
        analysis_group.addTab(self._analysis_tab, t("inference_analysis"))
        self._model_compare = ModelCompareView()
        analysis_group.addTab(self._model_compare, t("model_compare"))
        self._error_analyzer = ErrorAnalyzer()
        analysis_group.addTab(self._error_analyzer, t("fp_fn"))
        self._conf_optimizer = ConfOptimizer()
        analysis_group.addTab(self._conf_optimizer, t("conf_opt"))
        self._embedding_viewer = EmbeddingViewer()
        analysis_group.addTab(self._embedding_viewer, "Embedding")
        self._tabs.addTab(analysis_group, t("analysis"))

        # === 데이터 그룹 ===
        data_group = QTabWidget()
        data_group.setTabPosition(QTabWidget.South)
        self._dataset_explorer = DatasetExplorer()
        data_group.addTab(self._dataset_explorer, t("explorer"))
        self._dataset_splitter = DatasetSplitter()
        data_group.addTab(self._dataset_splitter, t("splitter"))
        data_group.addTab(ImageQualityChecker(), t("quality"))
        data_group.addTab(NearDuplicateDetector(), t("duplicates"))
        data_group.addTab(LabelAnomalyDetector(), t("label_anomaly"))
        data_group.addTab(FormatConverter(), t("format_conv"))
        data_group.addTab(AugmentationPreview(), t("augmentation"))
        data_group.addTab(ClassRemapper(), t("class_remap"))
        data_group.addTab(SimilaritySearch(), t("similarity"))
        data_group.addTab(SmartSampler(), t("sampler"))
        data_group.addTab(DatasetMerger(), t("merger"))
        data_group.addTab(LeakySplitDetector(), t("leaky_split"))
        self._tabs.addTab(data_group, t("data"))

        # 상태바
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._det_label = QLabel(f"{t('detection')}: 0")
        self._statusbar.addPermanentWidget(self._det_label)

    # ------------------------------------------------------------------ #
    # 시그널 연결
    # ------------------------------------------------------------------ #
    def _connect_signals(self):
        self._file_browser.video_selected.connect(self._on_video_selected)
        self._file_browser.model_selected.connect(self._on_model_selected)
        self._file_browser.model_type_changed.connect(self._on_settings_changed)
        self._file_browser.conf_changed.connect(self._on_conf_changed_rerun)
        self._video_widget.file_dropped.connect(self._file_browser.add_external_file)

        self._control_bar.play_clicked.connect(self._on_play)
        self._control_bar.pause_clicked.connect(self._on_pause)
        self._control_bar.stop_clicked.connect(self._on_stop)
        self._control_bar.seek_requested.connect(self._on_seek)
        self._control_bar.snapshot_clicked.connect(self._on_snapshot)
        self._control_bar.record_toggled.connect(self._on_record_toggled)
        self._control_bar.speed_changed.connect(self._on_speed_changed)
        self._control_bar.csv_record_toggled.connect(self._on_toggle_csv_record)
        self._control_bar.csv_export_clicked.connect(self._on_export_csv)
        self._control_bar.step_forward.connect(self._on_step_forward)
        self._control_bar.step_backward.connect(self._on_step_backward)

        self._settings_tab.settings_changed.connect(self._on_settings_changed)
        self._settings_tab.custom_type_added.connect(self._file_browser.refresh_model_types)
        self._class_filter.filter_changed.connect(self._on_settings_changed)
        self._tabs.currentChanged.connect(self._on_tab_changed)

    # ------------------------------------------------------------------ #
    # 키보드 단축키
    # ------------------------------------------------------------------ #
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            if self._thread and self._thread.isRunning():
                if self._thread._paused:
                    self._on_play()
                else:
                    self._on_pause()
            else:
                self._on_play()
        elif key == Qt.Key_Left:
            self._on_step_backward()
        elif key == Qt.Key_Right:
            self._on_step_forward()
        elif key == Qt.Key_S:
            self._on_snapshot()
        elif key == Qt.Key_R:
            self._control_bar._btn_record.click()
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            speeds = [1, 2, 3, 4, 8]
            cur = int(self._thread._speed) if self._thread else 1
            nxt = next((s for s in speeds if s > cur), speeds[-1])
            self._control_bar._speed_combo.setCurrentText(f"x{nxt}")
        elif key == Qt.Key_Minus:
            speeds = [1, 2, 3, 4, 8]
            cur = int(self._thread._speed) if self._thread else 1
            prev = next((s for s in reversed(speeds) if s < cur), speeds[0])
            self._control_bar._speed_combo.setCurrentText(f"x{prev}")
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------ #
    # 모델 / 비디오 로드
    # ------------------------------------------------------------------ #
    def _on_model_selected(self, path: str):
        try:
            model_type = self._config.model_type
            custom_name = ""
            if model_type.startswith("custom:"):
                custom_name = model_type.split(":", 1)[1]
                model_type = "custom"
            info = load_model(path, model_type=model_type,
                              pt_convert_callback=self._pt_convert_dialog)
            if custom_name:
                info.custom_type_name = custom_name
                cmt = self._config.custom_model_types.get(custom_name)
                if cmt and cmt.class_names:
                    info.names = cmt.class_names
            self._model_info = info
            self._config.init_class_styles(info.names)
            self._class_filter.populate(info.names)
            self._settings_tab.populate_classes(info.names)
            self._stats_widget.set_model_info(info)
            self._analysis_tab.set_model_info(info)
            self._status(f"모델 로드: {os.path.basename(path)}  |  클래스: {len(info.names)}개")
            # 이미지가 이미 로드되어 있으면 자동 추론
            if self._video_path and os.path.splitext(self._video_path)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                self._infer_single_image()
        except Exception as e:
            QMessageBox.critical(self, "모델 로드 실패", str(e))

    def _on_video_selected(self, path: str):
        self._stop_thread()
        self._video_path = path
        self._stats_widget.set_video_info(path)
        self._status(f"비디오: {os.path.basename(path)}")
        # 첫 번째 프레임 표시
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                from core.inference import DetectionResult
                self._last_frame = frame
                self._last_result = DetectionResult.empty()
                self._video_widget._set_pixmap_from_bgr(frame)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self._control_bar.set_total_frames(total)
                self._control_bar.update_position(0, total)
            cap.release()
        # 이미지 파일이면 자동 추론
        ext = os.path.splitext(path)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp"} and self._model_info:
            self._infer_single_image()

    def _pt_convert_dialog(self, pt_path: str, names: dict):
        """PT 파일 → ONNX 변환 여부 묻기"""
        msg = QMessageBox(self)
        msg.setWindowTitle("PT 모델 감지")
        msg.setText(
            f"'{os.path.basename(pt_path)}'은 PyTorch PT 파일입니다.\n\n"
            "추론을 위해 ONNX로 변환하시겠습니까?\n"
            "(ultralytics AGPL-3.0 라이센스 사용)"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)
        if msg.exec() != QMessageBox.Yes:
            return None

        onnx_path = os.path.splitext(pt_path)[0] + ".onnx"
        try:
            from ultralytics import YOLO  # type: ignore
            model = YOLO(pt_path)
            model.export(format="onnx", imgsz=640)
            if os.path.isfile(onnx_path):
                return onnx_path
        except ImportError:
            QMessageBox.warning(
                self, "ultralytics 미설치",
                "ultralytics 패키지가 설치되어 있지 않습니다.\n"
                "pip install ultralytics 후 다시 시도하세요."
            )
        except Exception as e:
            QMessageBox.critical(self, "변환 실패", str(e))
        return None

    # ------------------------------------------------------------------ #
    # 재생 제어
    # ------------------------------------------------------------------ #
    def _on_play(self):
        if self._thread is not None and self._thread.isRunning():
            self._thread.resume()
            self._control_bar.set_playing(True)
            return

        if not self._video_path:
            QMessageBox.information(self, "알림", "비디오를 먼저 선택해주세요.")
            return
        if self._model_info is None:
            QMessageBox.information(self, "알림", "모델을 먼저 선택해주세요.")
            return

        self._start_thread()

    def _on_pause(self):
        if self._thread:
            self._thread.pause()
            self._control_bar.set_playing(False)

    def _on_stop(self):
        self._stop_thread()
        self._control_bar.set_playing(False)
        self._control_bar.update_position(0, 0)

    def _on_seek(self, frame_idx: int):
        if self._thread:
            self._thread.seek(frame_idx)

    def _on_step_forward(self):
        if self._thread:
            self._thread.step_forward()

    def _on_step_backward(self):
        if self._thread:
            self._thread.step_backward()

    def _on_speed_changed(self, speed: float):
        if self._thread:
            self._thread.set_speed(speed)

    def _start_thread(self):
        self._thread = DetectThread(self._video_path, self._model_info, self._config)
        self._thread.frame_ready.connect(self._on_frame_ready)
        self._thread.fps_updated.connect(self._on_fps_updated)
        self._thread.progress_updated.connect(self._on_progress)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.error.connect(lambda msg: QMessageBox.critical(self, "오류", msg))

        # 총 프레임 수 사전 파악
        cap = cv2.VideoCapture(self._video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self._control_bar.set_total_frames(total)

        self._thread.start()
        self._control_bar.set_playing(True)

    def _stop_thread(self):
        if self._thread:
            self._thread.stop()
            self._thread.wait(3000)
            self._thread = None
        if self._recorder:
            self._recorder.release()
            self._recorder = None

    # ------------------------------------------------------------------ #
    # 스레드 콜백
    # ------------------------------------------------------------------ #
    def _on_frame_ready(self, frame, result):
        self._last_frame = frame
        self._last_result = result
        names = self._model_info.names if self._model_info else {}

        # CSV 기록은 항상 수행
        if isinstance(result, DetectionResult):
            if self._csv_recording and self._model_info:
                cur = getattr(self._thread, "_current_frame", 0) if self._thread else 0
                for box, score, cid in zip(result.boxes, result.scores, result.class_ids):
                    cid = int(cid)
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    self._csv_rows.append([cur, cid, names.get(cid, str(cid)),
                                            x1, y1, x2, y2, f"{float(score):.4f}"])

            if self._recorder and self._recorder.isOpened():
                drawn = self._video_widget._draw_detections(frame.copy(), result, self._config, names)
                self._recorder.write(drawn)

        # 뷰어 탭이 아니거나 프레임이 너무 빠르면 렌더링 스킵
        now = time.perf_counter()
        if self._tabs.currentIndex() != 0:
            return
        if (now - self._last_render_time) < self._MIN_RENDER_INTERVAL:
            return
        self._last_render_time = now

        if isinstance(result, ClassificationResult):
            vis = frame.copy()
            top_k = result.top_k[:5]
            y = 30
            for cid, conf in top_k:
                label = names.get(cid, str(cid))
                text = f"{label}: {conf:.3f}"
                cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y += 30
            self._video_widget._set_pixmap_from_bgr(vis)
            best_name = names.get(result.class_id, str(result.class_id))
            self._det_label.setText(f"분류: {best_name} ({result.confidence:.3f})")
        else:
            self._video_widget.display_frame(frame, result, self._config, names)
            if len(result.boxes) > 0:
                from collections import Counter
                counts = Counter(int(c) for c in result.class_ids)
                parts = [f"{names.get(cid, str(cid))}:{cnt}" for cid, cnt in counts.most_common(5)]
                detail = " | ".join(parts)
                self._det_label.setText(f"탐지: {len(result.boxes)}개  ({detail})")
            else:
                self._det_label.setText("탐지: 0개")

    def _on_fps_updated(self, video_fps: float, infer_fps: float, infer_ms: float):
        self._control_bar.update_fps(video_fps, infer_fps, infer_ms)
        self._stats_widget.update_infer_stats(video_fps, infer_fps, infer_ms)

    def _on_progress(self, current: int, total: int):
        self._control_bar.update_position(current, total)

    def _on_thread_finished(self):
        self._control_bar.set_playing(False)
        self._control_bar.seek_to_end()
        if self._recorder:
            self._recorder.release()
            self._recorder = None
        self._status(t("playback_done"))

    # ------------------------------------------------------------------ #
    # 스냅샷 / 녹화
    # ------------------------------------------------------------------ #
    def _on_snapshot(self):
        if self._last_frame is None:
            return
        names = self._model_info.names if self._model_info else {}
        frame = self._last_frame.copy()
        if self._last_result:
            frame = self._video_widget._draw_detections(frame, self._last_result, self._config, names)
        path = self._video_widget.snapshot("snapshots")
        if not path:
            os.makedirs("snapshots", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join("snapshots", f"snapshot_{ts}.jpg")
            cv2.imwrite(path, frame)
        self._status(f"스냅샷 저장: {path}")

    def _on_record_toggled(self, is_recording: bool):
        if is_recording:
            if self._video_path is None:
                QMessageBox.information(self, "알림", "비디오를 먼저 선택해주세요.")
                self._control_bar._btn_record.setChecked(False)
                return
            if self._model_info is None:
                QMessageBox.information(self, "알림", "모델을 먼저 선택해주세요.")
                self._control_bar._btn_record.setChecked(False)
                return

            # 재생 중이 아니면 자동 재생 시작
            if self._thread is None or not self._thread.isRunning():
                self._start_thread()
            elif self._thread._paused:
                self._thread.resume()
                self._control_bar.set_playing(True)

            # 녹화 시작 (프레임이 아직 없으면 첫 프레임 올 때까지 대기)
            cap = cv2.VideoCapture(self._video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            os.makedirs("recordings", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join("recordings", f"record_{ts}.mp4")
            self._recorder = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
            self._status(f"녹화 시작: {out_path}")
        else:
            if self._recorder:
                self._recorder.release()
                self._recorder = None
            self._status("녹화 중지 — 파일 저장 완료")

    # ------------------------------------------------------------------ #
    # CSV 내보내기
    # ------------------------------------------------------------------ #
    def _on_toggle_csv_record(self, checked: bool):
        self._csv_recording = checked
        if checked:
            self._csv_rows = []
            self._status(t("csv_record_start"))
        else:
            self._status(t("csv_record_stop", n=len(self._csv_rows)))

    def _toggle_dark_mode(self, checked):
        self._dark_mode = checked
        theme.set_dark(checked)
        self.setStyleSheet(theme.generate_qss(dark=checked))

    def _change_language(self, lang):
        set_language(lang)
        # rebuild UI by refreshing tab titles
        self._build_ui()
        self._connect_signals()

    def _on_export_csv(self):
        if not self._csv_rows:
            QMessageBox.information(self, t("notice"), t("csv_empty"))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "CSV", f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "class_id", "class_name", "x1", "y1", "x2", "y2", "score"])
            writer.writerows(self._csv_rows)
        self._status(t("csv_saved", path=path, n=len(self._csv_rows)))

    # ------------------------------------------------------------------ #
    # 기타
    # ------------------------------------------------------------------ #
    def _on_settings_changed(self):
        if self._last_frame is not None and self._last_result is not None and self._model_info:
            self._video_widget.display_frame(
                self._last_frame, self._last_result,
                self._config, self._model_info.names
            )

    def _on_tab_changed(self, index: int):
        """뷰어 탭으로 돌아오면 마지막 프레임 다시 렌더링"""
        if index == 0 and self._last_frame is not None and self._last_result is not None and self._model_info:
            self._video_widget.display_frame(
                self._last_frame, self._last_result,
                self._config, self._model_info.names
            )

    def _infer_single_image(self):
        """이미지 파일에 대해 현재 conf_threshold로 추론 실행"""
        if self._last_frame is None or self._model_info is None:
            return
        from core.inference import run_inference, run_classification
        names = self._model_info.names or {}
        if self._model_info.task_type == "classification":
            result = run_classification(self._model_info, self._last_frame)
            self._on_frame_ready(self._last_frame, result)
        else:
            result = run_inference(self._model_info, self._last_frame,
                                   self._config.conf_threshold)
            self._last_result = result
            self._video_widget.display_frame(
                self._last_frame, result, self._config, names)
            if len(result.boxes) > 0:
                from collections import Counter
                counts = Counter(int(c) for c in result.class_ids)
                parts = [f"{names.get(cid, str(cid))}:{cnt}" for cid, cnt in counts.most_common(5)]
                self._det_label.setText(f"탐지: {len(result.boxes)}개  ({' | '.join(parts)})")
            else:
                self._det_label.setText("탐지: 0개")

    def _on_conf_changed_rerun(self, conf: float):
        """conf threshold 변경 시 이미지면 재추론"""
        if self._video_path and os.path.splitext(self._video_path)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            self._infer_single_image()

    def _status(self, msg: str):
        self._statusbar.showMessage(msg)

    def closeEvent(self, event):
        self._stop_thread()
        self._config.save()
        super().closeEvent(event)
