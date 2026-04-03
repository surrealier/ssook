"""QThread: 비디오 읽기 + 추론 루프"""
import collections
import time

import cv2


def _read_video_fps(cap: cv2.VideoCapture) -> float:
    """HEVC timebase artifact(예: 600fps) 우회: 실제 프레임 간격으로 FPS 측정."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if 0 < fps <= 120:
        return fps

    # timebase가 비정상(>120)이면 실제 타임스탬프로 측정
    saved = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    timestamps = []
    for _ in range(8):
        ret, _ = cap.read()
        if not ret:
            break
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamps.append(ms)
    cap.set(cv2.CAP_PROP_POS_FRAMES, saved)

    if len(timestamps) >= 3:
        deltas = [
            timestamps[i + 1] - timestamps[i]
            for i in range(len(timestamps) - 1)
            if timestamps[i + 1] > timestamps[i]
        ]
        if deltas:
            avg_ms = sum(deltas) / len(deltas)
            measured = 1000.0 / avg_ms
            if 1.0 < measured <= 120.0:
                return measured

    return fps or 30.0
from PySide6.QtCore import QThread, Signal

from core.app_config import AppConfig
from core.inference import DetectionResult, ClassificationResult, run_inference, run_inference_batch, run_classification
from core.model_loader import ModelInfo


class DetectThread(QThread):
    frame_ready = Signal(object, object)      # (frame_bgr: np.ndarray, DetectionResult)
    fps_updated = Signal(float, float, float) # (video_fps, infer_fps, infer_ms)
    progress_updated = Signal(int, int)       # (current_frame, total_frames)
    finished = Signal()
    error = Signal(str)

    def __init__(self, video_path: str, model_info: ModelInfo, config: AppConfig):
        super().__init__()
        self.video_path = video_path
        self.model_info = model_info
        self.config = config

        self._paused = False
        self._stopped = False
        self._seek_to: int | None = None
        self._speed: float = 1.0

    # --- 제어 ---
    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._stopped = True
        self._paused = False

    def seek(self, frame_idx: int):
        self._seek_to = frame_idx

    def set_speed(self, speed: float):
        self._speed = max(0.1, speed)

    def step_forward(self):
        """1프레임 전진 (일시정지 상태에서만)"""
        self._seek_to = getattr(self, "_current_frame", 0) + 1

    def step_backward(self):
        """1프레임 후진 (일시정지 상태에서만)"""
        self._seek_to = max(0, getattr(self, "_current_frame", 0) - 1)

    # --- 메인 루프 ---
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit(f"비디오를 열 수 없습니다: {self.video_path}")
            return

        video_fps = min(_read_video_fps(cap), 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_delay = 1.0 / video_fps
        self._current_frame = 0

        infer_times = []
        frame_timestamps: collections.deque = collections.deque(maxlen=30)

        while not self._stopped:
            # seek 처리
            if self._seek_to is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_to)
                self._current_frame = self._seek_to
                self._seek_to = None

            # 일시정지
            if self._paused:
                time.sleep(0.03)
                continue

            t_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break

            self._current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            # 추론 (배치 처리)
            if self.model_info.session is not None:
                batch_size = self.config.batch_size
                # 고정 배치 모델이면 그 크기 사용
                expected = self.model_info.session.get_inputs()[0].shape[0]
                if isinstance(expected, int) and expected > 1:
                    batch_size = expected

                if batch_size > 1:
                    # 배치: 현재 프레임 + 추가 프레임 읽기
                    batch_frames = [frame]
                    for _ in range(batch_size - 1):
                        if self._stopped:
                            break
                        r2, f2 = cap.read()
                        if not r2:
                            break
                        batch_frames.append(f2)
                    results = run_inference_batch(
                        self.model_info, batch_frames, self.config.conf_threshold)
                    # 프레임 간 딜레이를 넣어 부드러운 재생
                    batch_base = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - len(batch_frames)
                    emit_t0 = time.perf_counter()
                    for j, (bf, br) in enumerate(zip(batch_frames, results)):
                        if self._stopped:
                            break
                        self._current_frame = batch_base + j
                        infer_times.append(br.infer_ms)
                        if len(infer_times) > 30:
                            infer_times.pop(0)
                        self.frame_ready.emit(bf, br)
                        self.progress_updated.emit(self._current_frame, total_frames)
                        # 다음 프레임 표시 시점까지 대기
                        target = emit_t0 + (j + 1) * frame_delay / self._speed
                        wait = target - time.perf_counter()
                        if wait > 0:
                            time.sleep(wait)
                    avg_infer = sum(infer_times) / len(infer_times) if infer_times else 0
                    infer_fps = 1000.0 / avg_infer if avg_infer > 0 else 0.0
                    self._current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    # 배치 내에서 이미 타이밍 처리했으므로 바깥 sleep 건너뜀
                    frame_timestamps.append(time.perf_counter())
                    if len(frame_timestamps) >= 2:
                        span = frame_timestamps[-1] - frame_timestamps[0]
                        actual_fps = (len(frame_timestamps) - 1) / span if span > 0 else 0.0
                    else:
                        actual_fps = 0.0
                    self.fps_updated.emit(actual_fps, infer_fps, avg_infer)
                    continue
                else:
                    if self.model_info.task_type == "classification":
                        cls_result = run_classification(self.model_info, frame)
                        result = DetectionResult.empty()
                        result.infer_ms = cls_result.infer_ms
                        # Classification 결과를 frame_ready로 전달
                        self.frame_ready.emit(frame, cls_result)
                    else:
                        result = run_inference(
                            self.model_info, frame, self.config.conf_threshold)
                        self.frame_ready.emit(frame, result)
                    infer_times.append(result.infer_ms if self.model_info.task_type != "classification" else cls_result.infer_ms)
                    if len(infer_times) > 30:
                        infer_times.pop(0)
                    avg_infer = sum(infer_times) / len(infer_times)
                    infer_fps = 1000.0 / avg_infer if avg_infer > 0 else 0.0
                    self.progress_updated.emit(self._current_frame, total_frames)
            else:
                result = DetectionResult.empty()
                infer_fps = 0.0
                avg_infer = 0.0
                self.frame_ready.emit(frame, result)
                self.progress_updated.emit(self._current_frame, total_frames)

            # 실제 렌더링 FPS 측정 (메타데이터 FPS 대신)
            frame_timestamps.append(time.perf_counter())
            if len(frame_timestamps) >= 2:
                span = frame_timestamps[-1] - frame_timestamps[0]
                actual_fps = (len(frame_timestamps) - 1) / span if span > 0 else 0.0
            else:
                actual_fps = 0.0
            self.fps_updated.emit(actual_fps, infer_fps, avg_infer)

            # 재생 속도 제어
            elapsed = time.perf_counter() - t_start
            sleep_time = frame_delay / self._speed - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()
        self.finished.emit()
