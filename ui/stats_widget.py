"""성능 통계 패널: GPU/CPU/추론 FPS/모델 정보/시스템 정보"""
import os
import platform
import subprocess
import sys

import psutil
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout,
)

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _nvidia_smi_query() -> dict | None:
    """nvidia-smi로 GPU 정보를 조회. 실패 시 None 반환."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
            creationflags=_CREATE_NO_WINDOW,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        return {
            "name": parts[0],
            "util": int(parts[1]),
            "mem_used": int(parts[2]),
            "mem_total": int(parts[3]),
            "temp": int(parts[4]),
        }
    except Exception:
        return None


def _get_system_info() -> dict:
    info = {}
    info["os"] = platform.system() + " " + platform.release()
    info["python"] = sys.version.split()[0]
    try:
        import onnxruntime as ort
        info["ort"] = ort.__version__
    except Exception:
        info["ort"] = "N/A"
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda or "N/A"
    except Exception:
        info["torch"] = "N/A"
        info["cuda"] = "N/A"
    gpu = _nvidia_smi_query()
    info["gpu_name"] = gpu["name"] if gpu else "N/A"
    return info


class StatsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(210)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # --- 모델 정보 ---
        model_group = QGroupBox("모델")
        model_form = QFormLayout(model_group)
        model_form.setSpacing(3)
        self._lbl_input = QLabel("—")
        self._lbl_output = QLabel("—")
        self._lbl_layout = QLabel("—")
        model_form.addRow("Input:", self._lbl_input)
        model_form.addRow("Output:", self._lbl_output)
        model_form.addRow("Layout:", self._lbl_layout)
        layout.addWidget(model_group)

        # --- 비디오 정보 ---
        video_group = QGroupBox("비디오")
        video_form = QFormLayout(video_group)
        video_form.setSpacing(3)
        self._lbl_vid_name = QLabel("—")
        self._lbl_vid_res = QLabel("—")
        self._lbl_vid_fps = QLabel("—")
        self._lbl_vid_frames = QLabel("—")
        self._lbl_vid_dur = QLabel("—")
        video_form.addRow("파일:", self._lbl_vid_name)
        video_form.addRow("해상도:", self._lbl_vid_res)
        video_form.addRow("FPS:", self._lbl_vid_fps)
        video_form.addRow("프레임:", self._lbl_vid_frames)
        video_form.addRow("길이:", self._lbl_vid_dur)
        layout.addWidget(video_group)

        # --- 추론 성능 ---
        infer_group = QGroupBox("추론")
        infer_form = QFormLayout(infer_group)
        infer_form.setSpacing(3)
        self._lbl_video_fps = QLabel("—")
        self._lbl_infer_fps = QLabel("—")
        self._lbl_infer_ms = QLabel("—")
        infer_form.addRow("영상 FPS:", self._lbl_video_fps)
        infer_form.addRow("추론 FPS:", self._lbl_infer_fps)
        infer_form.addRow("추론 ms:", self._lbl_infer_ms)
        layout.addWidget(infer_group)

        # --- 하드웨어 (현재 프로세스 기준) ---
        hw_group = QGroupBox("하드웨어 (프로세스)")
        hw_form = QFormLayout(hw_group)
        hw_form.setSpacing(3)
        self._lbl_cpu = QLabel("—")
        self._lbl_ram = QLabel("—")
        self._lbl_gpu_util = QLabel("—")
        self._lbl_gpu_mem = QLabel("—")
        self._lbl_gpu_temp = QLabel("—")
        hw_form.addRow("CPU:", self._lbl_cpu)
        hw_form.addRow("RAM:", self._lbl_ram)
        hw_form.addRow("GPU:", self._lbl_gpu_util)
        hw_form.addRow("VRAM:", self._lbl_gpu_mem)
        hw_form.addRow("GPU 온도:", self._lbl_gpu_temp)
        layout.addWidget(hw_group)

        # --- 시스템 정보 (정적) ---
        sys_group = QGroupBox("시스템")
        sys_form = QFormLayout(sys_group)
        sys_form.setSpacing(3)
        sinfo = _get_system_info()
        self._lbl_os = QLabel(sinfo["os"])
        self._lbl_python = QLabel(sinfo["python"])
        self._lbl_ort = QLabel(sinfo["ort"])
        self._lbl_torch = QLabel(sinfo["torch"])
        self._lbl_cuda = QLabel(sinfo["cuda"])
        self._lbl_gpu_name = QLabel(sinfo["gpu_name"])
        sys_form.addRow("OS:", self._lbl_os)
        sys_form.addRow("Python:", self._lbl_python)
        sys_form.addRow("ORT:", self._lbl_ort)
        sys_form.addRow("Torch:", self._lbl_torch)
        sys_form.addRow("CUDA:", self._lbl_cuda)
        sys_form.addRow("GPU:", self._lbl_gpu_name)
        layout.addWidget(sys_group)

        layout.addStretch()

        # 스타일: 작은 폰트
        for lbl in [
            self._lbl_input, self._lbl_output, self._lbl_layout,
            self._lbl_vid_name, self._lbl_vid_res, self._lbl_vid_fps,
            self._lbl_vid_frames, self._lbl_vid_dur,
            self._lbl_video_fps, self._lbl_infer_fps, self._lbl_infer_ms,
            self._lbl_cpu, self._lbl_ram,
            self._lbl_gpu_util, self._lbl_gpu_mem, self._lbl_gpu_temp,
            self._lbl_os, self._lbl_python, self._lbl_ort,
            self._lbl_torch, self._lbl_cuda, self._lbl_gpu_name,
        ]:
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size: 11px;")

        # 현재 프로세스 핸들
        self._proc = psutil.Process(os.getpid())
        self._proc.cpu_percent(interval=None)  # 첫 호출 초기화

        # GPU 사용 가능 여부 초기 확인
        self._has_gpu = _nvidia_smi_query() is not None
        if not self._has_gpu:
            for lbl in (self._lbl_gpu_util, self._lbl_gpu_mem, self._lbl_gpu_temp):
                lbl.setText("N/A")

        # 하드웨어 갱신 타이머 (500ms)
        self._hw_timer = QTimer(self)
        self._hw_timer.setInterval(500)
        self._hw_timer.timeout.connect(self._refresh_hw_stats)
        self._hw_timer.start()

    def set_video_info(self, path: str):
        """비디오 선택 시 파일 정보 표시"""
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # HEVC timebase artifact 우회: 실제 타임스탬프로 FPS 측정
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120:
                timestamps = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                for _ in range(8):
                    ret, _ = cap.read()
                    if not ret:
                        break
                    timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
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
                            fps = measured

            cap.release()
            dur_s = total / fps if fps > 0 else 0
            mins, secs = divmod(int(dur_s), 60)
            self._lbl_vid_name.setText(os.path.basename(path))
            self._lbl_vid_res.setText(f"{w} × {h}")
            self._lbl_vid_fps.setText(f"{fps:.2f}")
            self._lbl_vid_frames.setText(f"{total:,}")
            self._lbl_vid_dur.setText(f"{mins}:{secs:02d}")
        except Exception:
            for lbl in (self._lbl_vid_name, self._lbl_vid_res,
                        self._lbl_vid_fps, self._lbl_vid_frames, self._lbl_vid_dur):
                lbl.setText("—")

    def set_model_info(self, model_info):
        """모델 로드 시 input/output shape 표시"""
        try:
            inp = model_info.session.get_inputs()[0]
            out = model_info.session.get_outputs()[0]
            self._lbl_input.setText(str(inp.shape))
            self._lbl_output.setText(str(out.shape))
            if model_info.model_type == "darknet":
                self._lbl_layout.setText("CENTERNET")
            else:
                self._lbl_layout.setText(model_info.output_layout.upper())
        except Exception:
            self._lbl_input.setText("—")
            self._lbl_output.setText("—")
            self._lbl_layout.setText("—")

    def update_infer_stats(self, video_fps: float, infer_fps: float, infer_ms: float):
        self._lbl_video_fps.setText(f"{video_fps:.1f}")
        self._lbl_infer_fps.setText(f"{infer_fps:.1f}")
        self._lbl_infer_ms.setText(f"{infer_ms:.1f} ms")

    def _refresh_hw_stats(self):
        # 현재 프로세스 CPU / RAM
        try:
            cpu = self._proc.cpu_percent(interval=None)
            mem_rss = self._proc.memory_info().rss / 1024 / 1024
            self._lbl_cpu.setText(f"{cpu:.1f}%")
            self._lbl_ram.setText(f"{mem_rss:.0f} MB")
        except Exception:
            self._lbl_cpu.setText("N/A")
            self._lbl_ram.setText("N/A")

        # GPU (nvidia-smi)
        if self._has_gpu:
            gpu = _nvidia_smi_query()
            if gpu:
                self._lbl_gpu_util.setText(f"{gpu['util']}%")
                self._lbl_gpu_mem.setText(f"{gpu['mem_used']} / {gpu['mem_total']} MB")
                self._lbl_gpu_temp.setText(f"{gpu['temp']}°C")
            else:
                self._lbl_gpu_util.setText("—")
                self._lbl_gpu_mem.setText("—")
                self._lbl_gpu_temp.setText("—")
