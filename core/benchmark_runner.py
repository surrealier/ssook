"""벤치마크 러너: 전처리/추론/후처리 단계별 성능 측정 (Warmup 포함)"""
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
import psutil
try:
    from PySide6.QtCore import QThread, Signal
    _HAS_QT = True
except ImportError:
    _HAS_QT = False

from core.inference import (
    letterbox,
    postprocess_darknet,
    postprocess_v5,
    postprocess_v8,
)
from core.model_loader import load_model

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_CONF = 0.25   # 후처리용 고정 confidence (탐지 여부 평가 목적이 아닌 시간 측정)

_ORT_DTYPE_MAP: dict = {
    "tensor(float)":   np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)":  np.float64,
    "tensor(int8)":    np.int8,
    "tensor(uint8)":   np.uint8,
    "tensor(int32)":   np.int32,
    "tensor(int64)":   np.int64,
}


_smi_available = None  # None=미확인, True/False=캐싱

def _smi_query() -> "dict | None":
    global _smi_available
    if _smi_available is False:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
            creationflags=_CREATE_NO_WINDOW,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        _smi_available = True
        return {
            "util": int(parts[0]),
            "mem_used": int(parts[1]),
            "mem_total": int(parts[2]),
        }
    except Exception:
        _smi_available = False
        return None


@dataclass
class BenchmarkConfig:
    model_path: str
    model_type: str = "yolo"
    iterations: int = 500
    warmup: int = 300
    batch_size: int = 1
    src_hw: "tuple[int, int]" = (1080, 1920)  # 원본 입력 해상도 (H, W); 기본 1920×1080
    ep_key: str = "auto"


@dataclass
class BenchmarkResult:
    model_name: str
    model_type: str
    provider: str
    src_size: tuple          # 원본 프레임 해상도 (H, W)
    model_size: tuple        # 모델 입력 크기 (H, W)
    batch_size: int
    input_dtype: str
    warmup_count: int
    iter_count: int
    # 단계별 평균 시간
    mean_pre_ms: float       # 전처리 (resize/letterbox, 정규화, 배치화)
    mean_infer_ms: float     # 추론 (session.run)
    mean_post_ms: float      # 후처리 (decode, NMS)
    mean_total_ms: float     # 합계 = pre + infer + post
    # 총 시간 통계
    min_ms: float
    max_ms: float
    std_ms: float
    p50_ms: float            # 중앙값 (50번째 백분위수)
    p95_ms: float            # 95번째 백분위수
    p99_ms: float            # 99번째 백분위수
    fps: float               # 이미지/초 = batch_size × 1000 / mean_total_ms
    cpu_pct: float
    ram_mb: float
    gpu_pct: "int | None"
    gpu_mem_used: "int | None"
    gpu_mem_total: "int | None"


def run_benchmark_core(
    configs: list,
    on_progress,
    on_result,
    on_error,
    is_stopped,
) -> None:
    """
    벤치마크 핵심 로직. QThread 내 직접 실행 및 ep_worker 서브프로세스 양쪽에서 사용.
    on_progress(done, total, msg), on_result(BenchmarkResult),
    on_error(msg), is_stopped() -> bool
    """
    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)

    total_steps = sum(cfg.warmup + cfg.iterations for cfg in configs)
    done = 0

    for cfg in configs:
        if is_stopped():
            break

        model_name = os.path.basename(cfg.model_path)

        try:
            model_info = load_model(cfg.model_path, cfg.model_type)
        except Exception as e:
            on_error(f"[{model_name}] 로드 실패: {e}")
            done += cfg.warmup + cfg.iterations
            on_progress(done, total_steps, f"[{model_name}] 로드 실패 — 건너뜀")
            continue

        if model_info.session is None:
            on_error(f"[{model_name}] 세션 없음 — 건너뜀")
            done += cfg.warmup + cfg.iterations
            on_progress(done, total_steps, f"[{model_name}] 세션 없음 — 건너뜀")
            continue

        src_h, src_w = cfg.src_hw
        model_h, model_w = model_info.input_size
        batch = cfg.batch_size

        raw_batch = model_info.session.get_inputs()[0].shape[0]
        if isinstance(raw_batch, int) and raw_batch > 0 and raw_batch != batch:
            on_progress(done, total_steps,
                        f"[{model_name}] 배치 {batch} → 모델 고정값 {raw_batch}으로 조정")
            batch = raw_batch

        ort_type = model_info.session.get_inputs()[0].type
        np_dtype = _ORT_DTYPE_MAP.get(ort_type, np.float32)
        dtype_label = ort_type.replace("tensor(", "").replace(")", "")

        providers = model_info.session.get_providers()
        provider = providers[0] if providers else "Unknown"
        layout = model_info.output_layout
        input_name = model_info.input_name
        is_darknet = model_info.model_type == "darknet"

        rng = np.random.default_rng(42)
        dummy_frame = rng.integers(0, 256, (src_h, src_w, 3), dtype=np.uint8)

        def _preprocess():
            if is_darknet:
                img = cv2.resize(dummy_frame, (model_w, model_h))
                img = img[..., ::-1].transpose(2, 0, 1)
                t = np.ascontiguousarray(img[np.newaxis], dtype=np.float32) / 255.0
                r, p = 1.0, (0.0, 0.0)
            else:
                padded, r, p = letterbox(dummy_frame, (model_h, model_w))
                rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
                t = np.ascontiguousarray(
                    rgb.transpose(2, 0, 1)[np.newaxis], dtype=np.float32
                ) / 255.0
            if np_dtype != np.float32:
                if np.issubdtype(np_dtype, np.floating):
                    t = t.astype(np_dtype)
                else:
                    t = (t * 255).clip(0, 255).astype(np_dtype)
            if batch > 1:
                t = np.repeat(t, batch, axis=0)
            return t, r, p

        def _postprocess(model_out, ratio, pad):
            for b in range(batch):
                single = model_out[b: b + 1]
                if is_darknet:
                    postprocess_darknet(single, _CONF, dummy_frame.shape)
                elif layout == "v8":
                    postprocess_v8(single, _CONF, ratio, pad, dummy_frame.shape)
                else:
                    postprocess_v5(single, _CONF, ratio, pad, dummy_frame.shape)

        # Warmup
        for i in range(cfg.warmup):
            if is_stopped():
                break
            t_b, r_w, p_w = _preprocess()
            out_w = model_info.session.run(None, {input_name: t_b})
            _postprocess(out_w[0], r_w, p_w)
            done += 1
            if i % 10 == 0 or i == cfg.warmup - 1:
                on_progress(done, total_steps,
                            f"[{model_name}]  워밍업  {i + 1} / {cfg.warmup}")

        if is_stopped():
            break

        # Benchmark
        pre_times: list = []
        infer_times: list = []
        post_times: list = []
        cpu_samples: list = []
        gpu_util_samples: list = []

        for i in range(cfg.iterations):
            if is_stopped():
                break

            t0 = time.perf_counter()
            tensor, ratio_b, pad_b = _preprocess()
            pre_times.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            model_out = model_info.session.run(None, {input_name: tensor})
            infer_times.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            _postprocess(model_out[0], ratio_b, pad_b)
            post_times.append((time.perf_counter() - t0) * 1000.0)

            done += 1
            if i % 20 == 0:
                cpu_samples.append(proc.cpu_percent(interval=None))
                gpu = _smi_query()
                if gpu:
                    gpu_util_samples.append(gpu["util"])
                on_progress(done, total_steps,
                            f"[{model_name}]  측정  {i + 1} / {cfg.iterations}")

        if not infer_times:
            continue

        pre_arr   = np.array(pre_times)
        infer_arr = np.array(infer_times)
        post_arr  = np.array(post_times)
        total_arr = pre_arr + infer_arr + post_arr
        mean_total = float(np.mean(total_arr))
        gpu_final = _smi_query()

        result = BenchmarkResult(
            model_name=model_name,
            model_type=cfg.model_type,
            provider=provider,
            src_size=(src_h, src_w),
            model_size=(model_h, model_w),
            batch_size=batch,
            input_dtype=dtype_label,
            warmup_count=cfg.warmup,
            iter_count=len(infer_times),
            mean_pre_ms=float(np.mean(pre_arr)),
            mean_infer_ms=float(np.mean(infer_arr)),
            mean_post_ms=float(np.mean(post_arr)),
            mean_total_ms=mean_total,
            min_ms=float(np.min(total_arr)),
            max_ms=float(np.max(total_arr)),
            std_ms=float(np.std(total_arr)),
            p50_ms=float(np.percentile(total_arr, 50)),
            p95_ms=float(np.percentile(total_arr, 95)),
            p99_ms=float(np.percentile(total_arr, 99)),
            fps=batch * 1000.0 / mean_total if mean_total > 0 else 0.0,
            cpu_pct=float(np.mean(cpu_samples)) if cpu_samples else 0.0,
            ram_mb=proc.memory_info().rss / 1024 / 1024,
            gpu_pct=(
                int(np.mean(gpu_util_samples)) if gpu_util_samples
                else (gpu_final["util"] if gpu_final else None)
            ),
            gpu_mem_used=gpu_final["mem_used"] if gpu_final else None,
            gpu_mem_total=gpu_final["mem_total"] if gpu_final else None,
        )
        on_result(result)


if _HAS_QT:
    class BenchmarkRunner(QThread):
        progress_updated = Signal(int, int, str)
        result_ready = Signal(object)
        finished = Signal()
        error = Signal(str)

        def __init__(self, configs: list):
            super().__init__()
            self.configs = configs
            self._stopped = False

        def stop(self):
            self._stopped = True

        def _run_ep_subprocess(self, ep_key: str, configs: list, total_steps: int, done_offset: int) -> int:
            """Returns updated done_offset after this EP group finishes."""
            import dataclasses
            from core.ep_manager import get_ep_dir, _PROJECT_ROOT, launch_worker

            cfg_dicts = []
            for c in configs:
                d = dataclasses.asdict(c)
                d["src_hw"] = list(d["src_hw"])
                cfg_dicts.append(d)

            task = {
                "task": "benchmark",
                "ep_key": ep_key,
                "ep_dir": str(get_ep_dir(ep_key)),
                "proj_root": str(_PROJECT_ROOT),
                "configs": cfg_dicts,
            }

            proc = launch_worker(task)
            local_done = 0

            for line in proc.stdout:
                if self._stopped:
                    proc.terminate()
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue

                t = event.get("type")
                if t == "progress":
                    local_done = event["done"]
                    self.progress_updated.emit(
                        done_offset + local_done, total_steps, event["msg"]
                    )
                elif t == "result":
                    try:
                        data = event["data"]
                        data["src_size"] = tuple(data["src_size"])
                        data["model_size"] = tuple(data["model_size"])
                        result = BenchmarkResult(**data)
                        self.result_ready.emit(result)
                    except Exception as exc:
                        self.error.emit(f"결과 역직렬화 실패: {exc}")
                elif t == "error":
                    self.error.emit(event["msg"])
                elif t == "finished":
                    break

            proc.wait()
            ep_steps = sum(c.warmup + c.iterations for c in configs)
            return done_offset + ep_steps

        def run(self):
            total_steps = sum(cfg.warmup + cfg.iterations for cfg in self.configs)
            global_done = 0

            seen_eps: list = []
            groups: dict = {}
            for cfg in self.configs:
                key = cfg.ep_key
                if key not in groups:
                    seen_eps.append(key)
                    groups[key] = []
                groups[key].append(cfg)

            for ep_key in seen_eps:
                if self._stopped:
                    break
                ep_configs = groups[ep_key]

                if ep_key == "auto":
                    _offset = global_done

                    def _prog(d: int, t: int, m: str, off: int = _offset) -> None:
                        self.progress_updated.emit(off + d, total_steps, m)

                    run_benchmark_core(
                        ep_configs,
                        on_progress=_prog,
                        on_result=self.result_ready.emit,
                        on_error=self.error.emit,
                        is_stopped=lambda: self._stopped,
                    )
                    global_done += sum(c.warmup + c.iterations for c in ep_configs)
                else:
                    global_done = self._run_ep_subprocess(ep_key, ep_configs, total_steps, global_done)

            self.finished.emit()
