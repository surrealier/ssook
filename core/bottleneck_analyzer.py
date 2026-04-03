"""병목 분석기: ORT 프로파일링 + 시스템 메트릭 수집 + 병목 분류"""
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import psutil
import onnxruntime as ort
from PySide6.QtCore import QThread, Signal

from core.inference import letterbox, postprocess_darknet, postprocess_v5, postprocess_v8
from core.model_loader import load_model

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_CONF = 0.25   # 후처리용 고정 confidence

_ORT_DTYPE_MAP: dict = {
    "tensor(float)":   np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)":  np.float64,
    "tensor(int8)":    np.int8,
    "tensor(uint8)":   np.uint8,
    "tensor(int32)":   np.int32,
    "tensor(int64)":   np.int64,
}

_BOTTLENECK_LABELS = {
    "gpu_compute":   "GPU 계산 병목",
    "gpu_mem_bw":    "GPU 메모리 대역폭 병목",
    "pcie_transfer": "PCIe 전송 병목",
    "cpu_preprocess":"CPU 전처리 병목",
    "cpu_bound":     "CPU 병목",
    "balanced":      "균형 상태",
}

_RECOMMENDATIONS = {
    "gpu_compute": (
        "① TensorRT EP 또는 CUDA EP로 전환하여 GPU 연산 가속\n"
        "② 배치 크기 증가로 GPU 병렬 처리 극대화\n"
        "③ FP16/INT8 정밀도 적용으로 처리량 향상"
    ),
    "gpu_mem_bw": (
        "① 모델 양자화(INT8)로 메모리 전송량 감소\n"
        "② 채널 Pruning으로 Feature Map 크기 축소\n"
        "③ 입력 해상도 축소 검토"
    ),
    "pcie_transfer": (
        "① 배치 크기 증가로 전송 횟수 대비 처리량 향상\n"
        "② ORT I/O 바인딩(IOBinding) 활용 검토\n"
        "③ 입력 텐서 크기 축소 (해상도/채널 감소)"
    ),
    "cpu_preprocess": (
        "① 전처리를 GPU로 이동 (CuPy/CUDA 활용)\n"
        "② 전처리 멀티스레드 병렬화 검토\n"
        "③ 불필요한 색상 변환 단계 제거"
    ),
    "cpu_bound": (
        "① ORT 스레드 수 최적화 (intra/inter_op_num_threads)\n"
        "② CPU EP 최적화 옵션 적용 (AVX-512 등)\n"
        "③ 입력 해상도 축소 또는 모델 경량화"
    ),
    "balanced": (
        "① 현재 처리 시간이 고르게 분산됨\n"
        "② 배치 크기 조정으로 처리량 최적화\n"
        "③ 모델 경량화(Pruning/Distillation) 검토"
    ),
}


@dataclass
class BottleneckReport:
    model_name: str
    provider: str
    # 병목 진단
    bottleneck_type: str       # _BOTTLENECK_LABELS 키 중 하나
    bottleneck_score: float    # 0~1 (분류 신뢰도)
    recommendation: str        # 한국어 최적화 제안 텍스트
    # 단계별 평균 시간
    mean_pre_ms: float
    mean_infer_ms: float
    mean_post_ms: float
    mean_total_ms: float
    # 시스템 메트릭 (분석 중 평균)
    cpu_per_core: list = field(default_factory=list)   # list[float]
    gpu_sm_util: "int | None" = None
    gpu_mem_util: "int | None" = None   # GPU 메모리 대역폭 점유율 (%)
    pcie_rx_mbps: "float | None" = None
    pcie_tx_mbps: "float | None" = None
    pcie_gen: "int | None" = None
    pcie_width: "int | None" = None
    # ORT 프로파일 결과
    top_ops: list = field(default_factory=list)   # [(op_name, total_ms, pct), ...] 상위 15개
    total_nodes: int = 0
    profile_total_ms: float = 0.0
    # ONNX 그래프 병목 유형 분석
    onnx_bottleneck_ops: list = field(default_factory=list)  # [(op_type, count, names)]


# ── ORT 프로파일 파싱 ────────────────────────────────────────────────────────

# ONNX 그래프에서 병목/비효율 가능성이 있는 연산자 유형
_ONNX_BOTTLENECK_OPS = {
    "Gather":    "인덱싱 연산 — 불필요한 데이터 복사 유발, 상수 폴딩으로 제거 가능",
    "Identity":  "항등 연산 — 그래프 최적화로 제거 가능 (ORT graph_optimization_level 확인)",
    "Reshape":   "텐서 재배치 — 메모리 복사 유발 가능, 연속 Reshape 병합 검토",
    "Transpose": "축 전치 — 메모리 레이아웃 변경으로 캐시 미스 유발, NHWC↔NCHW 최소화",
    "Unsqueeze": "차원 확장 — 상수 폴딩 또는 그래프 단순화로 제거 가능",
    "Squeeze":   "차원 축소 — 상수 폴딩 또는 그래프 단순화로 제거 가능",
    "Cast":      "타입 변환 — FP32↔FP16 혼합 정밀도 시 빈번, 통일된 정밀도 사용 검토",
    "Concat":    "텐서 결합 — 메모리 할당/복사 비용, Skip Connection 구조 최적화 검토",
    "Slice":     "텐서 슬라이싱 — 불필요한 복사 유발, 인덱싱 패턴 최적화 검토",
    "Expand":    "브로드캐스트 확장 — 암시적 복사 유발, 연산 융합으로 제거 가능",
    "Pad":       "패딩 — Conv와 융합 가능 (ORT Conv+Pad fusion 확인)",
    "Split":     "텐서 분할 — 메모리 복사 비용, 분할 없는 구조로 변경 검토",
}


def _analyze_onnx_graph(model_path: str) -> list:
    """ONNX 그래프에서 병목/비효율 가능 연산자 분석 → [(op_type, count, [node_names])]"""
    try:
        import onnx
        model = onnx.load(model_path)
        op_nodes: dict = {}
        for node in model.graph.node:
            if node.op_type in _ONNX_BOTTLENECK_OPS:
                if node.op_type not in op_nodes:
                    op_nodes[node.op_type] = []
                op_nodes[node.op_type].append(node.name or node.output[0] if node.output else "?")
        result = [(op, len(names), names[:5]) for op, names in
                  sorted(op_nodes.items(), key=lambda x: -len(x[1]))]
        return result
    except Exception:
        return []

def _parse_ort_profile(path: str) -> "tuple[list, int, float]":
    """ORT 프로파일 JSON → (top_ops, total_nodes, profile_total_ms)"""
    try:
        with open(path, encoding="utf-8") as f:
            events = json.load(f)
    except Exception:
        return [], 0, 0.0

    op_totals: dict = {}
    total_nodes = 0

    for ev in events:
        # "ph": "X" = complete event (ts + dur 모두 있음)
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        if cat not in ("Node", "kernel", "Op"):
            continue
        total_nodes += 1
        args = ev.get("args", {})
        op = args.get("op_name") or args.get("node_name") or ev.get("name", "Unknown")
        dur = ev.get("dur", 0)
        op_totals[op] = op_totals.get(op, 0) + dur

    total_us = sum(op_totals.values())
    total_ms = total_us / 1000.0
    sorted_ops = sorted(op_totals.items(), key=lambda x: x[1], reverse=True)
    top_ops = [
        (name, dur_us / 1000.0, dur_us / total_us * 100 if total_us > 0 else 0.0)
        for name, dur_us in sorted_ops[:15]
    ]
    return top_ops, total_nodes, total_ms


# ── 시스템 메트릭 쿼리 ───────────────────────────────────────────────────────

def _query_gpu_extended() -> "dict | None":
    """nvidia-smi: SM 점유율 + 메모리 대역폭 점유율 + 온도"""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True, timeout=2, creationflags=_CREATE_NO_WINDOW,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        return {
            "sm_util":  int(parts[0]),
            "mem_util": int(parts[1]),
            "temp":     int(parts[2]),
        }
    except Exception:
        return None


def _query_pcie_dmon() -> "dict | None":
    """nvidia-smi dmon -s t -c 1 → PCIe TX/RX MB/s"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "dmon", "-s", "t", "-c", "1"],
            text=True, timeout=5, creationflags=_CREATE_NO_WINDOW,
        )
        for line in out.strip().splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    return {"tx_mbps": float(parts[1]), "rx_mbps": float(parts[2])}
                except ValueError:
                    continue
    except Exception:
        pass
    return None


def _query_pcie_link() -> "dict | None":
    """nvidia-smi: PCIe gen/width"""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=pcie.link.gen.current,pcie.link.width.current",
                "--format=csv,noheader,nounits",
            ],
            text=True, timeout=2, creationflags=_CREATE_NO_WINDOW,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        return {"gen": int(parts[0]), "width": int(parts[1])}
    except Exception:
        return None


# ── 병목 분류 ────────────────────────────────────────────────────────────────

def _classify_bottleneck(
    mean_pre_ms: float, mean_infer_ms: float, mean_post_ms: float,
    cpu_per_core: list,
    gpu_sm_util: "int | None",
    gpu_mem_util: "int | None",
    pcie_rx_mbps: "float | None",
) -> "tuple[str, float]":
    mean_total = mean_pre_ms + mean_infer_ms + mean_post_ms
    pre_ratio = mean_pre_ms / mean_total if mean_total > 0 else 0.0
    max_core = max(cpu_per_core) if cpu_per_core else 0.0
    pcie_rate_gbs = pcie_rx_mbps / 1024.0 if pcie_rx_mbps else 0.0

    candidates = []

    # GPU 계산 병목: SM 점유율 높고, 전처리 비중 낮고, PCIe 전송 적음
    if gpu_sm_util is not None and gpu_sm_util > 75 and pre_ratio < 0.25 and pcie_rate_gbs < 3:
        score = min((gpu_sm_util - 75) / 25.0, 1.0)
        candidates.append(("gpu_compute", score))

    # GPU 메모리 대역폭 병목: 메모리 점유율 높고, SM 점유율 낮음
    if gpu_mem_util is not None and gpu_mem_util > 85 and (gpu_sm_util or 0) < 70:
        score = min((gpu_mem_util - 85) / 15.0, 1.0)
        candidates.append(("gpu_mem_bw", score))

    # PCIe 전송 병목: RX 높고, SM 점유율 낮음
    if pcie_rx_mbps and pcie_rx_mbps > 2000 and (gpu_sm_util or 100) < 65:
        score = min(pcie_rx_mbps / 8000.0, 1.0)
        candidates.append(("pcie_transfer", score))

    # CPU 전처리 병목: 전처리 비중 크고, GPU 점유율 낮음
    if pre_ratio > 0.30 and (gpu_sm_util or 100) < 55:
        score = min((pre_ratio - 0.30) / 0.30, 1.0)
        candidates.append(("cpu_preprocess", score))

    # CPU 병목: 특정 코어 과부하, GPU 낮음
    if max_core > 85 and (gpu_sm_util or 100) < 40:
        score = min((max_core - 85) / 15.0, 1.0)
        candidates.append(("cpu_bound", score))

    if candidates:
        best = max(candidates, key=lambda x: x[1])
        return best[0], best[1]

    return "balanced", 0.5


# ── 핵심 로직 함수 ────────────────────────────────────────────────────────────

def run_bottleneck_core(
    model_path: str,
    model_type: str,
    batch_size: int,
    src_hw: "tuple[int, int]",
    on_progress,   # (done, total, msg) -> None
    on_report,     # (BottleneckReport) -> None
    on_error,      # (msg) -> None
    is_stopped,    # () -> bool
) -> None:
    """병목 분석 핵심 로직. QThread 및 ep_worker 서브프로세스 양쪽에서 사용."""
    model_name = os.path.basename(model_path)
    total = BottleneckAnalyzer._WARMUP + BottleneckAnalyzer._ITERATIONS

    # 1. 프로파일링 세션 생성
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        model_info = load_model(model_path, model_type, session_options=opts)
    except Exception as e:
        on_error(f"모델 로드 실패: {e}")
        return

    if model_info.session is None:
        on_error("세션 생성 실패")
        return

    session = model_info.session
    input_name = model_info.input_name
    model_h, model_w = model_info.input_size
    src_h, src_w = src_hw
    batch = batch_size
    is_darknet = model_info.model_type == "darknet"
    layout = model_info.output_layout

    providers = session.get_providers()
    provider = providers[0] if providers else "Unknown"

    ort_type = session.get_inputs()[0].type
    np_dtype = _ORT_DTYPE_MAP.get(ort_type, np.float32)

    # 배치 고정값 확인
    raw_batch = session.get_inputs()[0].shape[0]
    if isinstance(raw_batch, int) and raw_batch > 0 and raw_batch != batch:
        batch = raw_batch

    # PCIe 링크 정보 (정적 조회)
    pcie_link = _query_pcie_link()

    # 더미 프레임
    rng = np.random.default_rng(42)
    dummy = rng.integers(0, 256, (src_h, src_w, 3), dtype=np.uint8)
    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)

    # ── 전/후처리 헬퍼 ────────────────────────────────────────────────

    def _pre():
        if is_darknet:
            img = cv2.resize(dummy, (model_w, model_h))
            img = img[..., ::-1].transpose(2, 0, 1)
            t = np.ascontiguousarray(img[np.newaxis], dtype=np.float32) / 255.0
            r, p = 1.0, (0.0, 0.0)
        else:
            padded, r, p = letterbox(dummy, (model_h, model_w))
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

    def _post(out, ratio, pad):
        for b in range(batch):
            single = out[b: b + 1]
            if is_darknet:
                postprocess_darknet(single, _CONF, dummy.shape)
            elif layout == "v8":
                postprocess_v8(single, _CONF, ratio, pad, dummy.shape)
            else:
                postprocess_v5(single, _CONF, ratio, pad, dummy.shape)

    # 2. 워밍업
    done = 0
    _WARMUP = BottleneckAnalyzer._WARMUP
    _ITERATIONS = BottleneckAnalyzer._ITERATIONS

    for i in range(_WARMUP):
        if is_stopped():
            break
        tb, r, p = _pre()
        out_w = session.run(None, {input_name: tb})
        _post(out_w[0], r, p)
        done += 1
        if i % 10 == 0 or i == _WARMUP - 1:
            on_progress(done, total, f"워밍업 {i+1}/{_WARMUP}")

    if is_stopped():
        return

    # 3. 측정
    pre_times, infer_times, post_times = [], [], []
    cpu_samples, gpu_samples, pcie_samples = [], [], []

    for i in range(_ITERATIONS):
        if is_stopped():
            break

        t0 = time.perf_counter()
        tensor, ratio_b, pad_b = _pre()
        pre_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        model_out = session.run(None, {input_name: tensor})
        infer_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _post(model_out[0], ratio_b, pad_b)
        post_times.append((time.perf_counter() - t0) * 1000.0)

        done += 1
        if i % 10 == 0:
            cores = psutil.cpu_percent(interval=None, percpu=True)
            cpu_samples.append(cores)
            gpu = _query_gpu_extended()
            if gpu:
                gpu_samples.append(gpu)
            pcie = _query_pcie_dmon()
            if pcie:
                pcie_samples.append(pcie)
            on_progress(done, total, f"측정 {i+1}/{_ITERATIONS}")

    # 4. 프로파일 종료 + 파싱
    try:
        prof_path = session.end_profiling()
        top_ops, total_nodes, profile_total_ms = _parse_ort_profile(prof_path)
        try:
            os.unlink(prof_path)
        except Exception:
            pass
    except Exception:
        top_ops, total_nodes, profile_total_ms = [], 0, 0.0

    if not infer_times:
        on_error("측정 데이터 없음")
        return

    mean_pre   = float(np.mean(pre_times))
    mean_infer = float(np.mean(infer_times))
    mean_post  = float(np.mean(post_times))

    # CPU 코어별 평균
    if cpu_samples:
        n_cores = len(cpu_samples[0])
        cpu_avg = [
            float(np.mean([s[c] for s in cpu_samples]))
            for c in range(n_cores)
        ]
    else:
        cpu_avg = [float(proc.cpu_percent(interval=None))]

    gpu_sm   = int(np.mean([g["sm_util"]  for g in gpu_samples])) if gpu_samples else None
    gpu_mem  = int(np.mean([g["mem_util"] for g in gpu_samples])) if gpu_samples else None
    pcie_rx  = float(np.mean([p["rx_mbps"] for p in pcie_samples])) if pcie_samples else None
    pcie_tx  = float(np.mean([p["tx_mbps"] for p in pcie_samples])) if pcie_samples else None

    btype, bscore = _classify_bottleneck(
        mean_pre, mean_infer, mean_post, cpu_avg, gpu_sm, gpu_mem, pcie_rx,
    )

    report = BottleneckReport(
        model_name=model_name,
        provider=provider,
        bottleneck_type=btype,
        bottleneck_score=bscore,
        recommendation=_RECOMMENDATIONS[btype],
        mean_pre_ms=mean_pre,
        mean_infer_ms=mean_infer,
        mean_post_ms=mean_post,
        mean_total_ms=mean_pre + mean_infer + mean_post,
        cpu_per_core=cpu_avg,
        gpu_sm_util=gpu_sm,
        gpu_mem_util=gpu_mem,
        pcie_rx_mbps=pcie_rx,
        pcie_tx_mbps=pcie_tx,
        pcie_gen=pcie_link["gen"]   if pcie_link else None,
        pcie_width=pcie_link["width"] if pcie_link else None,
        top_ops=top_ops,
        total_nodes=total_nodes,
        profile_total_ms=profile_total_ms,
        onnx_bottleneck_ops=_analyze_onnx_graph(model_path),
    )
    on_report(report)


# ── BottleneckAnalyzer QThread ───────────────────────────────────────────────

class BottleneckAnalyzer(QThread):
    progress_updated = Signal(int, int, str)
    report_ready = Signal(object)      # BottleneckReport
    error = Signal(str)
    finished = Signal()

    _WARMUP = 50
    _ITERATIONS = 100

    def __init__(self, model_path: str, model_type: str = "yolo",
                 batch_size: int = 1, src_hw: "tuple[int, int]" = (1080, 1920),
                 ep_key: str = "auto"):
        super().__init__()
        self.model_path = model_path
        self.model_type = model_type
        self.batch_size = batch_size
        self.src_hw = src_hw
        self.ep_key = ep_key
        self._stopped = False

    def stop(self):
        self._stopped = True

    def _run_ep_subprocess(self) -> None:
        from core.ep_manager import get_ep_dir, _PROJECT_ROOT, launch_worker

        task = {
            "task": "bottleneck",
            "ep_key": self.ep_key,
            "ep_dir": str(get_ep_dir(self.ep_key)),
            "proj_root": str(_PROJECT_ROOT),
            "model_path": self.model_path,
            "model_type": self.model_type,
            "batch_size": self.batch_size,
            "src_hw": list(self.src_hw),
        }
        proc = launch_worker(task)

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
                self.progress_updated.emit(event["done"], event["total"], event["msg"])
            elif t == "report":
                try:
                    data = event["data"]
                    data["top_ops"] = [tuple(op) for op in data.get("top_ops", [])]
                    report = BottleneckReport(**data)
                    self.report_ready.emit(report)
                except Exception as exc:
                    self.error.emit(f"보고서 역직렬화 실패: {exc}")
            elif t == "error":
                self.error.emit(event["msg"])
            elif t == "finished":
                break

        proc.wait()

    def run(self):
        if self.ep_key != "auto":
            self._run_ep_subprocess()
            self.finished.emit()
            return

        run_bottleneck_core(
            model_path=self.model_path,
            model_type=self.model_type,
            batch_size=self.batch_size,
            src_hw=self.src_hw,
            on_progress=self.progress_updated.emit,
            on_report=self.report_ready.emit,
            on_error=self.error.emit,
            is_stopped=lambda: self._stopped,
        )
        self.finished.emit()
