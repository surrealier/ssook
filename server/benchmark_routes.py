"""/api/benchmark/* 라우터."""
import csv
import io
import os
import time as _time

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from core.benchmark_runner import BenchmarkConfig, run_benchmark_core
from server.path_safety import UnsafePathError, safe_model_file
from server.state import bench_state, executor

router = APIRouter()

# ── Benchmark API (async) ──────────────────────────────
class BenchmarkRequest(BaseModel):
    models: list[str]
    iterations: int = 100
    input_size: int = 640
    codecs: list[str] = ["none"]
    ep_key: str = "auto"  # auto | cuda | tensorrt | directml | openvino | coreml | cpu


@router.post("/api/benchmark/run")
async def run_benchmark(req: BenchmarkRequest):
    # Atomic guard: try_start flips running=True under the lock or returns False,
    # so two near-simultaneous POSTs can't both pass the check and clobber results.
    if not bench_state.try_start(progress=0, total=1, msg="Starting...", results=[]):
        return {"error": "Benchmark already running"}

    def _run():
        from core.benchmark_runner import BenchmarkConfig, run_benchmark_core
        # The live results list is owned by the worker and only ever published to
        # bench_state via the locked __setitem__ below, so a status reader taking
        # snapshot() sees a whole list — never one mid-append (torn) iteration.
        results: list = []

        def publish_results() -> None:
            # Reassign (locked __setitem__) rather than mutate in place: the reader's
            # snapshot() grabs the list reference under the same lock, so it observes
            # either the pre- or post-publish list, never a half-appended one.
            bench_state["results"] = list(results)

        bench_state.update(progress=0, total=0, msg="Starting...")
        publish_results()

        from core.app_config import AppConfig
        bench_cfg = AppConfig()
        configured_type = bench_cfg.model_type
        codecs = req.codecs or ["none"]
        configs = []
        codec_map = {}  # config index -> codec name
        idx = 0
        skipped: list = []
        for raw_path in req.models:
            # 사용자 제공 모델 경로는 로드 전에 boundary에서 정규화·검증 (traversal/임의 경로 차단).
            try:
                path = safe_model_file(raw_path)
            except UnsafePathError as e:
                skipped.append({"error": f"{os.path.basename(str(raw_path))}: {e}"})
                continue
            # Detect model_type per model: if input has 9 channels, use configured seq_ type
            # otherwise use default "yolo"
            mt = "yolo"
            try:
                import onnxruntime as _ort
                _s = _ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                _inp = _s.get_inputs()[0]
                in_ch = int(_inp.shape[1]) if len(_inp.shape) >= 4 and isinstance(_inp.shape[1], int) else 3
                if in_ch == 9 and configured_type.startswith("seq_"):
                    mt = configured_type
                elif in_ch == 9:
                    mt = "seq_yolo"
                else:
                    mt = configured_type if not configured_type.startswith("seq_") else "yolo"
                del _s
            except Exception:
                mt = configured_type
            for codec in codecs:
                configs.append(BenchmarkConfig(
                    model_path=path, iterations=req.iterations,
                    warmup=300, src_hw=(1080, 1920),
                    model_type=mt, input_size=req.input_size,
                    ep_key=req.ep_key,
                ))
                codec_map[idx] = codec
                idx += 1
        bench_state["total"] = sum(c.warmup + c.iterations for c in configs)
        # 검증 실패한 모델 경로는 결과 목록에 에러로 노출 (조용히 누락하지 않음).
        for entry in skipped:
            results.append(entry)
        publish_results()

        # Pre-encode dummy frames per codec
        import time as _time
        _rng = np.random.default_rng(42)
        _dummy = _rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        encoded_bufs = {}
        for codec in codecs:
            if codec == "none":
                encoded_bufs[codec] = None
            elif codec == "jpeg":
                _, buf = cv2.imencode('.jpg', _dummy, [cv2.IMWRITE_JPEG_QUALITY, 95])
                encoded_bufs[codec] = buf
            elif codec == "png":
                _, buf = cv2.imencode('.png', _dummy)
                encoded_bufs[codec] = buf
            elif codec == "webp":
                _, buf = cv2.imencode('.webp', _dummy, [cv2.IMWRITE_WEBP_QUALITY, 95])
                encoded_bufs[codec] = buf
            elif codec in ("h264", "h265"):
                fourcc_map = {"h264": "avc1", "h265": "hev1"}
                import tempfile
                tmp = os.path.join(tempfile.gettempdir(), f"ssook_bench_{codec}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*fourcc_map[codec])
                writer = cv2.VideoWriter(tmp, fourcc, 30, (1920, 1080))
                if writer.isOpened():
                    writer.write(_dummy)
                    writer.release()
                    encoded_bufs[codec] = tmp
                else:
                    encoded_bufs[codec] = None

        def on_progress(done, total, msg):
            bench_state["progress"] = done
            bench_state["msg"] = msg

        def on_result(cfg_idx, r):
            # codec은 config 인덱스로 매핑 — 일부 모델이 건너뛰어져도 라벨이 밀리지 않는다.
            codec = codec_map.get(cfg_idx, "none")

            # Measure decode time for this codec
            decode_ms = 0.0
            buf = encoded_bufs.get(codec)
            if buf is not None and codec in ("jpeg", "png", "webp"):
                times = []
                for _ in range(100):
                    t0 = _time.perf_counter()
                    cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    times.append((_time.perf_counter() - t0) * 1000.0)
                decode_ms = round(float(np.mean(times)), 3)
            elif buf is not None and codec in ("h264", "h265"):
                # 컨테이너를 한 번만 열고 cap.read()만 타이밍 측정 — 매 반복 file-open은
                # 실제 스트림 디코드가 아닌 container-open 오버헤드를 잰다.
                cap = cv2.VideoCapture(buf)
                if cap.isOpened():
                    times = []
                    for _ in range(100):
                        t0 = _time.perf_counter()
                        ok, _frame = cap.read()
                        times.append((_time.perf_counter() - t0) * 1000.0)
                        # 단일 프레임 mp4이므로 EOF 도달 시 다음 read를 위해 되감는다.
                        if not ok:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    decode_ms = round(float(np.mean(times)), 3) if times else 0.0
                cap.release()

            # Fixed-batch 모델은 배치 내 프레임 수만큼 디코드가 필요하므로
            # decode_ms를 batch_size배 해야 throughput이 과대평가되지 않는다.
            total_with_decode = round(r.mean_total_ms + decode_ms * r.batch_size, 2)
            fps_with_decode = round(r.batch_size * 1000.0 / total_with_decode, 1) if total_with_decode > 0 else 0

            results.append({
                "name": r.model_name, "codec": codec.upper() if codec != "none" else "Raw",
                "status": "ok",
                "precision": r.precision,       # 모델 가중치 정밀도 (fp32/fp16/int8)
                "input_dtype": r.input_dtype,   # 세션이 받는 입력 dtype
                "provider": r.provider,         # 실제로 쓰인 EP
                "ep_requested": r.ep_requested, # 요청한 EP (provider와 다르면 fallback)
                "ep_fallback": r.ep_fallback,
                "fps": fps_with_decode if decode_ms > 0 else round(r.fps, 1),
                "avg": total_with_decode if decode_ms > 0 else round(r.mean_total_ms, 2),
                # decode 미측정(codec=Raw)은 None — UI는 '—'로 표시(`x.decode_ms||'—'`),
                # CSV는 빈칸이 되어 숫자 열에 글리프가 섞이지 않는다.
                "decode_ms": decode_ms if decode_ms > 0 else None,
                "pre_ms": round(r.mean_pre_ms, 2),
                "infer_ms": round(r.mean_infer_ms, 2),
                "post_ms": round(r.mean_post_ms, 2),
                "min": round(r.min_ms, 2),
                "max": round(r.max_ms, 2),
                "std": round(r.std_ms, 2),
                "p50": round(r.p50_ms, 2),
                "p95": round(r.p95_ms, 2),
                "p99": round(r.p99_ms, 2),
                "cpu_pct": round(r.cpu_pct, 1),
                "ram_mb": round(r.ram_mb),
                "gpu_pct": r.gpu_pct,
                "gpu_pct_peak": r.gpu_pct_peak,
                "gpu_mem_used": r.gpu_mem_used,
                "gpu_mem_total": r.gpu_mem_total,
            })
            publish_results()

        def on_error(cfg_idx, msg):
            # 건너뛴(로드 실패/OOM) 모델도 어느 모델/codec이었는지 결과에 남긴다.
            codec = codec_map.get(cfg_idx, "none")
            try:
                name = os.path.basename(configs[cfg_idx].model_path)
            except (IndexError, AttributeError):
                name = "—"
            results.append({
                "name": name,
                "codec": codec.upper() if codec != "none" else "Raw",
                "status": "failed",
                "error": msg,
            })
            publish_results()

        try:
            run_benchmark_core(configs, on_progress, on_result, on_error, lambda: not bench_state["running"])
        except Exception as e:
            bench_state["msg"] = f"Error: {e}"
        finally:
            bench_state["running"] = False
            # Cleanup temp video files
            for codec in ("h264", "h265"):
                buf = encoded_bufs.get(codec)
                if buf and isinstance(buf, str) and os.path.isfile(buf):
                    try: os.remove(buf)
                    except Exception:
                        pass
            if not bench_state["msg"].startswith("Error"):
                bench_state["msg"] = "Complete"

    executor.submit(_run)
    return {"ok": True, "msg": "Benchmark started"}


@router.get("/api/benchmark/stop")
async def benchmark_stop():
    bench_state["running"] = False
    return {"ok": True}


@router.get("/api/benchmark/status")
async def benchmark_status():
    # snapshot() copies under the lock so FastAPI serializes a stable dict/list —
    # never the live results list while the worker thread appends to it.
    snap = bench_state.snapshot()
    return {
        "running": snap.get("running", False),
        "progress": snap.get("progress", 0),
        "total": snap.get("total", 0),
        "msg": snap.get("msg", ""),
        "results": snap.get("results", []),
    }


@router.get("/api/benchmark/export-csv")
async def benchmark_export_csv():
    # Read via snapshot so we iterate a copied list, not the live one the worker appends to.
    results = bench_state.snapshot().get("results", [])

    def _cell(v):
        # None/em-dash -> 빈칸. decode 미측정·CPU 런의 GPU 컬럼 등이 숫자 열에 글리프('—')
        # 없이 비워진다('—'는 화면 표시 전용이며 CSV에는 절대 들어가지 않는다).
        return "" if v is None or v == "—" else v

    # 모든 행을 헤더와 동일한 키로 기록 → 열 정합 보장(밀림 불가). 실패 모델도
    # status=failed + error 사유로 포함(조용히 누락하지 않음). latency_ms_p*는 FPS가
    # 아니라 p50/p95/p99 지연(ms). precision=가중치 정밀도, input_dtype=세션 입력 dtype.
    fieldnames = [
        "model_name", "status", "precision", "input_dtype", "codec",
        "provider", "ep_requested",
        "fps_mean", "latency_ms_mean", "decode_ms", "preprocess_ms", "inference_ms",
        "postprocess_ms", "latency_ms_p50", "latency_ms_p95", "latency_ms_p99",
        "min_ms", "max_ms", "std_ms", "cpu_pct", "ram_mb",
        "gpu_pct", "gpu_pct_peak", "gpu_mem_used_mb", "error",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in results:
        failed = r.get("status") == "failed" or "error" in r
        writer.writerow({
            "model_name": r.get("name", ""),
            "status": r.get("status", "failed" if failed else "ok"),
            "precision": _cell(r.get("precision")),
            "input_dtype": _cell(r.get("input_dtype")),
            "codec": r.get("codec", ""),
            "provider": _cell(r.get("provider")),
            "ep_requested": _cell(r.get("ep_requested")),
            "fps_mean": _cell(r.get("fps")),
            "latency_ms_mean": _cell(r.get("avg")),
            "decode_ms": _cell(r.get("decode_ms")),
            "preprocess_ms": _cell(r.get("pre_ms")),
            "inference_ms": _cell(r.get("infer_ms")),
            "postprocess_ms": _cell(r.get("post_ms")),
            "latency_ms_p50": _cell(r.get("p50")),
            "latency_ms_p95": _cell(r.get("p95")),
            "latency_ms_p99": _cell(r.get("p99")),
            "min_ms": _cell(r.get("min")),
            "max_ms": _cell(r.get("max")),
            "std_ms": _cell(r.get("std")),
            "cpu_pct": _cell(r.get("cpu_pct")),
            "ram_mb": _cell(r.get("ram_mb")),
            "gpu_pct": _cell(r.get("gpu_pct")),
            "gpu_pct_peak": _cell(r.get("gpu_pct_peak")),
            "gpu_mem_used_mb": _cell(r.get("gpu_mem_used")),
            "error": r.get("error", ""),
        })
    # Excel이 UTF-8을 자동 인식하도록 utf-8-sig(BOM)로 인코딩.
    content = buf.getvalue().encode("utf-8-sig")
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=benchmark_results.csv"},
    )

