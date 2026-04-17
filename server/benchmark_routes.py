"""/api/benchmark/* 라우터."""
import os
import time as _time

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.benchmark_runner import BenchmarkConfig, run_benchmark_core
from server.state import bench_state, executor

router = APIRouter()

# ── Benchmark API (async) ──────────────────────────────
class BenchmarkRequest(BaseModel):
    models: list[str]
    iterations: int = 100
    input_size: int = 640
    codecs: list[str] = ["none"]


@router.post("/api/benchmark/run")
async def run_benchmark(req: BenchmarkRequest):
    if bench_state["running"]:
        return {"error": "Benchmark already running"}

    bench_state.update(running=True, progress=0, total=1, msg="Starting...", results=[])

    def _run():
        from core.benchmark_runner import BenchmarkConfig, run_benchmark_core
        bench_state.update(progress=0, total=0, msg="Starting...", results=[])

        codecs = req.codecs or ["none"]
        configs = []
        codec_map = {}  # config index -> codec name
        idx = 0
        for path in req.models:
            for codec in codecs:
                configs.append(BenchmarkConfig(
                    model_path=path, iterations=req.iterations,
                    warmup=300, src_hw=(1080, 1920),
                ))
                codec_map[idx] = codec
                idx += 1
        bench_state["total"] = sum(c.warmup + c.iterations for c in configs)

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

        result_idx = [0]

        def on_progress(done, total, msg):
            bench_state["progress"] = done
            bench_state["msg"] = msg

        def on_result(r):
            ci = result_idx[0]
            codec = codec_map.get(ci, "none")
            result_idx[0] += 1

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
                times = []
                for _ in range(100):
                    t0 = _time.perf_counter()
                    cap = cv2.VideoCapture(buf)
                    cap.read()
                    cap.release()
                    times.append((_time.perf_counter() - t0) * 1000.0)
                decode_ms = round(float(np.mean(times)), 3)

            total_with_decode = round(r.mean_total_ms + decode_ms, 2)
            fps_with_decode = round(r.batch_size * 1000.0 / total_with_decode, 1) if total_with_decode > 0 else 0

            bench_state["results"].append({
                "name": r.model_name, "codec": codec.upper() if codec != "none" else "Raw",
                "provider": r.provider,
                "fps": fps_with_decode if decode_ms > 0 else round(r.fps, 1),
                "avg": total_with_decode if decode_ms > 0 else round(r.mean_total_ms, 2),
                "decode_ms": decode_ms if decode_ms > 0 else "—",
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
                "gpu_mem_used": r.gpu_mem_used,
                "gpu_mem_total": r.gpu_mem_total,
            })

        def on_error(msg):
            bench_state["results"].append({"error": msg})

        try:
            run_benchmark_core(configs, on_progress, on_result, on_error, lambda: False)
        except Exception as e:
            bench_state["msg"] = f"Error: {e}"
        finally:
            bench_state["running"] = False
            # Cleanup temp video files
            for codec in ("h264", "h265"):
                buf = encoded_bufs.get(codec)
                if buf and isinstance(buf, str) and os.path.isfile(buf):
                    try: os.remove(buf)
                    except: pass
            if not bench_state["msg"].startswith("Error"):
                bench_state["msg"] = "Complete"

    executor.submit(_run)
    return {"ok": True, "msg": "Benchmark started"}


@router.get("/api/benchmark/status")
async def benchmark_status():
    return {
        "running": bench_state["running"],
        "progress": bench_state["progress"],
        "total": bench_state["total"],
        "msg": bench_state["msg"],
        "results": bench_state["results"],
    }

