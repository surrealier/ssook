"""Benchmark runner robustness: per-config callback indexing + graceful skip.

These guard the v1.6.0 fix where a skipped model (load failure / OOM) must NOT
shift codec labels of later results — callbacks now receive the config index,
and progress must still reach total even when models are skipped.

Also covers the benchmark route concurrency/serialization fixes:
- CSV export labels latency percentile columns as latency_ms_p* (not fps_p*).
- run_benchmark uses the atomic try_start guard (no double-start).
- benchmark_status serializes a snapshot, never the live (concurrently
  appended) results list — readers never see a torn list.
"""
import asyncio
import csv
import io
import threading

from core.benchmark_runner import BenchmarkConfig, run_benchmark_core
from core.model_loader import _build_providers


def _drive(configs):
    """Run the core with collecting callbacks; returns (result_idxs, error_idxs, progress)."""
    results: list = []
    errors: list = []
    progress: list = []
    run_benchmark_core(
        configs,
        on_progress=lambda d, t, m: progress.append((d, t)),
        on_result=lambda idx, r: results.append(idx),
        on_error=lambda idx, m: errors.append(idx),
        is_stopped=lambda: False,
    )
    return results, errors, progress


def test_failed_models_report_their_own_config_index():
    # Nonexistent paths → load_model raises → on_error(cfg_idx, msg) for each.
    cfgs = [BenchmarkConfig(model_path=f"/nonexistent/m{i}.onnx", warmup=1, iterations=1)
            for i in range(3)]
    results, errors, _ = _drive(cfgs)
    assert results == []           # nothing succeeded
    assert errors == [0, 1, 2]     # each config reported under its OWN index (no shift)


def test_progress_reaches_total_even_when_all_skipped():
    # Skipped models must still advance `done` so the progress bar completes.
    cfgs = [BenchmarkConfig(model_path=f"/nonexistent/m{i}.onnx", warmup=2, iterations=3)
            for i in range(4)]
    _, _, progress = _drive(cfgs)
    total = 4 * (2 + 3)
    assert progress, "expected progress callbacks"
    assert progress[-1][0] == total
    assert progress[-1][1] == total


def test_build_providers_cuda_options_backcompat():
    # Passing cuda_options must not change provider selection when CUDA EP is
    # absent (CPU-only env), and the no-arg call must stay identical to None.
    assert _build_providers() == _build_providers(None)
    # cuda_options is only ever applied to a real CUDAExecutionProvider; on a
    # CPU build the result still contains the CPU provider and never crashes.
    provs = _build_providers({"gpu_mem_limit": "123", "cudnn_conv_algo_search": "HEURISTIC"})
    flat = [p[0] if isinstance(p, tuple) else p for p in provs]
    assert "CPUExecutionProvider" in flat


# ── Route-level: CSV labels, atomic start guard, snapshot consistency ──
# These exercise server.benchmark_routes against the shared bench_state. They
# import the route coroutines directly (not via TestClient) to avoid cross-talk
# with other route modules under concurrent test runs.

def _reset_bench_state():
    from server.state import bench_state
    bench_state.update(running=False, progress=0, total=0, msg="", results=[])
    return bench_state


def test_csv_header_labels_latency_percentiles_not_fps():
    # The CSV percentile columns hold latency-in-ms (p*_ms), so they must be
    # labeled latency_ms_p*, NOT fps_p* (the prior mislabel). The only true FPS
    # column is fps_mean.
    from server import benchmark_routes

    bench_state = _reset_bench_state()
    bench_state["results"] = [{
        "name": "m.onnx", "codec": "Raw", "provider": "CPUExecutionProvider",
        "fps": 42.0, "avg": 23.8, "decode_ms": "—",
        "pre_ms": 1.0, "infer_ms": 20.0, "post_ms": 2.8,
        "min": 22.0, "max": 30.0, "std": 1.5,
        "p50": 23.5, "p95": 27.1, "p99": 29.0,
        "cpu_pct": 50.0, "ram_mb": 100,
    }]

    resp = asyncio.run(benchmark_routes.benchmark_export_csv())
    text = resp.body.decode("utf-8") if isinstance(resp.body, bytes) else resp.body
    rows = list(csv.DictReader(io.StringIO(text)))

    header = rows[0].keys() if rows else []
    assert "latency_ms_p50" in header
    assert "latency_ms_p95" in header
    assert "latency_ms_p99" in header
    # The mislabeled names must be gone.
    assert "fps_p50" not in header
    assert "fps_p95" not in header
    assert "fps_p99" not in header
    # fps_mean stays — it is genuinely FPS.
    assert "fps_mean" in header

    # Values under the percentile columns are the latency p*_ms, not FPS.
    row = rows[0]
    assert float(row["latency_ms_p50"]) == 23.5
    assert float(row["latency_ms_p95"]) == 27.1
    assert float(row["latency_ms_p99"]) == 29.0
    assert float(row["fps_mean"]) == 42.0
    _reset_bench_state()


def test_try_start_guard_blocks_second_concurrent_run():
    # Two near-simultaneous starts: exactly one wins the atomic guard. We call
    # try_start directly (the route's first action) to avoid spawning the real
    # benchmark worker.
    bench_state = _reset_bench_state()

    outcomes: list = []
    barrier = threading.Barrier(2)

    def attempt():
        barrier.wait()
        outcomes.append(bench_state.try_start(progress=0, total=1, msg="Starting...", results=[]))

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(outcomes) == [False, True]  # exactly one started
    assert bench_state["running"] is True
    _reset_bench_state()


def test_status_returns_snapshot_isolated_from_worker_publish():
    # benchmark_status serializes from snapshot(), and the worker publishes new
    # results by REASSIGNING the list (publish_results: bench_state['results'] =
    # list(...)), never by in-place append. So a status response captured before
    # a publish keeps the old whole list — the reader never sees a torn view.
    from server import benchmark_routes

    bench_state = _reset_bench_state()
    bench_state["results"] = [{"name": "a"}, {"name": "b"}]

    status = asyncio.run(benchmark_routes.benchmark_status())
    assert status["results"] == [{"name": "a"}, {"name": "b"}]

    # Worker's publish pattern: reassign a brand-new list (locked __setitem__).
    bench_state["results"] = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    # The earlier response is unaffected — it holds the previously published list.
    assert status["results"] == [{"name": "a"}, {"name": "b"}]
    # A fresh status call observes the new published list.
    status2 = asyncio.run(benchmark_routes.benchmark_status())
    assert status2["results"] == [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    _reset_bench_state()


def test_status_survives_concurrent_appends_no_torn_read():
    # Hammer benchmark_status while another thread reassigns the results list
    # (the worker's publish pattern). snapshot() copies under the lock, so no
    # 'list changed size during iteration' / partial read can occur.
    from server import benchmark_routes

    bench_state = _reset_bench_state()
    stop = threading.Event()

    def writer():
        acc: list = []
        i = 0
        while not stop.is_set():
            acc.append({"name": f"m{i}"})
            # Mirror the route's publish_results(): reassign via locked __setitem__.
            bench_state["results"] = list(acc)
            i += 1

    errors: list = []

    def reader():
        # Call snapshot() directly in a tight loop (the same lock-isolated copy
        # benchmark_status uses) — avoids per-iteration event-loop setup cost
        # while still exercising the concurrent copy-vs-reassign path.
        for _ in range(20000):
            try:
                snap = bench_state.snapshot()
                # Force full materialization (what JSON serialization would do).
                _ = list(snap.get("results", []))
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)
                break

    w = threading.Thread(target=writer)
    r = threading.Thread(target=reader)
    w.start()
    r.start()
    r.join()
    stop.set()
    w.join()

    assert errors == [], f"status read raised under concurrent append: {errors}"
    _reset_bench_state()


# ── A: failed models surfaced in CSV; decode never an em-dash; new columns ──
def test_csv_includes_failed_models_with_status():
    from server import benchmark_routes
    bench_state = _reset_bench_state()
    bench_state["results"] = [
        {"name": "ok.onnx", "codec": "Raw", "status": "ok", "precision": "fp16",
         "input_dtype": "float", "provider": "CUDAExecutionProvider",
         "fps": 60.0, "avg": 16.6, "decode_ms": None,
         "pre_ms": 5.0, "infer_ms": 10.0, "post_ms": 1.6,
         "min": 15.0, "max": 20.0, "std": 1.0, "p50": 16.0, "p95": 19.0, "p99": 20.0,
         "cpu_pct": 30.0, "ram_mb": 900, "gpu_pct": 70, "gpu_pct_peak": 88, "gpu_mem_used": 1234},
        {"name": "broken_fp16.onnx", "codec": "Raw", "status": "failed",
         "error": "[broken_fp16.onnx] load failed: Type Error ... float16"},
    ]
    resp = asyncio.run(benchmark_routes.benchmark_export_csv())
    text = resp.body.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    by_name = {r["model_name"]: r for r in rows}
    # failed model is NOT silently dropped — it carries status=failed + the reason
    assert "broken_fp16.onnx" in by_name
    assert by_name["broken_fp16.onnx"]["status"] == "failed"
    assert "float16" in by_name["broken_fp16.onnx"]["error"]
    # ok model carries the new precision/input_dtype/gpu columns
    ok = by_name["ok.onnx"]
    assert ok["status"] == "ok" and ok["precision"] == "fp16" and ok["input_dtype"] == "float"
    assert ok["gpu_pct"] == "70" and ok["gpu_mem_used_mb"] == "1234"
    # every row materializes the full header (no column shift)
    ncols = len(rows[0])
    assert all(len(r) == ncols for r in rows)
    _reset_bench_state()


def test_csv_decode_is_blank_never_emdash():
    from server import benchmark_routes
    bench_state = _reset_bench_state()
    bench_state["results"] = [
        {"name": "a.onnx", "codec": "Raw", "decode_ms": None, "fps": 1, "avg": 1},
        {"name": "b.onnx", "codec": "Raw", "decode_ms": "—", "fps": 1, "avg": 1},  # legacy glyph
        {"name": "c.onnx", "codec": "JPEG", "decode_ms": 3.2, "fps": 1, "avg": 1},
    ]
    resp = asyncio.run(benchmark_routes.benchmark_export_csv())
    text = resp.body.decode("utf-8-sig")
    assert "—" not in text  # the em-dash glyph must never reach the CSV
    rows = {r["model_name"]: r for r in csv.DictReader(io.StringIO(text))}
    assert rows["a.onnx"]["decode_ms"] == ""      # not measured -> blank
    assert rows["b.onnx"]["decode_ms"] == ""      # legacy '—' coerced to blank
    assert rows["c.onnx"]["decode_ms"] == "3.2"   # measured -> numeric
    _reset_bench_state()


# ── B: _build_providers honors the requested EP (TensorRT) ──
def test_build_providers_honors_ep_key_tensorrt():
    import core.model_loader as ml
    orig = ml.ort.get_available_providers
    ml.ort.get_available_providers = lambda: [
        "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        provs = _build_providers(ep_key="tensorrt")
        flat = [p[0] if isinstance(p, tuple) else p for p in provs]
        # TensorRT first, then CUDA + CPU fallback (unsupported nodes degrade gracefully).
        assert flat[0] == "TensorrtExecutionProvider"
        assert "CUDAExecutionProvider" in flat and "CPUExecutionProvider" in flat
        assert _build_providers(ep_key="cpu") == ["CPUExecutionProvider"]
        # an EP not available in this process -> graceful auto fallback (CUDA first here)
        auto = _build_providers(ep_key="directml")
        assert (auto[0][0] if isinstance(auto[0], tuple) else auto[0]) == "CUDAExecutionProvider"
        # auto stays identical to the no-arg call
        assert _build_providers(ep_key="auto") == _build_providers()
    finally:
        ml.ort.get_available_providers = orig


def test_build_providers_cpu_only_env_graceful():
    import core.model_loader as ml
    orig = ml.ort.get_available_providers
    ml.ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    try:
        # requesting tensorrt on a CPU-only build falls back to CPU, never crashes
        assert _build_providers(ep_key="tensorrt") == ["CPUExecutionProvider"]
        assert _build_providers(ep_key="auto") == ["CPUExecutionProvider"]
    finally:
        ml.ort.get_available_providers = orig


# ── precision detection from the ONNX graph (not the fp32 I/O dtype) ──
def test_detect_precision():
    import os
    import tempfile
    import numpy as np
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    from core.benchmark_runner import _detect_precision
    d = tempfile.mkdtemp(prefix="prec_")

    def _save(nodes, inits, inputs, outputs, name):
        g = helper.make_graph(nodes, name, inputs, outputs, inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        m.ir_version = 8
        p = os.path.join(d, name + ".onnx")
        onnx.save(m, p)
        return p

    Xf = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Yf = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    # fp32: float32 weights only
    w32 = numpy_helper.from_array(np.ones((1, 4), np.float32), "w")
    assert _detect_precision(
        _save([helper.make_node("Add", ["X", "w"], ["Y"])], [w32], [Xf], [Yf], "fp32")) == "fp32"
    # fp16: a float16 initializer present
    Xh = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, 4])
    Yh = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, 4])
    w16 = numpy_helper.from_array(np.ones((1, 4), np.float16), "w16")
    assert _detect_precision(
        _save([helper.make_node("Add", ["X", "w16"], ["Y"])], [w16], [Xh], [Yh], "fp16")) == "fp16"
    # int8: a QuantizeLinear/DequantizeLinear (QDQ) pair — keeps float32 I/O
    sc = numpy_helper.from_array(np.array(0.1, np.float32), "sc")
    zp = numpy_helper.from_array(np.array(0, np.int8), "zp")
    qdq = [helper.make_node("QuantizeLinear", ["X", "sc", "zp"], ["q"]),
           helper.make_node("DequantizeLinear", ["q", "sc", "zp"], ["Y"])]
    assert _detect_precision(_save(qdq, [sc, zp], [Xf], [Yf], "int8")) == "int8"


# ── TensorRT runtime enablement + honest EP fallback reporting ──
def test_ensure_tensorrt_runtime_is_safe_and_cached():
    # Best-effort: returns a bool (True only if the tensorrt pkg is importable),
    # never raises, and is idempotent/cached so repeated calls are cheap no-ops.
    from core.ep_selector import ensure_tensorrt_runtime
    a = ensure_tensorrt_runtime()
    b = ensure_tensorrt_runtime()
    assert isinstance(a, bool) and a == b


def test_csv_surfaces_requested_ep_for_silent_fallback():
    # When TensorRT was requested but the session ran on CUDA (TRT libs missing →
    # ORT silently fell back), the CSV must show ep_requested != provider so the
    # user sees why int8/fp16 wasn't accelerated.
    from server import benchmark_routes
    bench_state = _reset_bench_state()
    bench_state["results"] = [{
        "name": "m.onnx", "codec": "Raw", "status": "ok", "precision": "int8",
        "input_dtype": "float", "provider": "CUDAExecutionProvider",
        "ep_requested": "tensorrt", "ep_fallback": True, "fps": 50, "avg": 20,
    }]
    resp = asyncio.run(benchmark_routes.benchmark_export_csv())
    row = list(csv.DictReader(io.StringIO(resp.body.decode("utf-8-sig"))))[0]
    assert row["ep_requested"] == "tensorrt"
    assert row["provider"] == "CUDAExecutionProvider"  # the actual (fallback) EP
    _reset_bench_state()
