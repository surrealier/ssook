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
