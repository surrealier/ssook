"""Benchmark runner robustness: per-config callback indexing + graceful skip.

These guard the v1.6.0 fix where a skipped model (load failure / OOM) must NOT
shift codec labels of later results — callbacks now receive the config index,
and progress must still reach total even when models are skipped.
"""
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
