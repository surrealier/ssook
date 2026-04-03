"""
EP 워커 서브프로세스 스크립트.

stdin  : JSON 태스크 딕셔너리 (단일 블록)
stdout : 줄 단위 JSON 이벤트 스트림
  {"type": "progress", "done": N, "total": M, "msg": "..."}
  {"type": "result",   "data": {...}}   ← BenchmarkResult
  {"type": "report",   "data": {...}}   ← BottleneckReport
  {"type": "error",    "msg": "..."}
  {"type": "finished"}
"""
import json
import os
import sys


def _emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def _setup_paths(ep_dir: str, proj_root: str) -> None:
    if ep_dir and os.path.isdir(ep_dir):
        sys.path.insert(0, ep_dir)
    if proj_root:
        # 이미 경로에 없으면 추가
        if proj_root not in sys.path:
            sys.path.insert(1 if ep_dir else 0, proj_root)


def _run_benchmark(task: dict) -> None:
    import dataclasses
    from core.benchmark_runner import BenchmarkConfig, run_benchmark_core

    configs = []
    for c in task.get("configs", []):
        # tuple 필드 복원
        c["src_hw"] = tuple(c["src_hw"])
        configs.append(BenchmarkConfig(**c))

    stopped = [False]

    run_benchmark_core(
        configs,
        on_progress=lambda d, t, m: _emit({"type": "progress", "done": d, "total": t, "msg": m}),
        on_result=lambda r: _emit({"type": "result", "data": dataclasses.asdict(r)}),
        on_error=lambda m: _emit({"type": "error", "msg": m}),
        is_stopped=lambda: stopped[0],
    )


def _run_bottleneck(task: dict) -> None:
    import dataclasses
    from core.bottleneck_analyzer import run_bottleneck_core

    run_bottleneck_core(
        model_path=task["model_path"],
        model_type=task.get("model_type", "yolo"),
        batch_size=task.get("batch_size", 1),
        src_hw=tuple(task.get("src_hw", [1080, 1920])),
        on_progress=lambda d, t, m: _emit({"type": "progress", "done": d, "total": t, "msg": m}),
        on_report=lambda r: _emit({"type": "report", "data": dataclasses.asdict(r)}),
        on_error=lambda m: _emit({"type": "error", "msg": m}),
        is_stopped=lambda: False,
    )


def main() -> None:
    try:
        task = json.loads(sys.stdin.read())
    except Exception as exc:
        _emit({"type": "error", "msg": f"태스크 파싱 실패: {exc}"})
        return

    _setup_paths(task.get("ep_dir", ""), task.get("proj_root", ""))

    task_type = task.get("task")
    try:
        if task_type == "benchmark":
            _run_benchmark(task)
        elif task_type == "bottleneck":
            _run_bottleneck(task)
        else:
            _emit({"type": "error", "msg": f"알 수 없는 태스크 타입: {task_type!r}"})
    except Exception as exc:
        _emit({"type": "error", "msg": str(exc)})

    _emit({"type": "finished"})


if __name__ == "__main__":
    main()
