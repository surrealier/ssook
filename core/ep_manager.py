"""
EP(Execution Provider) venv 기반 관리.

onnxruntime 각 변종은 동일한 Python 네임스페이스를 공유하므로 동시 설치 불가.
→ ep_venvs/ 하위에 변종별 venv를 생성하여 완전 격리,
  실행 시 해당 venv의 Python 인터프리터로 서브프로세스를 실행.
"""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
EP_VENVS_DIR = _PROJECT_ROOT / "ep_venvs"
WORKER_SCRIPT = Path(__file__).parent / "ep_worker.py"
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

EP_VARIANTS: dict = {
    "cpu": {
        "label": "CPU  (onnxruntime)",
        "pkg": "onnxruntime",
        "desc": "CPU only - OpenMP multithread",
        "provider": "CPUExecutionProvider",
        "platforms": ["win32", "linux", "darwin"],
    },
    "cuda": {
        "label": "CUDA / TensorRT  (onnxruntime-gpu)",
        "pkg": "onnxruntime-gpu",
        "desc": "NVIDIA GPU - CUDAExecutionProvider",
        "provider": "CUDAExecutionProvider",
        "platforms": ["win32", "linux"],
    },
    "directml": {
        "label": "DirectML  (onnxruntime-directml)",
        "pkg": "onnxruntime-directml",
        "desc": "Windows GPU generic - DmlExecutionProvider (AMD/Intel/NVIDIA)",
        "provider": "DmlExecutionProvider",
        "platforms": ["win32"],
    },
    "openvino": {
        "label": "OpenVINO  (onnxruntime-openvino)",
        "pkg": "onnxruntime-openvino",
        "desc": "Intel iGPU first, fallback to OpenVINO CPU",
        "provider": "OpenVINOExecutionProvider",
        "platforms": ["win32", "linux"],
    },
    "coreml": {
        "label": "CoreML  (onnxruntime + coremltools)",
        "pkg": "onnxruntime",
        "extra_pkgs": ["coremltools"],
        "desc": "Apple Silicon / macOS - CoreMLExecutionProvider",
        "provider": "CoreMLExecutionProvider",
        "platforms": ["darwin"],
    },
}

# 플랫폼별 자동 선택 우선순위 (위에서부터 시도)
_AUTO_PRIORITY = {
    "win32": ["cuda", "directml", "openvino", "cpu"],
    "linux": ["cuda", "openvino", "cpu"],
    "darwin": ["coreml", "cpu"],
}


def get_ep_dir(ep_key: str) -> Path:
    return EP_VENVS_DIR / ep_key


def _get_venv_python(ep_key: str) -> Path:
    """venv 내 Python 인터프리터 경로 반환."""
    base = get_ep_dir(ep_key)
    if sys.platform == "win32":
        return base / "Scripts" / "python.exe"
    return base / "bin" / "python"


def is_ep_available(ep_key: str) -> bool:
    """venv의 Python 실행 파일이 존재하면 설치된 것으로 간주."""
    return _get_venv_python(ep_key).is_file()


def get_available_eps() -> "dict[str, bool]":
    """현재 플랫폼에 해당하는 EP만 반환."""
    plat = sys.platform
    return {
        k: is_ep_available(k)
        for k, v in EP_VARIANTS.items()
        if plat in v["platforms"]
    }


def auto_select_ep() -> str:
    """
    현재 플랫폼/하드웨어에 맞는 최적 EP를 자동 선택.
    설치된 venv 중 우선순위가 가장 높은 것을 반환.
    아무것도 없으면 'auto' (메인 프로세스의 onnxruntime 사용).
    """
    plat = sys.platform
    priority = _AUTO_PRIORITY.get(plat, ["cpu"])

    # CUDA 선택 시 NVIDIA GPU 존재 여부 확인
    for ep_key in priority:
        if not is_ep_available(ep_key):
            continue
        if ep_key == "cuda" and not _has_nvidia_gpu():
            continue
        return ep_key

    return "auto"


def _has_nvidia_gpu() -> bool:
    """nvidia-smi 실행 가능 여부로 NVIDIA GPU 존재 판단."""
    try:
        subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5,
            creationflags=_CREATE_NO_WINDOW,
        )
        return True
    except Exception:
        return False


def get_platform_variants() -> "dict[str, dict]":
    """현재 플랫폼에서 설치 가능한 EP 변종 목록."""
    plat = sys.platform
    return {k: v for k, v in EP_VARIANTS.items() if plat in v["platforms"]}


def launch_worker(task: dict) -> subprocess.Popen:
    """
    ep_worker.py를 서브프로세스로 실행.
    ep_key에 해당하는 venv의 Python을 사용.
    """
    ep_key = task.get("ep_key", "")
    python_exe = str(_get_venv_python(ep_key)) if ep_key and is_ep_available(ep_key) else sys.executable

    task["proj_root"] = str(_PROJECT_ROOT)

    proc = subprocess.Popen(
        [python_exe, str(WORKER_SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        creationflags=_CREATE_NO_WINDOW,
    )
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(task, ensure_ascii=False))
    proc.stdin.close()
    return proc


def run_export_in_venv(
    ep_key: str,
    config: dict,
    on_progress=None,
) -> dict:
    """Run a deployment export inside the ``ep_key`` venv and return its result.

    Drives the ep_worker "export" task to completion: streams ``progress``
    events to ``on_progress`` (done, total, msg) and returns the single
    ``result`` payload (the dict from ``core.deployment.runner.run_export``).

    This is a blocking, single-shot helper — exports are not cancellable, so we
    simply read until the worker emits ``finished``. Raises RuntimeError if the
    worker dies without ever producing a result, so the caller can fall back to
    an in-process attempt rather than silently reporting success.
    """
    task = {"task": "export", "ep_key": ep_key, "config": config}
    proc = launch_worker(task)

    result: "dict | None" = None
    error_msg: "str | None" = None

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except Exception:
            # Worker prints only JSON event lines; ignore stray output.
            continue
        kind = event.get("type")
        if kind == "progress":
            if on_progress:
                on_progress(event.get("done", 0), event.get("total", 1), event.get("msg", ""))
        elif kind == "result":
            result = event.get("data")
        elif kind == "error":
            error_msg = event.get("msg")
        elif kind == "finished":
            break

    proc.wait()

    if result is not None:
        return result
    if error_msg is not None:
        # Surface as a result-shaped dict so callers treat it like any export error.
        return {"error": error_msg}
    raise RuntimeError(
        f"export worker for ep '{ep_key}' exited without a result "
        f"(rc={proc.returncode})"
    )
