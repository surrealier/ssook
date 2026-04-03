"""
EP(Execution Provider) 패키지 관리.
onnxruntime 각 변종은 동일한 Python 네임스페이스를 공유하므로 동시 설치 불가.
→ ep_packages/ 하위 디렉토리에 --target 설치하여 격리,
  실행 시 서브프로세스의 sys.path를 조작하여 원하는 변종을 로드.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
EP_PACKAGES_DIR = _PROJECT_ROOT / "ep_packages"
WORKER_SCRIPT = Path(__file__).parent / "ep_worker.py"
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

EP_VARIANTS: dict = {
    "cpu": {
        "label": "CPU  (onnxruntime)",
        "pkg":   "onnxruntime",
        "desc":  "기본 CPU 실행 — OpenMP 멀티스레드",
    },
    "cuda": {
        "label": "CUDA / TensorRT  (onnxruntime-gpu)",
        "pkg":   "onnxruntime-gpu",
        "desc":  "NVIDIA GPU — CUDAExecutionProvider 포함",
    },
    "openvino": {
        "label": "OpenVINO  (onnxruntime-openvino)",
        "pkg":   "onnxruntime-openvino",
        "desc":  "Intel CPU / iGPU — OpenVINOExecutionProvider",
    },
    "directml": {
        "label": "DirectML  (onnxruntime-directml)",
        "pkg":   "onnxruntime-directml",
        "desc":  "Windows GPU 범용 — DmlExecutionProvider (AMD / Intel / NVIDIA)",
    },
}


def get_ep_dir(ep_key: str) -> Path:
    return EP_PACKAGES_DIR / ep_key


def is_ep_available(ep_key: str) -> bool:
    """ep_packages/{key}/onnxruntime 패키지 폴더가 있으면 설치된 것으로 간주."""
    return (get_ep_dir(ep_key) / "onnxruntime").is_dir()


def get_available_eps() -> "dict[str, bool]":
    return {k: is_ep_available(k) for k in EP_VARIANTS}


def launch_worker(task: dict) -> subprocess.Popen:
    """
    ep_worker.py를 서브프로세스로 실행.
    stdin: task JSON (단일 블록)
    stdout: 줄 단위 JSON 이벤트 스트림
    """
    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, str(WORKER_SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
        text=True,
        bufsize=1,
        creationflags=_CREATE_NO_WINDOW,
    )
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(task, ensure_ascii=False))
    proc.stdin.close()
    return proc
