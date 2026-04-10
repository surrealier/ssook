"""
EP 런타임 선택기.

exe 실행 초기에 호출되어 환경(GPU)을 감지하고,
ep_runtimes/{key}/ 를 sys.path 선두에 삽입하여
적절한 onnxruntime 변종이 import되도록 한다.

반드시 onnxruntime을 import하기 전에 호출해야 한다.
"""
import os
import platform
import subprocess
import sys
from pathlib import Path

# exe: _MEIPASS/ep_runtimes/  |  소스: ep_venvs의 site-packages
if getattr(sys, "frozen", False):
    _BASE = Path(sys._MEIPASS) / "ep_runtimes"
    _MODE = "frozen"
else:
    _PROJECT = Path(__file__).resolve().parent.parent
    _ep_runtimes = _PROJECT / "ep_runtimes"
    _ep_venvs = _PROJECT / "ep_venvs"
    _BASE = _ep_runtimes if _ep_runtimes.is_dir() else _ep_venvs
    _MODE = "source"

# 플랫폼별 우선순위
_PRIORITY = {
    "Windows": ["cuda", "directml", "openvino", "cpu"],
    "Linux":   ["cuda", "openvino", "cpu"],
    "Darwin":  ["coreml", "cpu"],
}


def _has_nvidia_gpu() -> bool:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, timeout=5, text=True,
            creationflags=0x08000000 if sys.platform == "win32" else 0,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def _has_intel_gpu() -> bool:
    """Intel iGPU 존재 여부 (간이 판별)."""
    if sys.platform == "win32":
        try:
            r = subprocess.run(
                ["wmic", "path", "win32_videocontroller", "get", "name"],
                capture_output=True, timeout=5, text=True,
                creationflags=0x08000000,
            )
            return "intel" in r.stdout.lower()
        except Exception:
            return False
    return os.path.exists("/dev/dri")  # Linux Intel GPU


def _ep_available(key: str) -> bool:
    """번들된 ep_runtimes/{key}/onnxruntime 존재 여부."""
    return (_BASE / key / "onnxruntime").is_dir()


def _resolve_ep_path(key: str) -> "str | None":
    """EP의 onnxruntime이 있는 디렉토리 경로를 반환."""
    # frozen: ep_runtimes/{key}/  (onnxruntime이 바로 아래)
    if _MODE == "frozen":
        d = _BASE / key
        return str(d) if (d / "onnxruntime").is_dir() else None
    # source: ep_venvs/{key}/lib/python3.x/site-packages/
    venv = _BASE / key
    if sys.platform == "win32":
        sp = venv / "Lib" / "site-packages"
    else:
        lib = venv / "lib"
        sp = None
        if lib.is_dir():
            for d in sorted(lib.iterdir(), reverse=True):
                if d.name.startswith("python"):
                    sp = d / "site-packages"
                    break
    if sp and (sp / "onnxruntime").is_dir():
        return str(sp)
    return None


def select_and_activate() -> str:
    """
    최적 EP를 선택하고 sys.path에 삽입.
    반환: 선택된 ep_key.
    """
    plat = platform.system()
    priority = _PRIORITY.get(plat, ["cpu"])

    selected = None
    for key in priority:
        if not _resolve_ep_path(key):
            continue
        # 하드웨어 체크
        if key == "cuda" and not _has_nvidia_gpu():
            continue
        if key == "directml":
            # DirectML은 Windows GPU 범용 — NVIDIA 없을 때 유용
            pass
        if key == "openvino" and not _has_intel_gpu():
            continue
        selected = key
        break

    if selected is None:
        if _resolve_ep_path("cpu"):
            selected = "cpu"

    if selected:
        ep_path = _resolve_ep_path(selected)
        if ep_path not in sys.path:
            sys.path.insert(0, ep_path)
        # .libs 폴더도 DLL 검색 경로에 추가 (Windows CUDA DLL 등)
        if sys.platform == "win32":
            libs = Path(ep_path) / "onnxruntime.libs"
            if not libs.is_dir():
                libs = _BASE / selected / "onnxruntime.libs"
            if libs.is_dir():
                os.add_dll_directory(str(libs))
        print(f"[EP Selector] selected: {selected} -> {ep_path}")
    else:
        print("[EP Selector] WARNING: no EP available, using system onnxruntime")

    return selected or "auto"
