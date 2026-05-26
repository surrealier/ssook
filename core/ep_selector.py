"""
EP runtime selector.

Called at startup BEFORE onnxruntime is imported.
Detects GPU hardware, selects the matching onnxruntime from ep_runtimes/{key}/,
and injects it into sys.path so the correct variant is loaded.

Build prerequisite: python scripts/setup_ep.py
All EP variants are bundled into _internal/ep_runtimes/ by PyInstaller.
"""
import os
import platform
import subprocess
import sys
from pathlib import Path

# exe: _MEIPASS/ep_runtimes/  |  source: ep_venvs/
if getattr(sys, "frozen", False):
    _BASE = Path(sys._MEIPASS) / "ep_runtimes"
else:
    _PROJECT = Path(__file__).resolve().parent.parent
    _ep_runtimes = _PROJECT / "ep_runtimes"
    _ep_venvs = _PROJECT / "ep_venvs"
    _BASE = _ep_runtimes if _ep_runtimes.is_dir() else _ep_venvs

_IS_FROZEN = getattr(sys, "frozen", False)

# Platform priority: GPU first, CPU last resort
_PRIORITY = {
    "Windows": ["cuda", "directml", "openvino", "cpu"],
    "Linux":   ["cuda", "openvino", "cpu"],
    "Darwin":  ["coreml", "cpu"],
}

_EP_LABELS = {
    "cuda": "CUDA (NVIDIA GPU)",
    "directml": "DirectML (Windows GPU)",
    "openvino": "OpenVINO (Intel)",
    "coreml": "CoreML (Apple Silicon)",
    "cpu": "CPU",
}

_EP_PROVIDER_MAP = {
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cpu": "CPUExecutionProvider",
}

# --- result populated by select_and_activate + get_ep_status ---
ep_result: dict = {
    "selected": None,
    "provider": None,
    "available_eps": [],
    "bundled_eps": [],
    "skipped": [],
    "fallback": False,
    "fallback_reason": None,
    "fallback_fix": None,
}


# ── Hardware detection ──────────────────────────────────

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
    return os.path.exists("/dev/dri")


# ── EP path resolution ──────────────────────────────────

def _resolve_ep_path(key: str) -> "str | None":
    """Return the directory containing onnxruntime package for this EP."""
    if _IS_FROZEN:
        d = _BASE / key
        return str(d) if (d / "onnxruntime").is_dir() else None
    # source mode: ep_venvs/{key}/Lib/site-packages or lib/python3.x/site-packages
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


# ── Hardware requirement check per EP ───────────────────

_HW_CHECKS = {
    "cuda": (_has_nvidia_gpu,
             "NVIDIA GPU not detected (nvidia-smi failed)",
             "Install NVIDIA driver, or check GPU connection"),
    "openvino": (_has_intel_gpu,
                 "Intel GPU not detected",
                 "This EP requires Intel iGPU/dGPU"),
    # directml: any Windows GPU, no extra check
    # coreml: macOS = Apple Silicon, platform check is enough
    # cpu: always ok
}


# ── System onnxruntime probe ────────────────────────────

def _detect_system_provider(priority: list) -> str:
    """Probe system-installed onnxruntime for available GPU providers."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        for key in priority:
            prov = _EP_PROVIDER_MAP.get(key)
            if prov and prov in available:
                return prov
    except Exception:
        pass
    return "CPUExecutionProvider"


# ── Main entry point ────────────────────────────────────

def select_and_activate() -> str:
    """
    Detect GPU type, pick the matching onnxruntime from ep_runtimes,
    inject into sys.path. Must be called BEFORE import onnxruntime.
    """
    plat = platform.system()
    priority = _PRIORITY.get(plat, ["cpu"])

    bundled = [k for k in priority if _resolve_ep_path(k)]
    ep_result["bundled_eps"] = bundled

    selected = None
    for key in priority:
        path = _resolve_ep_path(key)
        if not path:
            if key != "cpu":  # don't warn about missing cpu if gpu is found
                ep_result["skipped"].append({
                    "ep": key,
                    "reason": f"{_EP_LABELS.get(key, key)} runtime not bundled",
                    "fix": f"Run: python scripts/setup_ep.py {key}  then rebuild",
                })
            continue

        # hardware check
        if key in _HW_CHECKS:
            check_fn, reason, fix = _HW_CHECKS[key]
            if not check_fn():
                ep_result["skipped"].append({"ep": key, "reason": reason, "fix": fix})
                continue

        selected = key
        break

    # CPU as last resort
    if selected is None:
        cpu_path = _resolve_ep_path("cpu")
        if cpu_path:
            selected = "cpu"

    if selected:
        ep_path = _resolve_ep_path(selected)
        if ep_path not in sys.path:
            sys.path.insert(0, ep_path)
        # Windows: add DLL search paths for CUDA/TensorRT/DirectML libs
        if sys.platform == "win32":
            for libs_name in ["onnxruntime.libs", "onnxruntime_gpu.libs", "onnxruntime_directml.libs"]:
                libs = Path(ep_path) / libs_name
                if not libs.is_dir():
                    libs = _BASE / selected / libs_name
                if libs.is_dir():
                    os.add_dll_directory(str(libs))
            # DirectML: also add the site-packages dir itself (DirectML.dll may live there)
            if selected == "directml":
                ort_pkg = Path(ep_path) / "onnxruntime" / "capi"
                if ort_pkg.is_dir():
                    os.add_dll_directory(str(ort_pkg))
        ep_result["selected"] = selected
        ep_result["provider"] = _EP_PROVIDER_MAP.get(selected, "CPUExecutionProvider")
        print(f"[EP Selector] selected: {selected} -> {ep_path}")
    else:
        # No bundled EP — probe system onnxruntime for GPU providers
        sys_provider = _detect_system_provider(priority)
        ep_result["selected"] = "system"
        ep_result["provider"] = sys_provider
        if sys_provider != "CPUExecutionProvider":
            # System has GPU support — no warnings needed
            ep_result["skipped"] = []
            print(f"[EP Selector] using system onnxruntime -> {sys_provider}")
        else:
            # System onnxruntime is CPU-only — keep skipped for diagnostics
            print("[EP Selector] using system onnxruntime (CPU only)")

    # Fallback detection: only flag if actually stuck on CPU
    if ep_result["provider"] == "CPUExecutionProvider" and priority and priority[0] != "cpu":
        ep_result["fallback"] = True
        if ep_result["skipped"]:
            first = ep_result["skipped"][0]
            ep_result["fallback_reason"] = first["reason"]
            ep_result["fallback_fix"] = first["fix"]
        else:
            ep_result["fallback_reason"] = "No GPU-capable onnxruntime found"
            ep_result["fallback_fix"] = "pip install onnxruntime-gpu  (requires CUDA 12 + cuDNN 9)"

    return ep_result["selected"]


def get_ep_status() -> dict:
    """Full EP status for API/UI. Call after onnxruntime is loaded."""
    try:
        import onnxruntime as ort
        ep_result["available_eps"] = ort.get_available_providers()
    except Exception:
        ep_result["available_eps"] = []
    return ep_result
