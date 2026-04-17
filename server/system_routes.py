"""/api/system/*, /api/fs/* 라우터."""
import gc
import os
import platform
import sys
import time
from typing import Optional

import psutil
from fastapi import APIRouter
from pydantic import BaseModel

from server.utils import check_gpu_available, get_gpu_info

router = APIRouter()

# ── Hardware Stats ──────────────────────────────────────
_gpu_available = None  # None=미확인, True/False=캐싱

def check_gpu_available():
    global _gpu_available
    if _gpu_available is None:
        try:
            import subprocess, sys as _sys
            flags = 0x08000000 if _sys.platform == "win32" else 0
            subprocess.check_output(["nvidia-smi", "--version"],
                                    text=True, timeout=2, creationflags=flags)
            _gpu_available = True
        except Exception:
            _gpu_available = False
    return _gpu_available

@router.get("/api/system/hw")
async def system_hw():
    import psutil
    proc = psutil.Process(os.getpid())
    info = {
        "cpu": round(proc.cpu_percent(interval=0), 1),
        "ram_mb": round(proc.memory_info().rss / 1024 / 1024),
    }
    if check_gpu_available():
        try:
            import subprocess, sys as _sys
            flags = 0x08000000 if _sys.platform == "win32" else 0
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=2, creationflags=flags,
            )
            parts = [p.strip() for p in out.strip().split(",")]
            info.update(gpu_name=parts[0], gpu_util=int(parts[1]),
                        gpu_mem_used=int(parts[2]), gpu_mem_total=int(parts[3]),
                        gpu_temp=int(parts[4]))
        except Exception:
            info.update(gpu_name="N/A", gpu_util=0, gpu_mem_used=0, gpu_mem_total=0, gpu_temp=0)
    else:
        info.update(gpu_name="N/A", gpu_util=0, gpu_mem_used=0, gpu_mem_total=0, gpu_temp=0)
    # 메모리 자동 정리: RSS가 시스템 RAM의 50% 초과 시 캐시 정리
    import psutil as _ps
    total_ram = _ps.virtual_memory().total
    if proc.memory_info().rss > total_ram * 0.5:
        _auto_cleanup_memory()
    return info


_MEM_CLEANUP_INTERVAL = 0  # 마지막 정리 시각

def _auto_cleanup_memory():
    """메모리 압박 시 자동 캐시 정리"""
    global _MEM_CLEANUP_INTERVAL
    now = time.time()
    if now - _MEM_CLEANUP_INTERVAL < 30:  # 30초 내 중복 정리 방지
        return
    _MEM_CLEANUP_INTERVAL = now
    # stale 비디오 세션 정리
    _cleanup_stale_sessions()
    # compare 결과 정리 (실행 중이 아닌 경우)
    if not _compare_state.get("running"):
        _compare_state["results"] = []
    # embedding 이미지 정리
    if not _embedding_state.get("running"):
        _embedding_state["image"] = None
    # 글로벌 팔레트 캐시 축소
    global _palette_cache
    _palette_cache = _palette_cache[:20] if len(_palette_cache) > 20 else _palette_cache
    import gc
    gc.collect()
    print("[Memory] Auto cleanup triggered")


# ── EP Status ───────────────────────────────────────────
@router.get("/api/system/ep")
async def system_ep():
    from core.ep_selector import get_ep_status
    return get_ep_status()

# ── System Info ─────────────────────────────────────────
@router.get("/api/system/info")
async def system_info():
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }
    try:
        import onnxruntime
        info["ort"] = onnxruntime.__version__
    except ImportError:
        info["ort"] = "N/A"
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda or "N/A"
    except ImportError:
        info["torch"] = "N/A"
        info["cuda"] = "N/A"
    if check_gpu_available():
        try:
            import subprocess, sys as _sys
            flags = 0x08000000 if _sys.platform == "win32" else 0
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                text=True, timeout=2, creationflags=flags,
            )
            info["gpu_name"] = out.strip()
        except Exception:
            info["gpu_name"] = "N/A"
    else:
        info["gpu_name"] = "N/A"
    return info


# ── File System API (for file/dir selection dialogs) ────
class FileSelectRequest(BaseModel):
    filters: Optional[str] = None


@router.post("/api/fs/select")
async def select_file(req: FileSelectRequest):
    """Return a file selection dialog via tkinter (fallback)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title="Select File",
            filetypes=[("All files", "*.*")] if not req.filters else _parse_filters(req.filters),
        )
        root.destroy()
        return {"path": path or ""}
    except Exception as e:
        return {"error": str(e), "path": ""}


@router.post("/api/fs/select-multi")
async def select_files(req: FileSelectRequest):
    """Return multiple file selection dialog via tkinter."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        paths = filedialog.askopenfilenames(
            title="Select Files",
            filetypes=[("All files", "*.*")] if not req.filters else _parse_filters(req.filters),
        )
        root.destroy()
        return {"paths": list(paths) if paths else []}
    except Exception as e:
        return {"error": str(e), "paths": []}


# ── 6. System Full Info ─────────────────────────────────
@router.get("/api/system/full-info")
async def system_full_info():
    import psutil
    gpu = get_gpu_info()
    ort_ver = "N/A"
    try:
        import onnxruntime
        ort_ver = onnxruntime.__version__
    except ImportError:
        pass
    return {
        "os": f"{platform.system()} {platform.release()} {platform.version()}",
        "cpu": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "python": platform.python_version(),
        "onnxruntime": ort_ver,
        **gpu,
    }


