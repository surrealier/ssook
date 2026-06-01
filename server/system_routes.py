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
    # 메모리 자동 정리: RSS가 시스템 RAM의 50% 초과 시 캐시 정리.
    # cleanup 내부 버그가 HW 폴링 엔드포인트를 절대 500 못 시키도록 방어.
    try:
        total_ram = psutil.virtual_memory().total
        if proc.memory_info().rss > total_ram * 0.5:
            _auto_cleanup_memory()
    except Exception:
        pass
    return info


_MEM_CLEANUP_INTERVAL = 0  # 마지막 정리 시각

def _auto_cleanup_memory():
    """메모리 압박 시 자동 캐시 정리.

    참조 심볼은 모두 다른 모듈 소유라 함수 내부에서 지연 import한다
    (모듈 로드 시 순환 import 회피 + 가장 필요한 시점에만 비용 지불).
    """
    global _MEM_CLEANUP_INTERVAL
    now = time.time()
    if now - _MEM_CLEANUP_INTERVAL < 30:  # 30초 내 중복 정리 방지
        return
    _MEM_CLEANUP_INTERVAL = now

    from server import utils
    from server.state import compare_state, embedding_state
    from server.viewer_routes import _cleanup_stale_sessions

    # stale 비디오 세션 정리
    _cleanup_stale_sessions()
    # compare 결과 정리 (실행 중이 아닌 경우) — TaskState 락 경유 read/write
    if not compare_state.snapshot().get("running"):
        compare_state["results"] = []
    # embedding 이미지 정리
    if not embedding_state.snapshot().get("running"):
        embedding_state["image"] = None
    # 글로벌 팔레트 캐시 축소 (utils 소유 헬퍼; 미배포 시 안전하게 skip)
    trim = getattr(utils, "trim_palette_cache", None)
    if trim is not None:
        trim(20)
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
def _parse_filters(s):
    """Parse filter string. Accepts '*.onnx;*.pt' or Qt-style 'Models (*.onnx *.pt)'."""
    if not s:
        return [("All files", "*.*")]
    pairs = []
    for part in s.split(";;"):
        part = part.strip()
        if "(" in part and ")" in part:
            label = part[:part.index("(")].strip()
            exts = part[part.index("(") + 1:part.index(")")].strip()
            pairs.append((label, exts))
        else:
            pairs.append(("Files", part))
    return pairs or [("All files", "*.*")]


class FileSelectRequest(BaseModel):
    filters: Optional[str] = None


def _ps_single_quote(value: str) -> str:
    """Escape a string for a PowerShell single-quoted literal.

    The dialog script interpolates user-controlled values (title, filter
    string from req.filters) into a PS source string passed to
    `powershell -Command`. Inside a single-quoted PS literal the ONLY
    metacharacter is `'`, which is escaped by doubling it. Without this a
    crafted filter such as `'; <cmd>; '` would break out of the literal
    and execute arbitrary commands.
    """
    return str(value).replace("'", "''")


def _applescript_double_quote(value: str) -> str:
    """Escape a string for an AppleScript double-quoted literal.

    Backslash first (so we don't double-escape), then the double quote.
    """
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _native_file_dialog(title="Select File", filters=None, multiple=False, directory=False):
    """OS 네이티브 파일 다이얼로그. Windows/macOS/Linux 지원."""
    import subprocess, sys
    system = platform.system()

    if system == "Windows":
        title_esc = _ps_single_quote(title)
        if directory:
            script = (
                "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null;"
                "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
                f"$d.Description = '{title_esc}';"
                "$d.ShowNewFolderButton = $true;"
                "if ($d.ShowDialog() -eq 'OK') { $d.SelectedPath } else { '' }"
            )
        else:
            filter_str = "All files (*.*)|*.*"
            if filters:
                parts = []
                for f in _parse_filters(filters):
                    parts.append(f"{f[0]}|{f[1].replace(' ', ';')}")
                filter_str = "|".join(parts) + "|All files (*.*)|*.*"
            filter_esc = _ps_single_quote(filter_str)
            script = (
                "Add-Type -AssemblyName System.Windows.Forms;"
                "$d = New-Object System.Windows.Forms.OpenFileDialog;"
                f"$d.Title = '{title_esc}';"
                f"$d.Filter = '{filter_esc}';"
                f"$d.Multiselect = {'$true' if multiple else '$false'};"
                "if ($d.ShowDialog() -eq 'OK') {"
                + ("  $d.FileNames -join '|'" if multiple else "  $d.FileName") +
                "} else { '' }"
            )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=120
        )
        path = result.stdout.strip()
        if multiple:
            return [p for p in path.split("|") if p]
        return path

    elif system == "Darwin":
        title_esc = _applescript_double_quote(title)
        if directory:
            script = 'tell application "System Events" to activate\n'
            script += f'set f to choose folder with prompt "{title_esc}"\nreturn POSIX path of f'
        elif multiple:
            script = 'tell application "System Events" to activate\n'
            script += f'set f to choose file with prompt "{title_esc}" with multiple selections allowed\n'
            script += 'set out to ""\nrepeat with p in f\nset out to out & POSIX path of p & "\\n"\nend repeat\nreturn out'
        else:
            script = 'tell application "System Events" to activate\n'
            script += f'set f to choose file with prompt "{title_esc}"\nreturn POSIX path of f'
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=120
        )
        path = result.stdout.strip()
        if multiple:
            return [p for p in path.split("\n") if p]
        return path

    else:  # Linux
        cmd = ["zenity"]
        if directory:
            cmd += ["--file-selection", "--directory", f"--title={title}"]
        else:
            cmd += ["--file-selection", f"--title={title}"]
            if multiple:
                cmd += ["--multiple", "--separator=|"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            path = result.stdout.strip()
            if multiple:
                return [p for p in path.split("|") if p]
            return path
        except FileNotFoundError:
            # Fallback to tkinter
            return _tkinter_file_dialog(title, filters, multiple, directory)


def _tkinter_file_dialog(title, filters, multiple, directory):
    """Fallback: tkinter dialog."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    if directory:
        path = filedialog.askdirectory(title=title)
        root.destroy()
        return path or ""
    ft = [("All files", "*.*")] if not filters else _parse_filters(filters)
    if multiple:
        paths = filedialog.askopenfilenames(title=title, filetypes=ft)
        root.destroy()
        return list(paths) if paths else []
    path = filedialog.askopenfilename(title=title, filetypes=ft)
    root.destroy()
    return path or ""


@router.post("/api/fs/select")
async def select_file(req: FileSelectRequest):
    """Native OS file selection dialog."""
    try:
        path = _native_file_dialog(title="Select File", filters=req.filters)
        return {"path": path or ""}
    except Exception as e:
        return {"error": str(e), "path": ""}


@router.post("/api/fs/select-multi")
async def select_files(req: FileSelectRequest):
    """Native OS multiple file selection dialog."""
    try:
        paths = _native_file_dialog(title="Select Files", filters=req.filters, multiple=True)
        return {"paths": paths if paths else []}
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




# ── Missing FS endpoints ────────────────────────────────
class FsListRequest(BaseModel):
    path: str = ""
    extensions: Optional[list] = None
    recursive: bool = False


@router.post("/api/fs/list")
async def fs_list(req: FsListRequest):
    """List files in a directory, optionally filtered by extensions."""
    from pathlib import Path
    p = Path(req.path) if req.path else Path(".")
    if not p.exists():
        return {"files": [], "error": f"Path not found: {req.path}"}
    files = []
    try:
        items = p.rglob("*") if req.recursive else p.iterdir()
        for f in items:
            if f.is_file():
                if req.extensions:
                    if f.suffix.lower() in [e.lower() if e.startswith('.') else f'.{e.lower()}' for e in req.extensions]:
                        files.append(str(f))
                else:
                    files.append(str(f))
    except PermissionError:
        pass
    files.sort()
    return {"files": files, "count": len(files)}


@router.post("/api/fs/select-dir")
async def select_dir():
    """Native OS directory selection dialog."""
    try:
        path = _native_file_dialog(title="Select Directory", directory=True)
        return {"path": path or ""}
    except Exception as e:
        return {"error": str(e), "path": ""}


class FsBrowseRequest(BaseModel):
    path: str = ""
    mode: str = "all"  # "all" | "dir" | "file"


@router.post("/api/fs/browse")
async def fs_browse(req: FsBrowseRequest):
    """Browse directory contents for the web-based file browser."""
    from pathlib import Path
    p = Path(req.path) if req.path else Path.home()
    if not p.exists():
        p = Path.home()
    if p.is_file():
        p = p.parent

    entries = []
    try:
        for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if item.name.startswith('.'):
                continue
            entries.append({
                "name": item.name,
                "path": str(item),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else 0,
            })
    except PermissionError:
        pass

    # Filter by mode
    if req.mode == "dir":
        entries = [e for e in entries if e["is_dir"]]

    return {
        "current": str(p),
        "parent": str(p.parent) if p.parent != p else "",
        "entries": entries,
    }


class ListFilesRequest(BaseModel):
    dir: str = ""
    extensions: Optional[list] = None


@router.post("/api/list-files")
async def list_files(req: ListFilesRequest):
    """List image/label files in a directory (used by pose/instance-seg tabs)."""
    from pathlib import Path
    p = Path(req.dir) if req.dir else Path(".")
    if not p.exists():
        return {"files": [], "error": f"Path not found: {req.dir}"}
    exts = req.extensions or [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    files = sorted([
        str(f) for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    ])
    return {"files": files, "count": len(files)}
