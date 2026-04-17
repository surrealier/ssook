"""server 패키지 공통 유틸리티."""
import base64
import glob as glob_module
import os
import sys
import time

import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread replacement that handles unicode/Korean paths on Windows."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, flags)
    except Exception:
        return None


def generate_palette(n):
    """HSV 균등 분포로 n개의 BGR 색상 생성."""
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        hsv = np.uint8([[[hue, 220, 220]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(x) for x in bgr))
    return colors


_palette_cache: list = []


def get_color(style, cid, total):
    """Return BGR color for a class: style.color > palette > green fallback."""
    global _palette_cache
    if style.color:
        return tuple(style.color)
    if total > 0:
        if len(_palette_cache) < total:
            _palette_cache = generate_palette(total)
        return _palette_cache[cid % len(_palette_cache)]
    return (0, 255, 0)


def draw_label(frame, text, x1, y1, color, font_scale, font_thick, show_bg):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
    ty = max(y1 - 4, th + 4)
    if show_bg:
        cv2.rectangle(frame, (x1, ty - th - baseline - 2), (x1 + tw + 2, ty + 2), color, -1)
        lum = color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299
        txt_color = (0, 0, 0) if lum > 128 else (255, 255, 255)
    else:
        txt_color = color
    cv2.putText(frame, text, (x1 + 1, ty - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, font_thick, cv2.LINE_AA)


def encode_jpeg(img, quality=80):
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def glob_images(img_dir, recursive=False):
    files = []
    for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(glob_module.glob(os.path.join(img_dir, e)))
        if recursive:
            files.extend(glob_module.glob(os.path.join(img_dir, "**", e), recursive=True))
    files = list(dict.fromkeys(files))
    files.sort()
    return files


def draw_detections(frame, result, names):
    vis = frame.copy()
    total_cls = len(names)
    _dummy_style = type('S', (), {'color': None})()
    for box, score, cid in zip(result.boxes, result.scores, result.class_ids):
        cid_int = int(cid)
        color = get_color(_dummy_style, cid_int, total_cls)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(cid_int, str(cid_int))} {score:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 4, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


_SEG_PALETTE = [
    (128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128),
    (0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0),
    (192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128),
    (0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128),
]


def overlay_segmentation(frame, mask, alpha=0.5):
    h, w = frame.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = frame.copy()
    for cid in np.unique(mask_resized):
        if cid == 0:
            continue
        color = _SEG_PALETTE[cid % len(_SEG_PALETTE)]
        overlay[mask_resized == cid] = color
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


# ── GPU 유틸 ──
_gpu_available = None


def check_gpu_available():
    global _gpu_available
    if _gpu_available is None:
        try:
            import subprocess
            flags = 0x08000000 if sys.platform == "win32" else 0
            subprocess.check_output(["nvidia-smi", "--version"],
                                    text=True, timeout=2, creationflags=flags)
            _gpu_available = True
        except Exception:
            _gpu_available = False
    return _gpu_available


def get_gpu_info():
    if not check_gpu_available():
        return {"gpu_name": "N/A", "gpu_driver": "N/A", "gpu_memory_gb": 0, "pci_bus": "N/A"}
    try:
        import subprocess
        flags = 0x08000000 if sys.platform == "win32" else 0
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,pci.bus_id",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2, creationflags=flags,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        return {"gpu_name": parts[0], "gpu_driver": parts[1],
                "gpu_memory_gb": round(int(parts[2]) / 1024, 1), "pci_bus": parts[3]}
    except Exception:
        return {"gpu_name": "N/A", "gpu_driver": "N/A", "gpu_memory_gb": 0, "pci_bus": "N/A"}


# ── GPU HW stats 캐싱 ──
_gpu_hw_cache = None
_gpu_hw_cache_time = 0.0
_GPU_HW_CACHE_TTL = 5  # seconds


def get_gpu_hw_stats():
    """nvidia-smi GPU 사용률/메모리/온도 — 5초 캐싱."""
    global _gpu_hw_cache, _gpu_hw_cache_time
    fallback = {"gpu_name": "N/A", "gpu_util": 0, "gpu_mem_used": 0, "gpu_mem_total": 0, "gpu_temp": 0}
    if not check_gpu_available():
        return fallback
    now = time.time()
    if _gpu_hw_cache is not None and now - _gpu_hw_cache_time < _GPU_HW_CACHE_TTL:
        return _gpu_hw_cache
    try:
        import subprocess
        flags = 0x08000000 if sys.platform == "win32" else 0
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2, creationflags=flags,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        _gpu_hw_cache = dict(gpu_name=parts[0], gpu_util=int(parts[1]),
                             gpu_mem_used=int(parts[2]), gpu_mem_total=int(parts[3]),
                             gpu_temp=int(parts[4]))
        _gpu_hw_cache_time = now
        return _gpu_hw_cache
    except Exception:
        return fallback