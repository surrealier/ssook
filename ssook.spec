# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for ssook (Web UI)
# Build: pyinstaller ssook.spec
#
# ep_venvs/에 설치된 모든 EP의 onnxruntime을 ep_runtimes/{key}/에 번들.
# 실행 시 core/ep_selector.py가 환경을 감지하여 적절한 폴더를 sys.path에 삽입.
# 사전 준비: python scripts/setup_ep.py

import glob, os, sys

block_cipher = None

# ── ep_venvs에서 모든 EP의 onnxruntime 패키지를 수집 ──
_ep_datas = []
_spec_dir = os.path.dirname(os.path.abspath(SPECPATH)) if 'SPECPATH' in dir() else os.getcwd()
_ep_venvs = os.path.join(_spec_dir, 'ep_venvs')

def _find_site_packages(venv_dir):
    if sys.platform == 'win32':
        sp = os.path.join(venv_dir, 'Lib', 'site-packages')
        return sp if os.path.isdir(sp) else None
    lib = os.path.join(venv_dir, 'lib')
    if not os.path.isdir(lib):
        return None
    for d in sorted(os.listdir(lib), reverse=True):
        if d.startswith('python'):
            sp = os.path.join(lib, d, 'site-packages')
            if os.path.isdir(sp):
                return sp
    return None

_ep_keys = ['cuda', 'directml', 'openvino', 'coreml', 'cpu']
_found = []
for _ep in _ep_keys:
    _venv = os.path.join(_ep_venvs, _ep)
    _sp = _find_site_packages(_venv)
    if _sp is None:
        continue
    _ort_dir = os.path.join(_sp, 'onnxruntime')
    if not os.path.isdir(_ort_dir):
        continue
    # onnxruntime 전체 폴더를 ep_runtimes/{key}/onnxruntime 으로 번들
    _ep_datas.append((_ort_dir, os.path.join('ep_runtimes', _ep, 'onnxruntime')))
    # onnxruntime 관련 .libs / .dylibs 폴더 (CUDA DLL 등)
    for _extra in ['onnxruntime.libs', 'onnxruntime_gpu.libs', 'nvidia']:
        _extra_dir = os.path.join(_sp, _extra)
        if os.path.isdir(_extra_dir):
            _ep_datas.append((_extra_dir, os.path.join('ep_runtimes', _ep, _extra)))
    _found.append(_ep)
    print(f'[ssook.spec] EP 번들: {_ep} ← {_sp}')

if not _found:
    print('[ssook.spec] ⚠ ep_venvs에서 onnxruntime을 찾지 못함')
else:
    print(f'[ssook.spec] 총 {len(_found)}개 EP 번들: {_found}')

a = Analysis(
    ['run_web.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('web',         'web'),
        ('settings',    'settings'),
        ('assets',      'assets'),
        ('server.py',   '.'),
        ('core',        'core'),
    ] + _ep_datas,
    hiddenimports=[
        'uvicorn', 'uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto',
        'uvicorn.protocols', 'uvicorn.protocols.http', 'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets', 'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan', 'uvicorn.lifespan.on',
        'fastapi', 'starlette', 'pydantic', 'anyio',
        'cv2', 'numpy', 'yaml', 'psutil',
        'core.app_config', 'core.model_loader', 'core.inference',
        'core.benchmark_runner', 'core.evaluation', 'core.ep_selector',
        'server',
        'webview', 'webview.platforms', 'webview.platforms.edgechromium',
        'clr_loader', 'pythonnet',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        'onnxruntime',  # 메인 환경의 onnxruntime 제외 — ep_runtimes에서 로드
        'PySide6', 'PyQt5', 'PyQt6',
        'torch', 'torchvision', 'torchaudio',
        'IPython', 'jupyter', 'notebook', 'ultralytics',
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ssook',
    debug=False,
    strip=False,
    upx=True,
    console=False,
    icon='assets/icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=['onnxruntime_providers_*.dll', 'onnxruntime_providers_shared.dll',
                 'cv2*.dll', 'opencv*.dll', 'libopenblas*.dll'],
    name='ssook',
)
