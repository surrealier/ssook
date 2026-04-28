# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for ssook — macOS (Web UI)
# Build: pyinstaller ssook_mac.spec

import os, sys

block_cipher = None

# ── ep_venvs에서 모든 EP의 onnxruntime 패키지를 수집 ──
_ep_datas = []
_spec_dir = os.path.dirname(os.path.abspath(SPECPATH)) if 'SPECPATH' in dir() else os.getcwd()
_ep_venvs = os.path.join(_spec_dir, 'ep_venvs')

def _find_site_packages(venv_dir):
    lib = os.path.join(venv_dir, 'lib')
    if not os.path.isdir(lib):
        return None
    for d in sorted(os.listdir(lib), reverse=True):
        if d.startswith('python'):
            sp = os.path.join(lib, d, 'site-packages')
            if os.path.isdir(sp):
                return sp
    return None

for _ep in ['coreml', 'cpu']:
    _sp = _find_site_packages(os.path.join(_ep_venvs, _ep))
    if _sp is None:
        continue
    _ort_dir = os.path.join(_sp, 'onnxruntime')
    if not os.path.isdir(_ort_dir):
        continue
    _ep_datas.append((_ort_dir, os.path.join('ep_runtimes', _ep, 'onnxruntime')))
    print(f'[ssook_mac.spec] EP bundle: {_ep}')

a = Analysis(
    ['run_web.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('web',         'web'),
        ('settings',    'settings'),
        ('assets',      'assets'),
        ('server',      'server'),
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
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        'PySide6', 'PyQt5', 'PyQt6', 'tkinter',
        'torch', 'torchvision', 'torchaudio',
        'IPython', 'jupyter', 'notebook', 'ultralytics',
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True, name='ssook',
    debug=False, strip=False, upx=False, console=False, target_arch=None,
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False, name='ssook',
)

app = BUNDLE(
    coll, name='ssook.app', icon='assets/icon.icns',
    bundle_identifier='com.ssook.app',
    info_plist={
        'CFBundleShortVersionString': '1.4.0',
        'CFBundleName': 'ssook',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,
    },
)
