# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for YOLO Visualizer
# Build: pyinstaller visualizer.spec

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('Videos',      'Videos'),
        ('Models',      'Models'),
        ('snapshots',   'snapshots'),
        ('settings',    'settings'),
        ('assets',      'assets'),
        ('ep_packages', 'ep_packages'),  # EP 격리 패키지
    ],
    hiddenimports=[
        'PySide6', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets',
        'onnxruntime', 'onnxruntime.capi', 'onnxruntime.capi.onnxruntime_pybind11_state',
        'cv2', 'numpy', 'yaml', 'ast',
        'core.app_config', 'core.model_loader', 'core.inference',
        'core.clip_inference',
        'core.benchmark_runner', 'core.bottleneck_analyzer',
        'core.ep_manager', 'core.ep_worker',
        'ui.main_window', 'ui.video_widget', 'ui.detect_thread',
        'ui.file_browser', 'ui.control_bar', 'ui.settings_tab', 'ui.class_filter',
        'ui.benchmark_tab', 'ui.analysis_tab', 'ui.evaluation_tab',
        'ui.dataset_explorer', 'ui.model_compare', 'ui.error_analyzer',
        'ui.embedding_viewer', 'ui.conf_optimizer', 'ui.dataset_splitter',
        'ui.clip_tab', 'ui.embedder_eval', 'ui.segmentation_tab',
        'ui.class_mapping_dialog', 'ui.stats_widget',
        'ui.image_quality_checker', 'ui.near_duplicate_detector',
        'ui.label_anomaly_detector', 'ui.format_converter',
        'ui.augmentation_preview', 'ui.class_remapper',
        'ui.similarity_search', 'ui.smart_sampler',
        'ui.dataset_merger', 'ui.leaky_split_detector',
        'ui.i18n', 'ui.batch_export',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torchvision', 'torchaudio',
        'torch.distributed', 'torch.testing', 'torch.ao',
        'matplotlib', 'pandas', 'scipy',
        'tkinter', 'PyQt5', 'PyQt6',
        'IPython', 'jupyter', 'notebook',
        'ultralytics',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Visualizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,              # GUI 앱 → 콘솔 창 없음
    icon='assets/icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    # onnxruntime provider DLL은 UPX 압축 제외 (로드 실패 방지)
    upx_exclude=['onnxruntime_providers_*.dll', 'onnxruntime_providers_shared.dll'],
    name='Visualizer',
)
