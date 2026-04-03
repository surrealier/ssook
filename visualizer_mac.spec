# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for YOLO Visualizer — macOS
# Build: pyinstaller visualizer_mac.spec

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
    upx=False,          # UPX not reliable on macOS
    console=False,
    target_arch=None,   # universal2 if needed: 'universal2'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='Visualizer',
)

app = BUNDLE(
    coll,
    name='Visualizer.app',
    icon='assets/icon.icns',
    bundle_identifier='com.visualizer.app',
    info_plist={
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleName': 'Visualizer',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,  # support dark mode
    },
)
