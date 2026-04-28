# Execution Providers

[← Back to Index](index.md) | 🌐 [한국어](ko/execution-providers.md) | [日本語](ja/execution-providers.md) | [中文](zh/execution-providers.md)

ssook automatically selects the best hardware accelerator (Execution Provider) for your system. This document explains how the EP system works and how to configure it.

---

## Table of Contents

- [Overview](#overview)
- [Supported Execution Providers](#supported-execution-providers)
- [How Auto-Selection Works](#how-auto-selection-works)
- [venv Isolation Architecture](#venv-isolation-architecture)
- [Installation](#installation)
- [Troubleshooting](#troubleshooting)

---

## Overview

ONNX Runtime supports multiple hardware backends called **Execution Providers (EPs)**. Each EP targets different hardware:

- **CUDA** → NVIDIA GPUs
- **DirectML** → Any Windows GPU (NVIDIA, AMD, Intel)
- **OpenVINO** → Intel CPUs and iGPUs
- **CoreML** → Apple Silicon (M1/M2/M3)
- **CPU** → Any processor (fallback)

The challenge: each EP requires a **different build** of the `onnxruntime` package, and they conflict with each other. You can't install `onnxruntime-gpu` and `onnxruntime-directml` in the same Python environment.

ssook solves this with **venv isolation** — each EP gets its own isolated environment.

---

## Supported Execution Providers

| EP | Platform | Hardware | Package |
|----|----------|----------|---------|
| CUDA | Windows, Linux | NVIDIA GPU | `onnxruntime-gpu` |
| DirectML | Windows | Any GPU | `onnxruntime-directml` |
| OpenVINO | Windows, Linux | Intel CPU/iGPU | `onnxruntime-openvino` |
| CoreML | macOS | Apple Silicon | `onnxruntime-coreml` |
| CPU | All | Any CPU | `onnxruntime` |

---

## How Auto-Selection Works

At startup, **before** `onnxruntime` is imported, ssook runs the EP selector:

```
┌─────────────────────────────┐
│   Detect platform (OS)      │
│   Windows / Linux / macOS   │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  Get priority list          │
│  Win: cuda→directml→        │
│       openvino→cpu          │
│  Linux: cuda→openvino→cpu   │
│  macOS: coreml→cpu          │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  For each EP in priority:   │
│  1. Is runtime bundled?     │
│  2. Is hardware present?    │
│  → If both yes: SELECT      │
│  → If no: skip, record why  │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  Inject selected EP's       │
│  site-packages into         │
│  sys.path[0]                │
│  + add DLL directories      │
│  (Windows)                  │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  import onnxruntime         │
│  (loads the correct build)  │
└─────────────────────────────┘
```

**Hardware detection methods:**

| EP | Detection Method |
|----|-----------------|
| CUDA | Run `nvidia-smi --query-gpu=name` — if it returns a GPU name, NVIDIA GPU is present |
| OpenVINO | Windows: `wmic path win32_videocontroller get name` checks for "intel". Linux: checks `/dev/dri` exists |
| DirectML | Any Windows GPU (no extra check needed) |
| CoreML | macOS platform check (Apple Silicon) |
| CPU | Always available |

**Fallback behavior:**
If the highest-priority EP is unavailable (no hardware, runtime not installed), ssook falls back to the next EP in the priority list. The fallback reason is recorded and shown in the Settings tab, along with a suggested fix.

---

## venv Isolation Architecture

Each EP runtime lives in its own isolated directory:

```
ep_venvs/                          (source mode)
├── cuda/
│   └── Lib/site-packages/
│       └── onnxruntime/           (onnxruntime-gpu)
├── directml/
│   └── Lib/site-packages/
│       └── onnxruntime/           (onnxruntime-directml)
├── openvino/
│   └── Lib/site-packages/
│       └── onnxruntime/           (onnxruntime-openvino)
└── cpu/
    └── Lib/site-packages/
        └── onnxruntime/           (onnxruntime)

ep_runtimes/                       (frozen exe mode)
├── cuda/
│   └── onnxruntime/
├── directml/
│   └── onnxruntime/
└── cpu/
    └── onnxruntime/
```

When an EP is selected, its `site-packages` directory is inserted at the **front** of `sys.path`. This ensures that `import onnxruntime` loads the correct build for the selected EP.

On Windows, DLL search directories are also registered for CUDA/TensorRT/DirectML shared libraries.

---

## Installation

### Automatic (recommended)

```bash
# Install all EPs for your platform
python scripts/setup_ep.py

# Install specific EPs only
python scripts/setup_ep.py cuda cpu

# Check installation status
python scripts/setup_ep.py --status
```

The setup script:
1. Creates a virtual environment for each EP
2. Installs the correct `onnxruntime` variant
3. Verifies the installation

### Manual

If you only need CPU inference, just install the base package:

```bash
pip install onnxruntime
```

For GPU acceleration:

```bash
# NVIDIA GPU
pip install onnxruntime-gpu

# Windows GPU (any vendor)
pip install onnxruntime-directml
```

> Note: Manual installation doesn't use venv isolation. Only one EP variant can be installed at a time.

---

## Troubleshooting

### "Falling back to CPU"

**Cause**: The preferred EP's hardware was not detected.

| EP | Check | Fix |
|----|-------|-----|
| CUDA | `nvidia-smi` fails | Install NVIDIA driver, verify GPU connection |
| OpenVINO | No Intel GPU detected | This EP requires Intel iGPU/dGPU |
| DirectML | — | Should work on any Windows GPU |

### "EP runtime not bundled"

**Cause**: The EP's onnxruntime package is not installed in `ep_venvs/`.

**Fix**: Run `python scripts/setup_ep.py <ep_name>` to install it.

### Checking current EP status

In ssook, go to **Settings** tab. The EP status section shows:
- Currently selected EP
- Available EPs on the system
- Bundled EPs in ep_venvs
- Skipped EPs with reasons and suggested fixes
- Whether a fallback occurred
