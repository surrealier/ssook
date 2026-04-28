# Execution Providers（执行提供程序）

[← 返回目录](../index.md) | 🌐 [English](../execution-providers.md) | [한국어](../ko/execution-providers.md) | [日本語](../ja/execution-providers.md) | **中文**

ssook 会自动为你的系统选择最佳的硬件加速器（Execution Provider）。本文档介绍 EP 系统的工作原理及配置方法。

---

## 目录

- [概述](#概述)
- [支持的 Execution Providers](#支持的-execution-providers)
- [自动选择机制](#自动选择机制)
- [venv 隔离架构](#venv-隔离架构)
- [安装](#安装)
- [故障排除](#故障排除)

---

## 概述

ONNX Runtime 支持多种硬件后端，称为 **Execution Providers（EP，执行提供程序）**。每种 EP 针对不同的硬件：

- **CUDA** → NVIDIA GPU
- **DirectML** → 任何 Windows GPU（NVIDIA、AMD、Intel）
- **OpenVINO** → Intel CPU 和 iGPU
- **CoreML** → Apple Silicon（M1/M2/M3）
- **CPU** → 任何处理器（兜底方案）

难点在于：每种 EP 需要**不同版本**的 `onnxruntime` 包，且它们之间互相冲突。你无法在同一个 Python 环境中同时安装 `onnxruntime-gpu` 和 `onnxruntime-directml`。

ssook 通过 **venv 隔离**解决了这个问题——每种 EP 拥有独立的隔离环境。

---

## 支持的 Execution Providers

| EP | 平台 | 硬件 | 包名 |
|----|------|------|------|
| CUDA | Windows, Linux | NVIDIA GPU | `onnxruntime-gpu` |
| DirectML | Windows | 任何 GPU | `onnxruntime-directml` |
| OpenVINO | Windows, Linux | Intel CPU/iGPU | `onnxruntime-openvino` |
| CoreML | macOS | Apple Silicon | `onnxruntime-coreml` |
| CPU | 全平台 | 任何 CPU | `onnxruntime` |

---

## 自动选择机制

启动时，在导入 `onnxruntime` **之前**，ssook 运行 EP 选择器：

```
┌─────────────────────────────┐
│   检测平台（操作系统）        │
│   Windows / Linux / macOS   │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  获取优先级列表              │
│  Win: cuda→directml→        │
│       openvino→cpu          │
│  Linux: cuda→openvino→cpu   │
│  macOS: coreml→cpu          │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  对优先级列表中的每个 EP：   │
│  1. 运行时是否已捆绑？       │
│  2. 硬件是否存在？           │
│  → 两者均是：选择该 EP       │
│  → 否则：跳过，记录原因      │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  将选中 EP 的                │
│  site-packages 注入          │
│  sys.path[0]                │
│  + 添加 DLL 目录             │
│  （Windows）                 │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  import onnxruntime         │
│  （加载正确的构建版本）       │
└─────────────────────────────┘
```

**硬件检测方法：**

| EP | 检测方法 |
|----|----------|
| CUDA | 运行 `nvidia-smi --query-gpu=name`——如果返回 GPU 名称，则存在 NVIDIA GPU |
| OpenVINO | Windows：`wmic path win32_videocontroller get name` 检查是否包含 "intel"。Linux：检查 `/dev/dri` 是否存在 |
| DirectML | 任何 Windows GPU（无需额外检查） |
| CoreML | macOS 平台检查（Apple Silicon） |
| CPU | 始终可用 |

**回退行为：**
如果最高优先级的 EP 不可用（无硬件、运行时未安装），ssook 会回退到优先级列表中的下一个 EP。回退原因会被记录并显示在 Settings 标签页中，同时提供修复建议。

---

## venv 隔离架构

每种 EP 运行时位于独立的隔离目录中：

```
ep_venvs/                          （源码模式）
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

ep_runtimes/                       （打包 exe 模式）
├── cuda/
│   └── onnxruntime/
├── directml/
│   └── onnxruntime/
└── cpu/
    └── onnxruntime/
```

当选中某个 EP 时，其 `site-packages` 目录被插入到 `sys.path` 的**最前面**。这确保 `import onnxruntime` 加载的是所选 EP 对应的正确构建版本。

在 Windows 上，还会为 CUDA/TensorRT/DirectML 共享库注册 DLL 搜索目录。

---

## 安装

### 自动安装（推荐）

```bash
# 为你的平台安装所有 EP
python scripts/setup_ep.py

# 仅安装特定 EP
python scripts/setup_ep.py cuda cpu

# 检查安装状态
python scripts/setup_ep.py --status
```

安装脚本会：
1. 为每种 EP 创建虚拟环境
2. 安装正确的 `onnxruntime` 变体
3. 验证安装

### 手动安装

如果只需要 CPU 推理，安装基础包即可：

```bash
pip install onnxruntime
```

GPU 加速：

```bash
# NVIDIA GPU
pip install onnxruntime-gpu

# Windows GPU（任何厂商）
pip install onnxruntime-directml
```

> 注意：手动安装不使用 venv 隔离。同一时间只能安装一种 EP 变体。

---

## 故障排除

### "Falling back to CPU"

**原因**：未检测到首选 EP 的硬件。

| EP | 检查项 | 修复方法 |
|----|--------|----------|
| CUDA | `nvidia-smi` 失败 | 安装 NVIDIA 驱动，确认 GPU 连接 |
| OpenVINO | 未检测到 Intel GPU | 此 EP 需要 Intel iGPU/dGPU |
| DirectML | — | 应在任何 Windows GPU 上正常工作 |

### "EP runtime not bundled"

**原因**：EP 的 onnxruntime 包未安装在 `ep_venvs/` 中。

**修复**：运行 `python scripts/setup_ep.py <ep_name>` 进行安装。

### 检查当前 EP 状态

在 ssook 中，进入 **Settings** 标签页。EP 状态区域显示：
- 当前选中的 EP
- 系统上可用的 EP
- ep_venvs 中已捆绑的 EP
- 被跳过的 EP 及原因和修复建议
- 是否发生了回退
