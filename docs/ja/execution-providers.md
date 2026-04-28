# Execution Providers（実行プロバイダ）

[← インデックスに戻る](../index.md) | 🌐 [English](../execution-providers.md) | [한국어](../ko/execution-providers.md) | **日本語** | [中文](../zh/execution-providers.md)

ssookは、システムに最適なハードウェアアクセラレータ（Execution Provider）を自動的に選択します。本ドキュメントでは、EPシステムの仕組みと設定方法を説明します。

---

## 目次

- [概要](#概要)
- [サポートされるExecution Provider](#サポートされるexecution-provider)
- [自動選択の仕組み](#自動選択の仕組み)
- [venv分離アーキテクチャ](#venv分離アーキテクチャ)
- [インストール](#インストール)
- [トラブルシューティング](#トラブルシューティング)

---

## 概要

ONNX Runtimeは、**Execution Providers（EP）**と呼ばれる複数のハードウェアバックエンドをサポートしています。各EPは異なるハードウェアをターゲットとします：

- **CUDA** → NVIDIA GPU
- **DirectML** → 任意のWindows GPU（NVIDIA、AMD、Intel）
- **OpenVINO** → Intel CPUおよびiGPU
- **CoreML** → Apple Silicon（M1/M2/M3）
- **CPU** → 任意のプロセッサ（フォールバック）

課題：各EPには `onnxruntime` パッケージの**異なるビルド**が必要であり、互いに競合します。同じPython環境に `onnxruntime-gpu` と `onnxruntime-directml` を同時にインストールすることはできません。

ssookはこの問題を**venv分離**で解決します — 各EPが独自の分離された環境を持ちます。

---

## サポートされるExecution Provider

| EP | プラットフォーム | ハードウェア | パッケージ |
|----|----------|----------|---------|
| CUDA | Windows, Linux | NVIDIA GPU | `onnxruntime-gpu` |
| DirectML | Windows | 任意のGPU | `onnxruntime-directml` |
| OpenVINO | Windows, Linux | Intel CPU/iGPU | `onnxruntime-openvino` |
| CoreML | macOS | Apple Silicon | `onnxruntime-coreml` |
| CPU | すべて | 任意のCPU | `onnxruntime` |

---

## 自動選択の仕組み

起動時、`onnxruntime` がインポートされる**前に**、ssookはEPセレクタを実行します：

```
┌─────────────────────────────┐
│   プラットフォーム検出（OS）    │
│   Windows / Linux / macOS   │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  優先度リストを取得           │
│  Win: cuda→directml→        │
│       openvino→cpu          │
│  Linux: cuda→openvino→cpu   │
│  macOS: coreml→cpu          │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  各EPについて優先順に：       │
│  1. ランタイムがバンドル済み？ │
│  2. ハードウェアが存在？      │
│  → 両方Yes：選択             │
│  → No：スキップ、理由を記録   │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  選択されたEPの               │
│  site-packagesを             │
│  sys.path[0]に挿入           │
│  + DLLディレクトリを追加      │
│  （Windows）                 │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  import onnxruntime         │
│  （正しいビルドが読み込まれる）│
└─────────────────────────────┘
```

**ハードウェア検出方法：**

| EP | 検出方法 |
|----|-----------------|
| CUDA | `nvidia-smi --query-gpu=name` を実行 — GPU名が返されればNVIDIA GPUが存在 |
| OpenVINO | Windows：`wmic path win32_videocontroller get name` で"intel"を確認。Linux：`/dev/dri` の存在を確認 |
| DirectML | 任意のWindows GPU（追加チェック不要） |
| CoreML | macOSプラットフォームチェック（Apple Silicon） |
| CPU | 常に利用可能 |

**フォールバック動作：**
最優先のEPが利用できない場合（ハードウェアなし、ランタイム未インストール）、ssookは優先度リストの次のEPにフォールバックします。フォールバックの理由は記録され、修正提案とともにSettingsタブに表示されます。

---

## venv分離アーキテクチャ

各EPランタイムは独自の分離されたディレクトリに配置されます：

```
ep_venvs/                          （ソースモード）
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

ep_runtimes/                       （フリーズexeモード）
├── cuda/
│   └── onnxruntime/
├── directml/
│   └── onnxruntime/
└── cpu/
    └── onnxruntime/
```

EPが選択されると、その `site-packages` ディレクトリが `sys.path` の**先頭**に挿入されます。これにより、`import onnxruntime` が選択されたEPの正しいビルドを読み込むことが保証されます。

Windowsでは、CUDA/TensorRT/DirectMLの共有ライブラリ用にDLL検索ディレクトリも登録されます。

---

## インストール

### 自動（推奨）

```bash
# プラットフォームのすべてのEPをインストール
python scripts/setup_ep.py

# 特定のEPのみインストール
python scripts/setup_ep.py cuda cpu

# インストール状態を確認
python scripts/setup_ep.py --status
```

セットアップスクリプトの動作：
1. 各EPの仮想環境を作成
2. 正しい `onnxruntime` バリアントをインストール
3. インストールを検証

### 手動

CPU推論のみが必要な場合は、ベースパッケージをインストールするだけです：

```bash
pip install onnxruntime
```

GPUアクセラレーションの場合：

```bash
# NVIDIA GPU
pip install onnxruntime-gpu

# Windows GPU（任意のベンダー）
pip install onnxruntime-directml
```

> 注意：手動インストールではvenv分離は使用されません。一度に1つのEPバリアントのみインストール可能です。

---

## トラブルシューティング

### "Falling back to CPU"

**原因**：優先EPのハードウェアが検出されませんでした。

| EP | 確認事項 | 修正方法 |
|----|-------|-----|
| CUDA | `nvidia-smi` が失敗 | NVIDIAドライバをインストール、GPU接続を確認 |
| OpenVINO | Intel GPUが検出されない | このEPにはIntel iGPU/dGPUが必要 |
| DirectML | — | 任意のWindows GPUで動作するはず |

### "EP runtime not bundled"

**原因**：EPのonnxruntimeパッケージが `ep_venvs/` にインストールされていません。

**修正方法**：`python scripts/setup_ep.py <ep_name>` を実行してインストールしてください。

### 現在のEP状態の確認

ssookで **Settings** タブに移動してください。EP状態セクションには以下が表示されます：
- 現在選択されているEP
- システムで利用可能なEP
- ep_venvsにバンドルされているEP
- スキップされたEPとその理由および修正提案
- フォールバックが発生したかどうか
