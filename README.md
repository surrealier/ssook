<div align="center">

<img src="assets/icon.png" width="120" alt="Visualizer Logo" />

# Visualizer

**All-in-one desktop toolkit for AI model inference, evaluation, analysis & data management**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PySide6](https://img.shields.io/badge/UI-PySide6-41CD52?logo=qt&logoColor=white)](https://doc.qt.io/qtforpython/)
[![ONNX Runtime](https://img.shields.io/badge/runtime-ONNX-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-blue)](#)
[![Tests](https://img.shields.io/badge/tests-26%20passed-brightgreen)](#testing)

*Stop juggling between terminals, notebooks, and scripts.*
*Visualizer brings inference, evaluation, error analysis, and dataset management into a single GUI.*

[Getting Started](#-getting-started) · [Features](#-features) · [Supported Models](#-supported-models) · [Screenshots](#-screenshots)

</div>

---

## Why Visualizer?

Building and deploying AI models involves constant context-switching between tools. You run inference in one script, evaluate metrics in another, analyze errors in a notebook, and manage datasets in yet another tool.

Visualizer eliminates this friction:

- **One GUI** — Inference, evaluation, analysis, and data management in a single window
- **Multi-task** — Detection, Classification, Segmentation, CLIP zero-shot, Embeddings
- **Multi-model comparison** — Side-by-side A/B testing with synchronized navigation
- **10+ data tools** — From augmentation preview to duplicate detection, all built-in
- **No code required** — Load a model, point to data, get results

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎬 Real-time Viewer
- Video/image inference with live overlay
- Detection: bounding boxes + class + confidence
- Classification: top-5 probability display
- Playback speed control, frame-by-frame navigation
- Snapshot capture & video recording
- CSV detection logging

</td>
<td width="50%">

### 📊 Multi-Model Evaluation
- Evaluate multiple models simultaneously
- Detection: mAP@50, mAP@50:95, per-class AP/P/R/F1
- Classification: Accuracy, per-class P/R/F1
- Segmentation: mIoU, mDice, per-class IoU/Dice
- CLIP: zero-shot classification accuracy
- Embedder: Retrieval@1/@K, cosine similarity
- Export to Excel / HTML reports

</td>
</tr>
<tr>
<td>

### 🔍 Deep Analysis
- **Inference Analysis** — Letterbox visualization, tensor inspection
- **Model Compare** — Side-by-side A/B with slider navigation
- **FP/FN Analysis** — Auto-classify by size (S/M/L) & position
- **Conf Optimizer** — Per-class threshold sweep with PR/F1 curves
- **Embedding Viewer** — t-SNE / UMAP / PCA 2D visualization

</td>
<td>

### 📁 Dataset Management
- **Explorer** — FiftyOne-style thumbnail gallery with filters
- **Splitter** — Stratified train/val/test split
- **Format Converter** — YOLO ↔ COCO JSON ↔ Pascal VOC XML
- **Class Remapper** — Remap, merge, or delete classes in bulk
- **Dataset Merger** — Combine datasets with dHash deduplication

</td>
</tr>
<tr>
<td>

### 🛡️ Data Quality Tools
- **Label Anomaly Detector** — OOB boxes, size outliers, overlaps
- **Image Quality Checker** — Blur, brightness, entropy, aspect ratio
- **Near-Duplicate Detector** — dHash perceptual hashing
- **Leaky Split Detector** — Cross-split duplicate detection
- **Similarity Search** — Cosine similarity top-K retrieval

</td>
<td>

### 🚀 Batch & Automation
- **Batch Inference** — Folder → model → export results
- **Output formats** — YOLO txt / JSON / CSV
- **Visualization export** — Save annotated images
- **Augmentation Preview** — Mosaic, Albumentations, quick test
- **Smart Sampler** — Stratified / balanced / random sampling
- **Benchmark** — FPS & latency across CPU/CUDA/TensorRT

</td>
</tr>
</table>

---

## 🤖 Supported Models

| Task | Model Format | Metrics |
|------|-------------|---------|
| **Detection** | YOLO v5/v8/v9/v11, CenterNet (Darknet) | mAP@50, mAP@50:95, Precision, Recall, F1 |
| **Classification** | ONNX (2D output) | Accuracy, per-class Precision/Recall/F1 |
| **Segmentation** | ONNX (C×H×W output) | mIoU, mDice, per-class IoU/Dice |
| **VLM / CLIP** | Image Encoder + Text Encoder ONNX | Zero-shot Classification |
| **Embedder** | ONNX (feature extractor) | Retrieval@1/@K, Cosine Similarity |

> **Fixed-batch models** (e.g., batch=4) are automatically detected and handled — single images are padded to match the batch dimension.

---

## 📸 Screenshots

<!-- Replace these placeholders with actual screenshots -->

<div align="center">

| Viewer | Evaluation | Analysis |
|:------:|:----------:|:--------:|
| ![Viewer](https://via.placeholder.com/300x200/1a1a2e/e0e0e0?text=Viewer+Tab) | ![Evaluation](https://via.placeholder.com/300x200/16213e/e0e0e0?text=Evaluation+Tab) | ![Analysis](https://via.placeholder.com/300x200/0f3460/e0e0e0?text=Analysis+Tab) |

| Dataset Explorer | Model Compare | Embedding Viewer |
|:----------------:|:-------------:|:----------------:|
| ![Explorer](https://via.placeholder.com/300x200/533483/e0e0e0?text=Dataset+Explorer) | ![Compare](https://via.placeholder.com/300x200/e94560/e0e0e0?text=Model+Compare) | ![Embedding](https://via.placeholder.com/300x200/0a1931/e0e0e0?text=Embedding+Viewer) |

</div>

> 💡 **Tip:** Replace placeholder images with actual screenshots. Capture each tab and save to `docs/screenshots/`.

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- (Optional) CUDA-compatible GPU for accelerated inference

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/visualizer.git
cd visualizer

# Install dependencies
pip install -r requirements.txt

# Or with optional extras
pip install -e ".[all]"    # matplotlib, scikit-learn, openpyxl, umap-learn
pip install -e ".[charts]" # matplotlib, scikit-learn, openpyxl only
```

### GPU Support

```bash
# For CUDA acceleration (install separately)
pip install onnxruntime-gpu

# For PyTorch model loading
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Launch

```bash
python run.py
```

### Build Standalone Executable

```bash
pip install pyinstaller
pyinstaller visualizer.spec
```

---

## 📖 Usage Guide

### Quick Start Workflow

```
1. Launch  →  Settings tab  →  Load ONNX model
2. Viewer tab  →  Open video/image  →  See real-time inference
3. Evaluation tab  →  Set GT labels folder  →  Run evaluation
4. Analysis tab  →  Dive into FP/FN, confidence optimization
5. Data tab  →  Explore, split, clean, augment your dataset
```

### Tab Overview

<details>
<summary><b>🎬 Viewer</b> — Real-time inference visualization</summary>

- Open any image or video file
- Detection results overlay with bounding boxes, class names, and confidence scores
- Classification shows top-5 predictions in the corner
- Controls: play/pause, speed adjustment, frame-by-frame (prev/next), snapshot, record
- CSV logging of all detections per frame

</details>

<details>
<summary><b>⚙ Settings</b> — Model & display configuration</summary>

- Load ONNX or PT models
- Adjust confidence threshold and NMS settings
- Customize per-class: enable/disable, color (RGB), box thickness
- All settings persisted in YAML (`settings/app_config.yaml`)

</details>

<details>
<summary><b>📊 Evaluation</b> — Comprehensive model benchmarking</summary>

- **Detection/Classification**: Compare multiple models side-by-side against GT labels (YOLO txt format)
- **Segmentation**: GT masks as PNG (pixel value = class ID), overlay comparison
- **CLIP**: Load image + text encoders, input class labels for zero-shot evaluation
- **Embedder**: Extract embeddings from folder structure (subfolder = class), compute Retrieval@K
- **Benchmark**: Measure FPS and latency across execution providers (CPU/CUDA/TensorRT)
- Export results to Excel or HTML

</details>

<details>
<summary><b>🔍 Analysis</b> — Deep-dive into model behavior</summary>

- **Inference Analysis**: Single-image detailed inspection with letterbox and tensor visualization
- **Model Compare**: Load 2 models, navigate same image set with slider, compare box counts and inference time
- **FP/FN Analysis**: Automatic false positive/negative classification based on GT matching, with size (Small/Medium/Large) and position (Top/Center/Bottom) breakdowns, crop thumbnail gallery
- **Confidence Optimizer**: Per-class threshold sweep (0.05–0.95), PR + F1 curve visualization, auto-find F1-maximizing threshold
- **Embedding Viewer**: t-SNE / UMAP / PCA 2D scatter plots, label by folder name / filename prefix / none, per-class coloring with legend

</details>

<details>
<summary><b>📁 Data</b> — Complete dataset management suite</summary>

**Explorer** (FiftyOne-style)
- Thumbnail gallery with image + label overlay
- Filter by class, missing labels, empty labels, min box count
- Distribution charts: class balance, box sizes, aspect ratios, boxes per image

**Splitter**
- Train/Val/Test ratio configuration
- Stratified split to maintain class balance
- Copy or symlink mode
- Output: `{split}/images/`, `{split}/labels/`

**Format Converter**
- Bidirectional: YOLO ↔ COCO JSON ↔ Pascal VOC XML
- Batch conversion with progress tracking

**Class Remapper**
- Remap class IDs, merge multiple classes into one, or delete classes
- Auto-reindex option for contiguous IDs
- Bulk processing of all label files

**Dataset Merger**
- Combine multiple datasets into one
- dHash-based duplicate detection with configurable threshold
- Per-dataset prefix to avoid filename collisions

</details>

<details>
<summary><b>🛡️ Data Quality</b> — Automated quality assurance</summary>

**Label Anomaly Detector**
- Out-of-bounds bbox detection
- Size outlier detection per class
- Excessive overlap (high IoU) detection
- Visual inspection with image preview

**Image Quality Checker**
- Blur detection (Laplacian variance)
- Brightness analysis (too dark / too bright)
- Overexposure percentage
- Entropy (information content) check
- Abnormal aspect ratio flagging

**Near-Duplicate Detector**
- dHash perceptual hashing
- Configurable Hamming distance threshold
- Group duplicates with visual preview

**Leaky Split Detector**
- Cross-split (train/val/test) duplicate detection
- dHash-based comparison across all split pairs
- Prevents data leakage in your experiments

**Similarity Search**
- Build index from image folder (64×64 grayscale vectors)
- Query any image → top-K most similar results
- Cosine similarity scoring

</details>

<details>
<summary><b>🚀 Batch Inference</b> — Bulk processing</summary>

- Point to an image folder and a model
- Auto-detect task type (Detection / Classification)
- Export: YOLO txt, JSON, or CSV
- Optional: save visualization images with overlays

</details>

<details>
<summary><b>🎨 Augmentation Preview</b> — Visual augmentation testing</summary>

- 2×2 Mosaic augmentation with label adjustment
- OpenCV-based augmentations (flip, rotate, brightness, etc.)
- Albumentations integration (if installed)
- Preview before applying to full dataset
- Quick YOLO training test integration

</details>

<details>
<summary><b>📊 Smart Sampler</b> — Intelligent data sampling</summary>

- **Random**: Simple random subset
- **Balanced**: Equal samples per class
- **Stratified**: Maintain original class distribution
- Configurable target count and random seed
- Copy images + labels to output directory

</details>

---

## ⚙ Configuration

All settings are stored in `settings/app_config.yaml` and persist across sessions:

```yaml
# Model settings
model_type: yolo
conf_threshold: 0.25
batch_size: 1

# Display settings
box_thickness: 2
label_size: 0.55
show_labels: true
show_confidence: true

# Per-class customization
class_styles:
  0:
    color: null        # null = auto-assign
    enabled: true
    thickness: null     # null = use global
  3:
    color: [255, 25, 71]  # Custom RGB
    enabled: true
    thickness: null
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage (if installed)
python -m pytest tests/ -v --cov=core
```

---

## 📦 Dependencies

### Required

| Package | Version | Purpose |
|---------|---------|---------|
| PySide6 | ≥ 6.8.0 | Qt-based GUI framework |
| opencv-python | ≥ 4.10.0 | Image/video processing |
| numpy | ≥ 1.26.0 | Numerical operations |
| onnxruntime | ≥ 1.17.0 | ONNX model inference |
| psutil | ≥ 5.9.0 | System resource monitoring |
| PyYAML | ≥ 6.0.0 | Configuration management |

### Optional

| Package | Version | Purpose |
|---------|---------|---------|
| matplotlib | ≥ 3.8.0 | Charts & distribution plots |
| scikit-learn | ≥ 1.4.0 | t-SNE, PCA dimensionality reduction |
| openpyxl | ≥ 3.1.0 | Excel report export |
| umap-learn | ≥ 0.5.0 | UMAP embedding visualization |
| albumentations | — | Advanced augmentation pipeline |
| onnxruntime-gpu | — | CUDA/TensorRT acceleration |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

[⬆ Back to Top](#visualizer)

</div>
