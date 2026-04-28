# General Features

[← Back to Index](index.md) | 🌐 [한국어](ko/general-features.md) | [日本語](ja/general-features.md) | [中文](zh/general-features.md)

This page provides a brief overview of ssook's general features that don't require detailed technical explanation.

---

## Inference

### Real-time Viewer

Load an ONNX model and open a video or image to see inference results in real time.

- Supports Detection, Classification, and Segmentation models
- YOLO v5/v8/v9/v11, CenterNet (Darknet), and custom ONNX models
- Controls: play/pause, frame skip, speed adjustment, seek bar
- Snapshot capture and crop saving
- Configurable box thickness, label size, confidence threshold
- Fixed-batch models (e.g., batch=4) are automatically detected and handled

### Model A/B Compare

Run two models on the same images and compare results side-by-side with a slider.

### Benchmark

Measure inference performance with detailed statistics:

- **FPS**: Frames per second
- **Latency**: P50, P95, P99 percentiles
- **System usage**: CPU and GPU utilization during inference
- **Export**: System info and results to CSV

---

## Data Management

### Dataset Explorer

Browse and analyze your dataset with multiple view modes:

- **File list**: Image thumbnails with bounding box counts
- **Class distribution (by box)**: How many boxes per class
- **Class distribution (by image)**: How many images contain each class
- **Box size distribution**: Small / Medium / Large breakdown
- **Aspect ratio distribution**: Width/height ratio histogram

Filters: multi-class checkbox filter, box count filter (>=, =, <=). Double-click an image to see it with bounding box overlay.

### Dataset Splitter

Split a dataset into train/val/test sets:

- **Random split**: Shuffle and divide by ratio
- **Stratified split**: Maintain class proportions across splits
- Custom ratios (e.g., 0.8/0.2/0.0 to skip test set)
- Progress tracking

### Format Converter

Batch convert annotation formats:

- **YOLO** ↔ **COCO JSON** ↔ **Pascal VOC XML**
- Handles class mapping automatically
- Recursive folder support

### Class Remapper

Remap, merge, or delete class IDs in bulk:

- Change class ID 5 → 0
- Merge classes 3 and 4 into class 2
- Delete all annotations for class 7
- Recursive folder support

### Dataset Merger

Combine multiple datasets into one:

- Merges images and labels from multiple source directories
- **dHash duplicate detection**: Identifies perceptually similar images to avoid duplicates
- Configurable similarity threshold

### Label Anomaly Detector

Find problematic annotations:

- **Out-of-bounds boxes**: Bounding boxes extending beyond image edges
- **Size outliers**: Abnormally small or large boxes
- **Excessive overlaps**: Highly overlapping boxes that may be duplicates

### Image Quality Checker

Detect image quality issues:

- **Blur detection**: Laplacian variance below threshold
- **Brightness issues**: Too dark or too bright images
- **Overexposure**: Washed-out images
- **Abnormal aspect ratios**: Unusually wide or tall images

### Leaky Split Detector

Find duplicate images across train/val/test splits. If the same image appears in both training and validation sets, evaluation results are unreliable.

- Compares images across split directories using perceptual hashing
- Reports duplicate pairs with their locations

### Similarity Search

Query any image and find the top-K most similar images in your dataset using perceptual hashing.

### Augmentation Preview

Preview augmentation effects before applying them to your dataset:

- **Mosaic**: Combine 4 images into one
- **Flip**: Horizontal/vertical
- **Rotate**: Arbitrary angle
- **Albumentations**: Various transforms

---

## Settings

Configuration is stored in `settings/app_config.yaml` and persists across sessions:

- **Model type**: YOLO, CenterNet, Classification, Segmentation, CLIP, Embedder, Custom
- **Confidence threshold**: Minimum detection confidence (default: 0.25)
- **Display options**: Box thickness, label size, show/hide labels and confidence
- **Class-specific styles**: Per-class colors and display settings
- **Language**: English / Korean
- **Test model download**: Download sample models and test data from the Settings tab
