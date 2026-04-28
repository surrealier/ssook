# Evaluation Metrics

[← Back to Index](index.md) | 🌐 [한국어](ko/evaluation-metrics.md) | [日本語](ja/evaluation-metrics.md) | [中文](zh/evaluation-metrics.md)

ssook evaluates models across five task types — Detection, Classification, Segmentation, CLIP, and Embedder — each with its own set of metrics. This document explains how each metric is computed and how to interpret the results.

---

## Table of Contents

- [Detection Metrics](#detection-metrics)
  - [IoU (Intersection over Union)](#iou-intersection-over-union)
  - [Precision, Recall, F1](#precision-recall-f1)
  - [mAP@50](#map50)
  - [mAP@50:95](#map5095)
- [Segmentation Metrics](#segmentation-metrics)
- [Classification Metrics](#classification-metrics)
- [Embedder Metrics](#embedder-metrics)
- [Confidence Optimizer](#confidence-optimizer)
- [FP/FN Error Analysis](#fpfn-error-analysis)

---

## Detection Metrics

### IoU (Intersection over Union)

IoU measures how well a predicted bounding box overlaps with the ground truth box:

```
IoU = Area of Overlap / Area of Union
```

- **IoU = 1.0**: Perfect overlap
- **IoU = 0.5**: The standard threshold — a prediction with IoU ≥ 0.5 is considered a correct detection (True Positive)
- **IoU = 0.0**: No overlap at all

ssook computes IoU using a **vectorized matrix operation**: given M predictions and N ground truths, it computes all M×N IoU values simultaneously for efficiency.

### Precision, Recall, F1

After matching predictions to ground truths using IoU:

- **True Positive (TP)**: Prediction matches a ground truth (IoU ≥ threshold, correct class)
- **False Positive (FP)**: Prediction doesn't match any ground truth
- **False Negative (FN)**: Ground truth not matched by any prediction

```
Precision = TP / (TP + FP)    — "Of all detections, how many are correct?"
Recall    = TP / (TP + FN)    — "Of all ground truths, how many were found?"
F1        = 2 × P × R / (P + R) — Harmonic mean of Precision and Recall
```

### mAP@50

Mean Average Precision at IoU threshold 0.5:

1. For each class, sort all predictions by confidence (highest first).
2. Walk through predictions in order. For each prediction, check if it matches an unmatched ground truth with IoU ≥ 0.5.
3. Compute cumulative Precision and Recall at each step.
4. Compute **AP** (Average Precision) using **101-point interpolation**: sample the precision-recall curve at 101 evenly spaced recall points (0.00, 0.01, ..., 1.00) and average the interpolated precision values.
5. **mAP@50** = mean of AP across all classes.

### mAP@50:95

A stricter metric that averages mAP across 10 IoU thresholds: 0.50, 0.55, 0.60, ..., 0.95.

ssook optimizes this by computing the IoU matrix once and reusing it across all thresholds, rather than running 10 separate evaluations.

Higher IoU thresholds require more precise localization, so mAP@50:95 is always lower than mAP@50.

---

## Segmentation Metrics

For semantic segmentation models (output: C×H×W class probability maps):

### IoU (per-class)

```
IoU = |Pred ∩ GT| / |Pred ∪ GT|
```

Computed per-pixel: count pixels where both prediction and ground truth agree on the class (intersection), divided by pixels where either predicts the class (union).

### Dice Coefficient (per-class)

```
Dice = 2 × |Pred ∩ GT| / (|Pred| + |GT|)
```

Similar to IoU but gives more weight to the intersection. Dice is always ≥ IoU for the same prediction.

### mIoU / mDice

Mean IoU and mean Dice across all classes that have ground truth pixels. Classes with no ground truth pixels are excluded from the mean.

**How to use:**
1. Go to **Evaluation** tab
2. Select task type: **Segmentation**
3. Load the segmentation model and ground truth masks
4. Run evaluation — per-class IoU/Dice and overall mIoU/mDice are displayed

---

## Classification Metrics

For classification models (output: class probabilities):

- **Accuracy**: Fraction of correctly classified images
- **Per-class Precision/Recall/F1**: Computed using one-vs-rest approach for each class
- **Overall P/R/F1**: Macro-average across all classes

**How to use:**
1. Go to **Evaluation** tab
2. Select task type: **Classification**
3. Load the classification model and ground truth labels
4. Run evaluation

---

## Embedder Metrics

For feature extraction models (output: embedding vectors):

### Cosine Similarity

```
similarity = (A · B) / (‖A‖ × ‖B‖)
```

Embeddings are L2-normalized before comparison, so cosine similarity equals the dot product. Range: -1 (opposite) to 1 (identical).

### Retrieval@1

For each query image, find the most similar gallery image by cosine similarity. Retrieval@1 = fraction of queries where the top-1 result has the correct class label.

### Retrieval@K

Same as Retrieval@1, but the query is considered correct if the correct class appears anywhere in the top-K results.

**How to use:**
1. Go to **Evaluation** tab
2. Select task type: **Embedder**
3. Load the feature extractor model
4. Set the dataset directory (folder structure: `class_name/image.jpg`)
5. Run evaluation — Retrieval@1, Retrieval@K, and mean cosine similarity are displayed

---

## Confidence Optimizer

The Confidence Optimizer finds the optimal confidence threshold for each class by sweeping thresholds and measuring F1.

**How it works:**
1. Run inference on all evaluation images to get predictions with confidence scores.
2. For each class, sweep the confidence threshold from 0.0 to 1.0.
3. At each threshold, compute Precision, Recall, and F1.
4. Plot the **PR curve** (Precision vs. Recall) for each class.
5. Identify the threshold that maximizes F1 for each class.

**Why per-class thresholds matter:**
Different classes may have different optimal thresholds. A "person" class with many examples might work best at 0.3, while a rare "wheelchair" class might need 0.5 to avoid false positives.

**How to use:**
1. Go to **Analysis** tab → **Confidence Optimizer**
2. Load a model and evaluation dataset
3. Click **Run** — PR curves and optimal thresholds are displayed per class
4. The F1-maximizing threshold for each class is highlighted

---

## FP/FN Error Analysis

The Error Analyzer automatically classifies detection errors to help you understand where and why the model fails.

**Error classification:**

| Error Type | Definition |
|-----------|-----------|
| **False Positive (FP)** | Model detects an object where none exists in ground truth |
| **False Negative (FN)** | Ground truth object not detected by the model |

**Size-based breakdown:**

Errors are categorized by bounding box area:

| Size | Area Range | Typical Objects |
|------|-----------|----------------|
| **Small (S)** | < 32² pixels | Distant people, small signs |
| **Medium (M)** | 32²–96² pixels | Nearby people, vehicles |
| **Large (L)** | > 96² pixels | Close-up objects, large vehicles |

**Position-based analysis:**

Errors are also analyzed by position in the image (center vs. edges), helping identify if the model struggles with objects at certain locations.

**How to use:**
1. Go to **Analysis** tab → **FP/FN Error Analysis**
2. Load a model and evaluation dataset with ground truth
3. Click **Run**
4. Review the error breakdown by type, size, and position
5. Use the insights to guide data collection (e.g., if small objects have high FN rate, collect more small-object training data)
