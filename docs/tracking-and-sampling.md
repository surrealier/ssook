# Tracking & Sampling

[← Back to Index](index.md) | 🌐 [한국어](ko/tracking-and-sampling.md) | [日本語](ja/tracking-and-sampling.md) | [中文](zh/tracking-and-sampling.md)

ssook includes object tracking for video inference, smart sampling for dataset curation, and near-duplicate detection for data cleaning. This document explains how each works.

---

## Table of Contents

- [Object Tracking](#object-tracking)
  - [SORT](#sort)
  - [ByteTrack](#bytetrack)
  - [Parameters](#tracking-parameters)
- [Smart Sampler](#smart-sampler)
  - [Balanced Sampling](#balanced-sampling)
  - [Stratified Sampling](#stratified-sampling)
  - [Random Sampling](#random-sampling)
- [Near-Duplicate Detection](#near-duplicate-detection)

---

## Object Tracking

Object tracking assigns consistent IDs to detected objects across video frames. ssook provides two tracking algorithms that work on top of any detection model.

### SORT

**Simple Online and Realtime Tracking**

SORT is a straightforward IoU-based tracker:

1. **Detection**: The detection model produces bounding boxes for the current frame.
2. **IoU matching**: Compute the IoU (overlap) between every existing track's last known position and every new detection. This produces an M×N cost matrix.
3. **Greedy assignment**: Match tracks to detections by picking the highest IoU pairs first. A match requires IoU ≥ `iou_threshold` (default: 0.3).
4. **Update matched tracks**: Update the track's position, increment hit count, reset time-since-update.
5. **Create new tracks**: Unmatched detections become new tracks with new IDs.
6. **Age unmatched tracks**: Tracks not matched in this frame get their time-since-update incremented. Tracks exceeding `max_age` frames without a match are removed.
7. **Output**: Only tracks with ≥ `min_hits` total matches are reported (prevents flickering from single-frame detections).

Each track maintains a **trajectory** — the last 60 center points — for visualization.

### ByteTrack

ByteTrack improves on SORT by using **two-stage association** to recover low-confidence detections:

**Stage 1 — High-confidence matching:**
1. Split detections into **high-confidence** (score ≥ `high_thresh`, default: 0.5) and **low-confidence** (score ≥ `low_thresh`, default: 0.1).
2. Match all existing tracks against high-confidence detections using IoU, same as SORT.
3. Update matched tracks; collect unmatched tracks.

**Stage 2 — Low-confidence recovery:**
4. Match the **remaining unmatched tracks** against low-confidence detections using IoU.
5. This recovers tracks that were temporarily occluded or had low detection confidence.
6. Tracks still unmatched after both stages get their time-since-update incremented.

**New track creation:**
7. Only unmatched **high-confidence** detections create new tracks. Low-confidence detections alone don't start new tracks (prevents noise).

**Why ByteTrack is better:**
- Standard trackers discard low-confidence detections entirely, causing track fragmentation when objects are partially occluded.
- ByteTrack uses low-confidence detections to maintain existing tracks, but doesn't let them create new tracks (avoiding false positives).

### Tracking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_age` | 30 | Frames to keep a track alive without matches |
| `min_hits` | 3 | Minimum matches before a track is reported |
| `iou_threshold` | 0.3 | Minimum IoU for a valid match |
| `high_thresh` | 0.5 | ByteTrack: minimum confidence for first-stage matching |
| `low_thresh` | 0.1 | ByteTrack: minimum confidence for second-stage matching |

**How to use in ssook:**
1. Go to **Viewer** tab
2. Load a detection model and open a video
3. Enable tracking and select the algorithm (ByteTrack or SORT)
4. Objects are assigned persistent IDs with trajectory trails

---

## Smart Sampler

When working with large datasets, you often need a representative subset. The Smart Sampler provides three strategies.

### Balanced Sampling

**Goal**: Equal representation of all classes, with spatial diversity within each class.

**How it works:**
1. Parse all label files to build a map of class → images.
2. Divide the target count equally among classes: `per_class = target / num_classes`.
3. For each class, if the pool is larger than `per_class`, use **farthest-point sampling**:
   - Compute a feature for each image: the mean bounding box center `[cx, cy]` (normalized 0–1).
   - Start with a random seed image.
   - Iteratively select the image whose feature is **farthest** from all already-selected images (using squared Euclidean distance).
   - This ensures spatial diversity — selected images cover different regions of the image space rather than clustering in one area.
4. If the pool is smaller than `per_class`, take all images from that class.

**When to use**: When you have class imbalance and want equal representation, or when you need diverse spatial coverage.

### Stratified Sampling

**Goal**: Maintain the original class distribution at a smaller scale.

**How it works:**
1. For each class, compute its proportion: `class_count / total_associations`.
2. Sample `proportion × target` images from each class (randomly).
3. If the total is less than the target, fill the remaining slots with random images from the unselected pool.

**When to use**: When you want a smaller dataset that statistically represents the original distribution.

### Random Sampling

**Goal**: Simple unbiased random subset.

**How it works**: `random.sample(all_images, target)` with a fixed seed for reproducibility.

**When to use**: When class balance doesn't matter, or as a baseline.

**How to use in ssook:**
1. Go to **Data** tab → **Smart Sampler**
2. Set image directory, label directory, and output directory
3. Select strategy (Balanced / Stratified / Random)
4. Set target count and random seed
5. Click **Run** — a before/after class distribution table is shown

---

## Near-Duplicate Detection

Near-duplicate detection finds images that are perceptually similar (but not necessarily byte-identical) using **dHash** (difference hash).

**How dHash works:**

1. **Resize** the image to 9×8 pixels (grayscale). This removes high-frequency detail and normalizes size.
2. **Compute horizontal gradients**: For each row, compare adjacent pixels. If the left pixel is brighter than the right, output 1; otherwise 0.
3. This produces a **64-bit binary hash** (8 rows × 8 comparisons).
4. **Compare hashes**: The **Hamming distance** (number of differing bits) between two hashes measures perceptual difference.

**Threshold interpretation:**

| Threshold | Meaning |
|-----------|---------|
| 0 | Exact perceptual match (identical images) |
| 1–5 | Very similar (minor compression artifacts, slight crop) |
| 6–10 | Similar (small edits, color adjustments) — **default: 10** |
| 11–20 | Somewhat similar (noticeable differences) |
| 21–64 | Increasingly different |

**Why dHash over pixel comparison:**
- Robust to resizing, minor cropping, and compression artifacts
- Extremely fast: O(1) per comparison after hashing
- Works across different image formats (JPEG vs PNG)

**How to use in ssook:**
1. Go to **Data** tab → **Near-Duplicate Detector**
2. Set the image directory
3. Set the threshold (default: 10)
4. Click **Run** — duplicate pairs are listed with their Hamming distance
