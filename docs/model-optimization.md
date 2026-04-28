# Model Optimization

[← Back to Index](index.md) | 🌐 [한국어](ko/model-optimization.md) | [日本語](ja/model-optimization.md) | [中文](zh/model-optimization.md)

ssook provides a full optimization pipeline to reduce model size, lower latency, and decrease memory usage — all without writing code. This document explains each technique, how it works, and how to use it in ssook.

---

## Table of Contents

- [Quantization](#quantization)
  - [Dynamic INT8](#dynamic-int8)
  - [Static INT8](#static-int8)
  - [FP16 Conversion](#fp16-conversion)
  - [Mixed Precision](#mixed-precision)
- [Pruning](#pruning)
  - [Weight Pruning](#weight-pruning)
  - [Channel Pruning](#channel-pruning)
- [Graph Optimization](#graph-optimization)
  - [ORT Graph Optimizer](#ort-graph-optimizer)
  - [ONNX Simplifier](#onnx-simplifier)
  - [Dead Node Elimination](#dead-node-elimination)
- [Optimization Pipeline](#optimization-pipeline)
- [Performance Comparison](#performance-comparison)
- [Tips & Caveats](#tips--caveats)

---

## Quantization

Quantization converts model weights (and optionally activations) from high-precision floating point (FP32) to lower-precision formats (INT8, FP16). This reduces model size and speeds up inference, especially on CPU.

### Dynamic INT8

**How it works:**
- Weights are converted from FP32 to INT8 **at export time**.
- Activations remain in FP32 and are quantized **on-the-fly** during inference.
- No calibration data is needed — the quantization range for weights is computed directly from the weight values.

**Trade-offs:**
- ✅ Simplest method — just select the model and run
- ✅ ~50% model size reduction
- ✅ 1.5–2× speedup on CPU
- ⚠️ Less optimal than static quantization because activation ranges are estimated at runtime

**How to use in ssook:**
1. Go to the **Tools** tab → **Calibration / Quantization**
2. Load your ONNX model
3. Select **Dynamic INT8**
4. Choose weight type (UInt8 or Int8)
5. Click **Run**

### Static INT8

**How it works:**
- Both weights and activations are converted to INT8.
- A **calibration dataset** (50–200 representative images) is used to measure the actual activation ranges during inference.
- ssook automatically preprocesses calibration images to match the model's input shape (resize, normalize, batch).
- The QDQ (Quantize-Dequantize) format inserts Q/DQ nodes around quantizable operations, preserving the graph structure for maximum compatibility.

**Trade-offs:**
- ✅ Best speedup: 2–4× on CPU
- ✅ ~75% model size reduction
- ✅ Minimal accuracy loss when calibration data is representative
- ⚠️ Requires calibration images from your target domain

**How to use in ssook:**
1. Go to **Tools** → **Calibration / Quantization**
2. Load your ONNX model
3. Select **Static INT8**
4. Set the calibration image directory and max images
5. Configure options (per-channel, QDQ format)
6. Click **Run** — a progress bar shows calibration progress

### FP16 Conversion

**How it works:**
- All FP32 weights and constants are converted to FP16 (half-precision float).
- FP16 uses 16 bits instead of 32, halving the storage per value.
- The representable range is smaller (±65,504 vs ±3.4×10³⁸), but sufficient for most neural network weights.

**Trade-offs:**
- ✅ ~50% model size reduction
- ✅ Virtually no accuracy loss for most models
- ✅ Significant speedup on GPUs with FP16 hardware (NVIDIA Tensor Cores, Apple Neural Engine)
- ⚠️ Minimal speedup on CPU (CPU FP16 support varies)
- ⚠️ Requires the `onnxconverter-common` package

**How to use in ssook:**
1. Go to **Tools** → **Calibration / Quantization**
2. Load your ONNX model
3. Select **FP16**
4. Click **Run**

### Mixed Precision

**How it works:**
- Not all layers respond equally well to quantization. Layers with wide weight ranges or many outlier values lose more accuracy when quantized.
- ssook computes a **sensitivity score** for each quantizable layer:
  ```
  sensitivity = weight_range × (1 + outlier_ratio × 10)
  ```
  where `outlier_ratio` is the fraction of weights beyond 3 standard deviations.
- The top N% most sensitive layers are **excluded** from INT8 quantization and kept in FP32.
- Remaining layers are quantized to INT8 normally.

**Trade-offs:**
- ✅ Better accuracy than full INT8 — sensitive layers stay in FP32
- ✅ Still achieves 2–3× speedup
- ✅ Automatic — no manual layer selection needed
- ⚠️ Slightly less compression than full INT8

**How to use in ssook:**
1. Go to **Tools** → **Calibration / Quantization**
2. Load your ONNX model
3. Select **Mixed Precision**
4. Set the exclusion percentage (default: 20% most sensitive layers)
5. Click **Run**

---

## Pruning

Pruning removes unnecessary weights or channels from a model to reduce its size and computational cost.

### Weight Pruning

**How it works:**
- **Magnitude-based unstructured pruning**: weights with the smallest absolute values contribute least to the output and can be set to zero.
- Given a `sparsity_ratio` (e.g., 0.3), the bottom 30% of weights by magnitude are zeroed out.
- Targets weight tensors in Conv, MatMul, Gemm, and ConvTranspose operations.
- The resulting sparse model is smaller when compressed and can be faster on runtimes that support sparse computation.

**Trade-offs:**
- ✅ 10–30% size reduction (more with compression)
- ✅ Minimal accuracy loss at moderate sparsity (≤30%)
- ⚠️ Speedup depends on runtime sparse support — standard ONNX Runtime does not accelerate sparse ops
- ⚠️ High sparsity (>50%) may noticeably degrade accuracy

**How to use in ssook:**
1. Go to **Tools** → **Model Optimization**
2. Load your ONNX model
3. Select **Weight Pruning**
4. Set sparsity ratio (0.0–1.0, recommended: 0.2–0.3)
5. Click **Run**

### Channel Pruning

**How it works:**
- **L1-norm structured pruning**: for each Conv layer, the L1-norm (sum of absolute values) of each output channel's weights is computed.
- Channels with the lowest L1-norm contribute least to the layer's output.
- The bottom N% of channels (by L1-norm) are removed entirely — both the weights and the corresponding bias values.
- Unlike weight pruning, this is **structured**: the resulting model has fewer channels and runs faster on any hardware without special sparse support.

**Trade-offs:**
- ✅ Direct FLOPs reduction proportional to pruned channels
- ✅ Faster on all hardware (no sparse runtime needed)
- ⚠️ May require fine-tuning to recover accuracy
- ⚠️ A minimum channel count is enforced to prevent layers from being pruned to zero

**How to use in ssook:**
1. Go to **Tools** → **Model Optimization**
2. Load your ONNX model
3. Select **Channel Pruning**
4. Set pruning ratio (recommended: 0.1–0.3) and minimum channels (default: 4)
5. Click **Run**

---

## Graph Optimization

Graph optimization restructures the computation graph to eliminate redundancy and fuse operations, without changing the model's mathematical behavior.

### ORT Graph Optimizer

**How it works:**
- Uses ONNX Runtime's built-in graph optimization passes.
- **Constant folding**: pre-computes operations whose inputs are all constants (e.g., Shape → Gather → Unsqueeze chains).
- **Operator fusion**: merges adjacent operations into single, optimized kernels:
  - Conv + BatchNormalization → single Conv (BN parameters folded into Conv weights)
  - Conv + Relu → ConvRelu fused kernel
  - MatMul + Add → fused Gemm
  - Conv + Add + Relu → single fused kernel
- Three optimization levels: **Basic** (constant folding), **Extended** (+ common subexpression elimination), **All** (+ advanced fusions).

**Trade-offs:**
- ✅ 5–15% latency reduction
- ✅ Zero accuracy loss — mathematically equivalent
- ✅ Always safe to apply

### ONNX Simplifier

**How it works:**
- Uses the `onnxsim` library to perform shape inference and constant folding at the ONNX graph level.
- Resolves dynamic shapes where possible, simplifies redundant operations, and produces a cleaner graph.

**Trade-offs:**
- ✅ Cleaner graph, often smaller file
- ✅ Zero accuracy loss
- ⚠️ Requires the `onnxsim` package

### Dead Node Elimination

**How it works:**
- Traces the graph backward from output nodes to find all reachable nodes.
- Any node not reachable from an output is removed.
- This cleans up leftover nodes from model export or previous transformations.

**Trade-offs:**
- ✅ Reduces graph size
- ✅ Zero accuracy loss
- ✅ Good to run before other optimizations

**How to use graph optimizations in ssook:**
1. Go to **Tools** → **Model Optimization**
2. Load your ONNX model
3. Select the optimization method
4. For ORT Graph Optimizer, choose the level (Basic / Extended / All)
5. Click **Run**

---

## Optimization Pipeline

ssook allows you to **chain multiple optimizations** in sequence. For example:

1. Dead Node Elimination → clean the graph
2. ORT Graph Optimizer → fuse operations
3. Channel Pruning → remove unimportant channels
4. Dynamic INT8 Quantization → compress weights

Each step takes the output of the previous step as input. The pipeline tracks size changes and results at each stage.

**How to use:**
1. Go to **Tools** → **Model Optimization**
2. Use the **Model Diagnosis** feature to get recommended optimization steps
3. The diagnosis engine suggests a prioritized pipeline based on your model's characteristics
4. Apply the recommended pipeline, or customize the steps

---

## Performance Comparison

Typical results on CPU (Intel i7-1165G7):

| Model | Method | Size | Inference Time | mAP@50 |
|-------|--------|------|---------------|--------|
| YOLOv8n FP32 | Original | 12.2 MB | ~45 ms | 37.3 |
| YOLOv8n INT8 (Dynamic) | Dynamic Quantization | 3.4 MB | ~25 ms | 36.8 |
| YOLOv8n INT8 (Static) | Static Quantization | 3.2 MB | ~18 ms | 37.0 |
| YOLOv8n FP16 | FP16 Conversion | 6.1 MB | ~38 ms | 37.3 |

> Results vary by hardware and model. Always verify accuracy using ssook's Evaluation tab after optimization.

---

## Tips & Caveats

1. **Start with Graph Optimization** — it's free (no accuracy loss) and makes subsequent optimizations more effective.
2. **Use Model Diagnosis first** — it analyzes your model and recommends the best optimization strategy.
3. **Verify accuracy after quantization** — use the Evaluation tab to compare mAP before and after.
4. **Calibration data matters** — for Static INT8, use images from your actual deployment domain (50–200 images is usually sufficient).
5. **Check opset version** — quantization requires opset ≥ 11. Models with lower opset may fail.
6. **CenterNet / DETR models** — some custom operations may not support INT8. Use FP16 or Mixed Precision instead.
7. **Channel pruning needs fine-tuning** — for best results, fine-tune the pruned model on your training data after pruning.
