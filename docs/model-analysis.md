# Model Analysis

[← Back to Index](index.md) | 🌐 [한국어](ko/model-analysis.md) | [日本語](ja/model-analysis.md) | [中文](zh/model-analysis.md)

ssook provides three complementary tools to understand your ONNX model before optimizing or deploying it: **Model Diagnosis**, **Model Profiler**, and **Model Inspector**.

---

## Table of Contents

- [Model Diagnosis](#model-diagnosis)
  - [Architecture Detection](#architecture-detection)
  - [Weight Analysis](#weight-analysis)
  - [Quantization Suitability](#quantization-suitability)
  - [Pruning Potential](#pruning-potential)
  - [Graph Efficiency](#graph-efficiency)
  - [Health Score](#health-score)
  - [Recommendation Engine](#recommendation-engine)
- [Model Profiler](#model-profiler)
  - [Layer-Level Timing](#layer-level-timing)
  - [FLOPs & MACs Estimation](#flops--macs-estimation)
  - [Memory Analysis](#memory-analysis)
  - [Bottleneck Diagnosis](#bottleneck-diagnosis)
- [Model Inspector](#model-inspector)
  - [Graph Metadata](#graph-metadata)
  - [I/O Shapes](#io-shapes)
  - [EP Compatibility](#ep-compatibility)

---

## Model Diagnosis

Model Diagnosis performs a comprehensive analysis of your model's structure and generates actionable optimization recommendations.

### Architecture Detection

ssook automatically identifies the model architecture by matching the set of operations in the graph against known patterns:

| Architecture | Key Operations |
|-------------|---------------|
| **YOLO** | Conv, Concat, Resize, Sigmoid |
| **RF-DETR** | MultiHeadAttention, LayerNormalization, Conv, Gather |
| **EVA-02** | LayerNormalization, Attention, Gelu, MatMul |
| **Transformer** | LayerNormalization, Attention, MatMul, Softmax |
| **CNN** | Conv present, no attention ops |
| **Hybrid** | Both Conv and attention ops present |

The detection uses overlap scoring — if ≥50% of a pattern's ops are present, that architecture is identified. This helps tailor optimization recommendations to the model type.

### Weight Analysis

For each weight tensor in Conv, MatMul, Gemm, and ConvTranspose layers, the diagnosis computes:

- **Range** (min/max): How spread out the values are
- **Mean / Std**: The distribution center and spread
- **Sparsity**: Fraction of near-zero weights (|w| < 10⁻⁷) — indicates existing pruning or naturally sparse layers
- **Outlier ratio**: Fraction of weights beyond 3 standard deviations — high outlier ratios make quantization harder

### Quantization Suitability

The diagnosis evaluates how well the model will respond to quantization:

- **Quantizable ratio**: What percentage of operations can be quantized. Operations like Conv, MatMul, Gemm, Relu, MaxPool are quantizable. Operations like custom ops or certain normalization layers are not.
- **Per-node sensitivity**: Each quantizable layer gets a sensitivity score:
  ```
  sensitivity = max(|weights|) × (1 + outlier_ratio × 10)
  ```
  Layers with high sensitivity (wide weight range + many outliers) lose more accuracy when quantized.
- **Non-quantizable ops**: Lists operations that cannot be quantized, helping you decide between full INT8 and mixed precision.

### Pruning Potential

For each Conv layer, the diagnosis analyzes:

- **Channel importance**: The L1-norm of each output channel's weights. Channels with very low L1-norm contribute little to the output.
- **Low-importance channels**: Count of channels whose L1-norm is below 10% of the layer mean — these are candidates for channel pruning.
- **Overall sparsity**: The fraction of near-zero weights across all layers — indicates how much weight pruning has already been applied or could be applied.

### Graph Efficiency

- **Memory-bound op ratio**: Percentage of operations that are memory-bound (Reshape, Transpose, Concat, Gather, Slice, etc.) rather than compute-bound. High ratios suggest the model spends more time moving data than computing.
- **Fusable patterns**: Count of adjacent operation pairs that can be fused into single kernels (Conv+BN, Conv+Relu, MatMul+Add, Conv+Add+Relu). More fusable patterns = more room for graph optimization.

### Health Score

A single 0–100 score summarizing the model's optimization state:

```
health_score = 100 - (critical_findings × 20) - (warning_findings × 5)
```

- **100**: Model appears well-optimized
- **60–80**: Some optimization opportunities exist
- **< 60**: Significant optimization recommended

### Recommendation Engine

Based on the diagnosis, ssook generates a prioritized list of optimization recommendations:

| Priority | Method | When Recommended |
|----------|--------|-----------------|
| 0 | Dead Node Elimination | Always — clean the graph first |
| 1 | ORT Graph Optimizer | When fusable patterns are detected |
| 2 | Dynamic INT8 | When quantizable ratio > 50% |
| 2 | FP16 | When model > 10 MB |
| 2 | Weight Pruning | When sparsity < 50% and params > 100K |
| 2 | Channel Pruning | When low-importance channels > 5% |
| 3 | Static INT8 / Mixed Precision | When quantizable ratio > 50% (with/without non-quantizable ops) |

Each recommendation includes:
- **Reason**: Why this optimization is suggested
- **Expected impact**: Estimated speedup or size reduction
- **Executable**: Whether ssook can apply it directly (vs. requiring external tools like PyTorch)

**How to use:**
1. Go to **Tools** → **Model Optimization**
2. Click **Diagnose** after loading a model
3. Review the health score, findings, and recommendations
4. Click **Apply** on any recommendation to run it, or build a custom pipeline

---

## Model Profiler

The profiler measures actual runtime performance and estimates computational cost.

### Layer-Level Timing

**How it works:**
1. ONNX Runtime's built-in profiling is enabled via `SessionOptions.enable_profiling`.
2. The model runs inference with a dummy input matching the model's expected shape.
3. ORT writes a JSON trace file with per-node timing events.
4. ssook parses events with category `"Node"` and extracts the `dur` (duration in microseconds) for each operation.

The result is a ranked list of layers by execution time, showing exactly where the model spends its time.

### FLOPs & MACs Estimation

ssook estimates the computational cost of each layer:

| Operation | FLOPs Formula |
|-----------|--------------|
| **Conv** | 2 × C_out × C_in × K_h × K_w × H_out × W_out |
| **ConvTranspose** | 2 × C_in × C_out × K_h × K_w × H_out × W_out |
| **MatMul / Gemm** | 2 × M × K × N |
| **BatchNormalization** | 4 × elements (mean, variance, normalize, scale+shift) |

MACs (Multiply-Accumulate operations) = FLOPs / 2. These estimates use shape inference to determine output dimensions.

### Memory Analysis

- **Weight memory**: Total parameter count × 4 bytes (FP32). Shown in MB.
- **Peak activation memory**: Estimated as the largest intermediate tensor × 3 (rough estimate for concurrent tensors in memory).
- **Total memory**: Weight memory + peak activation memory.

### Bottleneck Diagnosis

Layers taking more than 100μs are categorized:

| Category | Operations | Meaning |
|----------|-----------|---------|
| **Compute** | Conv, MatMul, Gemm, ConvTranspose | CPU/GPU-bound — benefit from quantization or pruning |
| **Memory** | Reshape, Transpose, Concat, Gather, Slice | Memory-bound — benefit from graph optimization or fusion |
| **Other** | Everything else | May benefit from specialized optimizations |

Severity levels: **High** (>1000μs), **Medium** (>500μs), **Low** (>100μs).

The profiler also reports:
- **Quantizable ratio**: What fraction of ops can be quantized
- **Estimated INT8 speedup**: Rough estimate based on quantizable ratio (up to ~3× for fully quantizable models)
- **Graph depth**: Longest path from output to input (deeper graphs may have more sequential bottlenecks)

**How to use:**
1. Go to **Tools** → **Model Profiler**
2. Load your ONNX model
3. Set the number of runs (default: 20) and warmup iterations (default: 3)
4. Click **Profile**
5. Review latency statistics (avg, P50, P95, P99), top bottleneck layers, FLOPs breakdown, and optimization suggestions

---

## Model Inspector

The inspector provides a quick overview of the model's structure and compatibility.

### Graph Metadata

- **File size** (MB)
- **Opset version**: The ONNX operator set version. Higher versions support more operations. Quantization requires opset ≥ 11.
- **IR version**: The ONNX intermediate representation version.
- **Producer**: The framework that exported the model (e.g., "pytorch", "tf2onnx").
- **Number of nodes**: Total operations in the graph.
- **Operation counts**: Breakdown by operation type (e.g., Conv: 52, Relu: 48, BatchNormalization: 24).
- **Number of parameters**: Total learnable parameters across all initializers.

### I/O Shapes

For each input and output tensor:
- **Name**: The tensor name used in the graph
- **Shape**: Dimensions (may include dynamic dimensions like `"batch"` or `"height"`)
- **Data type**: e.g., `tensor(float)`, `tensor(float16)`, `tensor(int64)`

This is useful for understanding what the model expects as input and what it produces.

### EP Compatibility

ssook tests which Execution Providers can load the model and estimates how well each EP supports the model's operations:

- **Available EPs**: All EPs installed on the system
- **Compatible EPs**: EPs that can successfully load the model
- **Per-EP analysis**: For each compatible EP, ssook estimates:
  - **Supported ops**: Operations that run natively on the EP
  - **Fallback ops**: Operations that fall back to CPU
  - **Supported ratio**: Fraction of operations running on the EP (higher = better GPU utilization)

This uses a heuristic mapping of known EP op support (CUDA, TensorRT, OpenVINO, DirectML) to estimate compatibility without requiring actual GPU execution.

**How to use:**
1. Go to **Tools** → **Model Inspector**
2. Load your ONNX model
3. Review the graph metadata, I/O shapes, and EP compatibility report
