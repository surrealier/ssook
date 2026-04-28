# 模型分析

[← 返回目录](../index.md) | 🌐 [English](../model-analysis.md) | [한국어](../ko/model-analysis.md) | [日本語](../ja/model-analysis.md) | **中文**

ssook 提供三个互补工具，帮助你在优化或部署 ONNX 模型之前深入了解它：**Model Diagnosis（模型诊断）**、**Model Profiler（模型性能分析器）** 和 **Model Inspector（模型检查器）**。

---

## 目录

- [Model Diagnosis（模型诊断）](#model-diagnosis模型诊断)
  - [架构检测](#架构检测)
  - [权重分析](#权重分析)
  - [量化适用性](#量化适用性)
  - [剪枝潜力](#剪枝潜力)
  - [图效率](#图效率)
  - [健康评分](#健康评分)
  - [推荐引擎](#推荐引擎)
- [Model Profiler（模型性能分析器）](#model-profiler模型性能分析器)
  - [逐层计时](#逐层计时)
  - [FLOPs 与 MACs 估算](#flops-与-macs-估算)
  - [内存分析](#内存分析)
  - [瓶颈诊断](#瓶颈诊断)
- [Model Inspector（模型检查器）](#model-inspector模型检查器)
  - [图元数据](#图元数据)
  - [输入/输出形状](#输入输出形状)
  - [EP 兼容性](#ep-兼容性)

---

## Model Diagnosis（模型诊断）

模型诊断对模型结构进行全面分析，并生成可操作的优化建议。

### 架构检测

ssook 通过将图中的操作集合与已知模式进行匹配，自动识别模型架构：

| 架构 | 关键操作 |
|------|----------|
| **YOLO** | Conv, Concat, Resize, Sigmoid |
| **RF-DETR** | MultiHeadAttention, LayerNormalization, Conv, Gather |
| **EVA-02** | LayerNormalization, Attention, Gelu, MatMul |
| **Transformer** | LayerNormalization, Attention, MatMul, Softmax |
| **CNN** | 存在 Conv，无 Attention 操作 |
| **Hybrid（混合）** | 同时存在 Conv 和 Attention 操作 |

检测使用重叠评分——如果某模式的操作中有 ≥50% 存在于图中，则识别为该架构。这有助于针对模型类型定制优化建议。

### 权重分析

对 Conv、MatMul、Gemm 和 ConvTranspose 层中的每个权重张量，诊断会计算：

- **范围**（最小值/最大值）：数值的分布跨度
- **均值 / 标准差**：分布的中心和离散程度
- **稀疏度**：接近零的权重比例（|w| < 10⁻⁷）——表示已有的剪枝或天然稀疏层
- **异常值比例**：超出 3 个标准差的权重比例——异常值比例高会增加量化难度

### 量化适用性

诊断评估模型对量化的响应程度：

- **可量化比例**：可以量化的操作占总操作的百分比。Conv、MatMul、Gemm、Relu、MaxPool 等操作可量化；自定义操作或某些归一化层不可量化。
- **逐节点敏感度**：每个可量化层获得一个敏感度分数：
  ```
  sensitivity = max(|weights|) × (1 + outlier_ratio × 10)
  ```
  敏感度高的层（权重范围大 + 异常值多）在量化时精度损失更大。
- **不可量化操作**：列出无法量化的操作，帮助你在全 INT8 和混合精度之间做出选择。

### 剪枝潜力

对每个 Conv 层，诊断分析：

- **通道重要性**：每个输出通道权重的 L1 范数。L1 范数很低的通道对输出贡献很小。
- **低重要性通道**：L1 范数低于该层均值 10% 的通道数——这些是通道剪枝的候选对象。
- **整体稀疏度**：所有层中接近零的权重比例——表示已应用或可应用的权重剪枝程度。

### 图效率

- **内存受限操作比例**：内存受限操作（Reshape、Transpose、Concat、Gather、Slice 等）占总操作的百分比，而非计算受限操作。比例高说明模型在数据搬运上花费的时间多于计算。
- **可融合模式**：可融合为单个内核的相邻操作对数量（Conv+BN、Conv+Relu、MatMul+Add、Conv+Add+Relu）。可融合模式越多 = 图优化空间越大。

### 健康评分

一个 0–100 的综合评分，概括模型的优化状态：

```
health_score = 100 - (critical_findings × 20) - (warning_findings × 5)
```

- **100**：模型已充分优化
- **60–80**：存在一些优化机会
- **< 60**：建议进行重大优化

### 推荐引擎

基于诊断结果，ssook 生成按优先级排序的优化建议：

| 优先级 | 方法 | 推荐条件 |
|--------|------|----------|
| 0 | Dead Node Elimination（死节点消除） | 始终推荐——先清理图 |
| 1 | ORT Graph Optimizer | 检测到可融合模式时 |
| 2 | Dynamic INT8 | 可量化比例 > 50% 时 |
| 2 | FP16 | 模型 > 10 MB 时 |
| 2 | Weight Pruning（权重剪枝） | 稀疏度 < 50% 且参数 > 100K 时 |
| 2 | Channel Pruning（通道剪枝） | 低重要性通道 > 5% 时 |
| 3 | Static INT8 / Mixed Precision | 可量化比例 > 50% 时（有/无不可量化操作） |

每条建议包含：
- **原因**：为什么建议此优化
- **预期效果**：预估的加速或体积缩减
- **可执行性**：ssook 是否可以直接应用（还是需要 PyTorch 等外部工具）

**使用方法：**
1. 进入 **Tools** → **Model Optimization**
2. 加载模型后点击 **Diagnose**
3. 查看健康评分、发现项和建议
4. 点击任意建议的 **Apply** 执行，或构建自定义流水线

---

## Model Profiler（模型性能分析器）

性能分析器测量实际运行时性能并估算计算开销。

### 逐层计时

**工作原理：**
1. 通过 `SessionOptions.enable_profiling` 启用 ONNX Runtime 内置的性能分析。
2. 使用与模型预期形状匹配的虚拟输入运行推理。
3. ORT 输出包含逐节点计时事件的 JSON 跟踪文件。
4. ssook 解析类别为 `"Node"` 的事件，提取每个操作的 `dur`（持续时间，微秒）。

结果是按执行时间排序的层列表，精确显示模型在哪里花费了时间。

### FLOPs 与 MACs 估算

ssook 估算每层的计算开销：

| 操作 | FLOPs 公式 |
|------|-----------|
| **Conv** | 2 × C_out × C_in × K_h × K_w × H_out × W_out |
| **ConvTranspose** | 2 × C_in × C_out × K_h × K_w × H_out × W_out |
| **MatMul / Gemm** | 2 × M × K × N |
| **BatchNormalization** | 4 × elements（均值、方差、归一化、缩放+偏移） |

MACs（乘累加操作）= FLOPs / 2。这些估算使用形状推断来确定输出维度。

### 内存分析

- **权重内存**：总参数量 × 4 字节（FP32），以 MB 显示。
- **峰值激活内存**：估算为最大中间张量 × 3（对内存中并发张量的粗略估计）。
- **总内存**：权重内存 + 峰值激活内存。

### 瓶颈诊断

耗时超过 100μs 的层按以下类别分类：

| 类别 | 操作 | 含义 |
|------|------|------|
| **计算受限** | Conv, MatMul, Gemm, ConvTranspose | CPU/GPU 受限——可通过量化或剪枝优化 |
| **内存受限** | Reshape, Transpose, Concat, Gather, Slice | 内存受限——可通过图优化或融合优化 |
| **其他** | 其余所有操作 | 可能受益于专门的优化 |

严重程度：**High**（>1000μs）、**Medium**（>500μs）、**Low**（>100μs）。

性能分析器还报告：
- **可量化比例**：可量化操作的占比
- **预估 INT8 加速**：基于可量化比例的粗略估计（完全可量化模型最高约 3 倍）
- **图深度**：从输出到输入的最长路径（更深的图可能有更多串行瓶颈）

**使用方法：**
1. 进入 **Tools** → **Model Profiler**
2. 加载 ONNX 模型
3. 设置运行次数（默认：20）和预热迭代次数（默认：3）
4. 点击 **Profile**
5. 查看延迟统计（avg、P50、P95、P99）、主要瓶颈层、FLOPs 分布和优化建议

---

## Model Inspector（模型检查器）

检查器提供模型结构和兼容性的快速概览。

### 图元数据

- **文件大小**（MB）
- **Opset 版本**：ONNX 算子集版本。更高版本支持更多操作。量化要求 opset ≥ 11。
- **IR 版本**：ONNX 中间表示版本。
- **Producer（生产者）**：导出模型的框架（如 "pytorch"、"tf2onnx"）。
- **节点数**：图中的总操作数。
- **操作统计**：按操作类型分类（如 Conv: 52, Relu: 48, BatchNormalization: 24）。
- **参数数量**：所有初始化器中的可学习参数总数。

### 输入/输出形状

对每个输入和输出张量：
- **名称**：图中使用的张量名称
- **形状**：维度（可能包含动态维度，如 `"batch"` 或 `"height"`）
- **数据类型**：如 `tensor(float)`、`tensor(float16)`、`tensor(int64)`

这有助于了解模型期望的输入和产生的输出。

### EP 兼容性

ssook 测试哪些 Execution Provider 可以加载模型，并估算每个 EP 对模型操作的支持程度：

- **可用 EP**：系统上安装的所有 EP
- **兼容 EP**：能成功加载模型的 EP
- **逐 EP 分析**：对每个兼容的 EP，ssook 估算：
  - **支持的操作**：在该 EP 上原生运行的操作
  - **回退操作**：回退到 CPU 的操作
  - **支持比例**：在该 EP 上运行的操作占比（越高 = GPU 利用率越好）

这使用已知 EP 操作支持（CUDA、TensorRT、OpenVINO、DirectML）的启发式映射来估算兼容性，无需实际 GPU 执行。

**使用方法：**
1. 进入 **Tools** → **Model Inspector**
2. 加载 ONNX 模型
3. 查看图元数据、输入/输出形状和 EP 兼容性报告
