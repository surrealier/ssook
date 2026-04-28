# 模型优化

[← 返回目录](../index.md) | 🌐 [English](../model-optimization.md) | [한국어](../ko/model-optimization.md) | [日本語](../ja/model-optimization.md) | **中文**

ssook 提供完整的优化流水线，可在无需编写代码的情况下缩减模型体积、降低延迟并减少内存占用。本文档介绍每种技术的原理及在 ssook 中的使用方法。

---

## 目录

- [Quantization（量化）](#quantization量化)
  - [Dynamic INT8（动态 INT8）](#dynamic-int8动态-int8)
  - [Static INT8（静态 INT8）](#static-int8静态-int8)
  - [FP16 转换](#fp16-转换)
  - [Mixed Precision（混合精度）](#mixed-precision混合精度)
- [Pruning（剪枝）](#pruning剪枝)
  - [Weight Pruning（权重剪枝）](#weight-pruning权重剪枝)
  - [Channel Pruning（通道剪枝）](#channel-pruning通道剪枝)
- [Graph Optimization（图优化）](#graph-optimization图优化)
  - [ORT Graph Optimizer](#ort-graph-optimizer)
  - [ONNX Simplifier](#onnx-simplifier)
  - [Dead Node Elimination（死节点消除）](#dead-node-elimination死节点消除)
- [优化流水线](#优化流水线)
- [性能对比](#性能对比)
- [提示与注意事项](#提示与注意事项)

---

## Quantization（量化）

量化将模型权重（以及可选的激活值）从高精度浮点数（FP32）转换为低精度格式（INT8、FP16），从而缩减模型体积并加速推理，在 CPU 上效果尤为显著。

### Dynamic INT8（动态 INT8）

**工作原理：**
- 权重在**导出时**从 FP32 转换为 INT8。
- 激活值保持 FP32，在推理过程中**动态**量化。
- 无需校准数据——权重的量化范围直接根据权重值计算。

**优缺点：**
- ✅ 最简单的方法——选择模型后直接运行即可
- ✅ 模型体积缩减约 50%
- ✅ CPU 上加速 1.5–2 倍
- ⚠️ 由于激活值范围在运行时估算，效果不如静态量化

**在 ssook 中使用：**
1. 进入 **Tools** 标签页 → **Calibration / Quantization**
2. 加载 ONNX 模型
3. 选择 **Dynamic INT8**
4. 选择权重类型（UInt8 或 Int8）
5. 点击 **Run**

### Static INT8（静态 INT8）

**工作原理：**
- 权重和激活值均转换为 INT8。
- 使用**校准数据集**（50–200 张代表性图像）在推理过程中测量实际激活值范围。
- ssook 自动对校准图像进行预处理，使其匹配模型的输入形状（缩放、归一化、批处理）。
- QDQ（Quantize-Dequantize）格式在可量化操作周围插入 Q/DQ 节点，保留图结构以获得最大兼容性。

**优缺点：**
- ✅ 最佳加速效果：CPU 上 2–4 倍
- ✅ 模型体积缩减约 75%
- ✅ 当校准数据具有代表性时，精度损失极小
- ⚠️ 需要来自目标领域的校准图像

**在 ssook 中使用：**
1. 进入 **Tools** → **Calibration / Quantization**
2. 加载 ONNX 模型
3. 选择 **Static INT8**
4. 设置校准图像目录和最大图像数
5. 配置选项（per-channel、QDQ 格式）
6. 点击 **Run**——进度条显示校准进度

### FP16 转换

**工作原理：**
- 所有 FP32 权重和常量转换为 FP16（半精度浮点数）。
- FP16 使用 16 位而非 32 位，每个值的存储空间减半。
- 可表示范围较小（±65,504 vs ±3.4×10³⁸），但对大多数神经网络权重来说已经足够。

**优缺点：**
- ✅ 模型体积缩减约 50%
- ✅ 对大多数模型几乎无精度损失
- ✅ 在支持 FP16 硬件的 GPU 上显著加速（NVIDIA Tensor Cores、Apple Neural Engine）
- ⚠️ 在 CPU 上加速效果有限（CPU 对 FP16 的支持因平台而异）
- ⚠️ 需要 `onnxconverter-common` 包

**在 ssook 中使用：**
1. 进入 **Tools** → **Calibration / Quantization**
2. 加载 ONNX 模型
3. 选择 **FP16**
4. 点击 **Run**

### Mixed Precision（混合精度）

**工作原理：**
- 并非所有层对量化的响应都相同。权重范围较大或异常值较多的层在量化时精度损失更大。
- ssook 为每个可量化层计算**敏感度分数**：
  ```
  sensitivity = weight_range × (1 + outlier_ratio × 10)
  ```
  其中 `outlier_ratio` 是超出 3 个标准差的权重比例。
- 敏感度最高的前 N% 层**不进行** INT8 量化，保持 FP32。
- 其余层正常量化为 INT8。

**优缺点：**
- ✅ 精度优于全 INT8——敏感层保持 FP32
- ✅ 仍可实现 2–3 倍加速
- ✅ 全自动——无需手动选择层
- ⚠️ 压缩率略低于全 INT8

**在 ssook 中使用：**
1. 进入 **Tools** → **Calibration / Quantization**
2. 加载 ONNX 模型
3. 选择 **Mixed Precision**
4. 设置排除百分比（默认：排除敏感度最高的 20% 层）
5. 点击 **Run**

---

## Pruning（剪枝）

剪枝通过移除不必要的权重或通道来缩减模型体积和计算开销。

### Weight Pruning（权重剪枝）

**工作原理：**
- **基于幅值的非结构化剪枝**：绝对值最小的权重对输出贡献最小，可以置零。
- 给定 `sparsity_ratio`（如 0.3），将幅值最小的 30% 权重置零。
- 目标操作包括 Conv、MatMul、Gemm 和 ConvTranspose。
- 生成的稀疏模型在压缩后体积更小，在支持稀疏计算的运行时上可能更快。

**优缺点：**
- ✅ 体积缩减 10–30%（压缩后更多）
- ✅ 中等稀疏度（≤30%）下精度损失极小
- ⚠️ 加速效果取决于运行时的稀疏支持——标准 ONNX Runtime 不加速稀疏操作
- ⚠️ 高稀疏度（>50%）可能明显降低精度

**在 ssook 中使用：**
1. 进入 **Tools** → **Model Optimization**
2. 加载 ONNX 模型
3. 选择 **Weight Pruning**
4. 设置稀疏率（0.0–1.0，推荐：0.2–0.3）
5. 点击 **Run**

### Channel Pruning（通道剪枝）

**工作原理：**
- **基于 L1 范数的结构化剪枝**：对每个 Conv 层，计算每个输出通道权重的 L1 范数（绝对值之和）。
- L1 范数最低的通道对该层输出贡献最小。
- 移除 L1 范数最低的 N% 通道——包括权重和对应的偏置值。
- 与权重剪枝不同，这是**结构化**的：生成的模型通道数更少，在任何硬件上都能更快运行，无需特殊的稀疏支持。

**优缺点：**
- ✅ FLOPs 直接按剪枝通道比例减少
- ✅ 在所有硬件上都更快（无需稀疏运行时）
- ⚠️ 可能需要微调以恢复精度
- ⚠️ 设有最小通道数限制，防止层被剪枝至零

**在 ssook 中使用：**
1. 进入 **Tools** → **Model Optimization**
2. 加载 ONNX 模型
3. 选择 **Channel Pruning**
4. 设置剪枝率（推荐：0.1–0.3）和最小通道数（默认：4）
5. 点击 **Run**

---

## Graph Optimization（图优化）

图优化通过重构计算图来消除冗余并融合操作，不改变模型的数学行为。

### ORT Graph Optimizer

**工作原理：**
- 使用 ONNX Runtime 内置的图优化 Pass。
- **常量折叠（Constant Folding）**：预计算所有输入均为常量的操作（如 Shape → Gather → Unsqueeze 链）。
- **算子融合（Operator Fusion）**：将相邻操作合并为单个优化内核：
  - Conv + BatchNormalization → 单个 Conv（BN 参数折叠到 Conv 权重中）
  - Conv + Relu → ConvRelu 融合内核
  - MatMul + Add → 融合 Gemm
  - Conv + Add + Relu → 单个融合内核
- 三个优化级别：**Basic**（常量折叠）、**Extended**（+ 公共子表达式消除）、**All**（+ 高级融合）。

**优缺点：**
- ✅ 延迟降低 5–15%
- ✅ 零精度损失——数学上等价
- ✅ 始终可以安全应用

### ONNX Simplifier

**工作原理：**
- 使用 `onnxsim` 库在 ONNX 图级别执行形状推断和常量折叠。
- 尽可能解析动态形状，简化冗余操作，生成更简洁的图。

**优缺点：**
- ✅ 图更简洁，文件通常更小
- ✅ 零精度损失
- ⚠️ 需要 `onnxsim` 包

### Dead Node Elimination（死节点消除）

**工作原理：**
- 从输出节点反向追踪，找到所有可达节点。
- 移除所有从输出不可达的节点。
- 清理模型导出或之前变换遗留的多余节点。

**优缺点：**
- ✅ 缩减图规模
- ✅ 零精度损失
- ✅ 适合在其他优化之前运行

**在 ssook 中使用图优化：**
1. 进入 **Tools** → **Model Optimization**
2. 加载 ONNX 模型
3. 选择优化方法
4. 对于 ORT Graph Optimizer，选择级别（Basic / Extended / All）
5. 点击 **Run**

---

## 优化流水线

ssook 允许**串联多个优化步骤**。例如：

1. Dead Node Elimination → 清理图
2. ORT Graph Optimizer → 融合操作
3. Channel Pruning → 移除不重要的通道
4. Dynamic INT8 Quantization → 压缩权重

每个步骤以上一步的输出作为输入。流水线会跟踪每个阶段的体积变化和结果。

**使用方法：**
1. 进入 **Tools** → **Model Optimization**
2. 使用 **Model Diagnosis** 功能获取推荐的优化步骤
3. 诊断引擎根据模型特征建议优先级排序的流水线
4. 应用推荐的流水线，或自定义步骤

---

## 性能对比

CPU 上的典型结果（Intel i7-1165G7）：

| 模型 | 方法 | 体积 | 推理时间 | mAP@50 |
|------|------|------|----------|--------|
| YOLOv8n FP32 | 原始 | 12.2 MB | ~45 ms | 37.3 |
| YOLOv8n INT8（Dynamic） | 动态量化 | 3.4 MB | ~25 ms | 36.8 |
| YOLOv8n INT8（Static） | 静态量化 | 3.2 MB | ~18 ms | 37.0 |
| YOLOv8n FP16 | FP16 转换 | 6.1 MB | ~38 ms | 37.3 |

> 结果因硬件和模型而异。优化后请务必使用 ssook 的 Evaluation 标签页验证精度。

---

## 提示与注意事项

1. **从图优化开始**——零精度损失，且能使后续优化更有效。
2. **先使用 Model Diagnosis**——它会分析模型并推荐最佳优化策略。
3. **量化后验证精度**——使用 Evaluation 标签页对比优化前后的 mAP。
4. **校准数据很重要**——对于 Static INT8，请使用来自实际部署领域的图像（通常 50–200 张即可）。
5. **检查 opset 版本**——量化要求 opset ≥ 11。低版本 opset 的模型可能会失败。
6. **CenterNet / DETR 模型**——某些自定义操作可能不支持 INT8。请改用 FP16 或 Mixed Precision。
7. **通道剪枝需要微调**——为获得最佳效果，请在剪枝后使用训练数据对模型进行微调。
