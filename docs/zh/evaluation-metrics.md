# 评估指标

[← 返回目录](../index.md) | 🌐 [English](../evaluation-metrics.md) | [한국어](../ko/evaluation-metrics.md) | [日本語](../ja/evaluation-metrics.md) | **中文**

ssook 支持五种任务类型的模型评估——Detection（检测）、Classification（分类）、Segmentation（分割）、CLIP 和 Embedder——每种都有各自的指标体系。本文档介绍每个指标的计算方式及结果解读方法。

---

## 目录

- [Detection Metrics（检测指标）](#detection-metrics检测指标)
  - [IoU（交并比）](#iou交并比)
  - [Precision、Recall、F1](#precisionrecallf1)
  - [mAP@50](#map50)
  - [mAP@50:95](#map5095)
- [Segmentation Metrics（分割指标）](#segmentation-metrics分割指标)
- [Classification Metrics（分类指标）](#classification-metrics分类指标)
- [Embedder Metrics（嵌入器指标）](#embedder-metrics嵌入器指标)
- [Confidence Optimizer（置信度优化器）](#confidence-optimizer置信度优化器)
- [FP/FN Error Analysis（误检/漏检分析）](#fpfn-error-analysis误检漏检分析)

---

## Detection Metrics（检测指标）

### IoU（交并比）

IoU（Intersection over Union）衡量预测边界框与真实标注框的重叠程度：

```
IoU = 交集面积 / 并集面积
```

- **IoU = 1.0**：完美重叠
- **IoU = 0.5**：标准阈值——IoU ≥ 0.5 的预测被视为正确检测（True Positive，真正例）
- **IoU = 0.0**：完全无重叠

ssook 使用**向量化矩阵运算**计算 IoU：给定 M 个预测和 N 个真实标注，同时计算所有 M×N 个 IoU 值以提高效率。

### Precision、Recall、F1

使用 IoU 将预测与真实标注匹配后：

- **True Positive（TP，真正例）**：预测匹配到真实标注（IoU ≥ 阈值，类别正确）
- **False Positive（FP，误检）**：预测未匹配到任何真实标注
- **False Negative（FN，漏检）**：真实标注未被任何预测匹配

```
Precision = TP / (TP + FP)    — "所有检测中，有多少是正确的？"
Recall    = TP / (TP + FN)    — "所有真实目标中，有多少被检测到？"
F1        = 2 × P × R / (P + R) — Precision 和 Recall 的调和平均值
```

### mAP@50

IoU 阈值为 0.5 时的平均精度均值（Mean Average Precision）：

1. 对每个类别，按置信度从高到低排序所有预测。
2. 依次遍历预测。对每个预测，检查是否匹配到 IoU ≥ 0.5 的未匹配真实标注。
3. 在每一步计算累积 Precision 和 Recall。
4. 使用 **101 点插值**计算 **AP**（Average Precision）：在 101 个等间距的 Recall 点（0.00, 0.01, ..., 1.00）上采样 Precision-Recall 曲线，取插值 Precision 的平均值。
5. **mAP@50** = 所有类别 AP 的均值。

### mAP@50:95

更严格的指标，在 10 个 IoU 阈值上取 mAP 的平均值：0.50, 0.55, 0.60, ..., 0.95。

ssook 通过一次计算 IoU 矩阵并在所有阈值上复用来优化此过程，而非运行 10 次独立评估。

更高的 IoU 阈值要求更精确的定位，因此 mAP@50:95 始终低于 mAP@50。

---

## Segmentation Metrics（分割指标）

用于语义分割模型（输出：C×H×W 类别概率图）：

### IoU（逐类别）

```
IoU = |Pred ∩ GT| / |Pred ∪ GT|
```

逐像素计算：统计预测和真实标注在该类别上一致的像素数（交集），除以任一方预测为该类别的像素数（并集）。

### Dice Coefficient（Dice 系数，逐类别）

```
Dice = 2 × |Pred ∩ GT| / (|Pred| + |GT|)
```

与 IoU 类似，但对交集赋予更高权重。对于相同的预测，Dice 始终 ≥ IoU。

### mIoU / mDice

所有具有真实标注像素的类别的 IoU 和 Dice 均值。没有真实标注像素的类别不参与均值计算。

**使用方法：**
1. 进入 **Evaluation** 标签页
2. 选择任务类型：**Segmentation**
3. 加载分割模型和真实标注掩码
4. 运行评估——显示逐类别 IoU/Dice 和整体 mIoU/mDice

---

## Classification Metrics（分类指标）

用于分类模型（输出：类别概率）：

- **Accuracy（准确率）**：正确分类的图像比例
- **逐类别 Precision/Recall/F1**：对每个类别使用 one-vs-rest 方法计算
- **整体 P/R/F1**：所有类别的宏平均（Macro-average）

**使用方法：**
1. 进入 **Evaluation** 标签页
2. 选择任务类型：**Classification**
3. 加载分类模型和真实标注标签
4. 运行评估

---

## Embedder Metrics（嵌入器指标）

用于特征提取模型（输出：嵌入向量）：

### Cosine Similarity（余弦相似度）

```
similarity = (A · B) / (‖A‖ × ‖B‖)
```

嵌入向量在比较前进行 L2 归一化，因此余弦相似度等于点积。范围：-1（完全相反）到 1（完全相同）。

### Retrieval@1

对每张查询图像，通过余弦相似度找到最相似的图库图像。Retrieval@1 = 排名第一的结果类别正确的查询比例。

### Retrieval@K

与 Retrieval@1 相同，但只要正确类别出现在前 K 个结果中即视为正确。

**使用方法：**
1. 进入 **Evaluation** 标签页
2. 选择任务类型：**Embedder**
3. 加载特征提取 ONNX 模型
4. 设置数据集目录（文件夹结构：`class_name/image.jpg`）
5. 运行评估——显示 Retrieval@1、Retrieval@K 和平均余弦相似度

---

## Confidence Optimizer（置信度优化器）

置信度优化器通过遍历阈值并测量 F1，为每个类别找到最优置信度阈值。

**工作原理：**
1. 对所有评估图像运行推理，获取带置信度分数的预测结果。
2. 对每个类别，从 0.0 到 1.0 遍历置信度阈值。
3. 在每个阈值处计算 Precision、Recall 和 F1。
4. 为每个类别绘制 **PR 曲线**（Precision vs. Recall）。
5. 找到使每个类别 F1 最大化的阈值。

**为什么需要逐类别阈值：**
不同类别可能有不同的最优阈值。样本量大的"行人"类别可能在 0.3 时效果最好，而稀有的"轮椅"类别可能需要 0.5 才能避免误检。

**使用方法：**
1. 进入 **Analysis** 标签页 → **Confidence Optimizer**
2. 加载模型和评估数据集
3. 点击 **Run**——显示每个类别的 PR 曲线和最优阈值
4. 每个类别的 F1 最大化阈值会被高亮显示

---

## FP/FN Error Analysis（误检/漏检分析）

错误分析器自动对检测错误进行分类，帮助你了解模型在哪里以及为什么失败。

**错误分类：**

| 错误类型 | 定义 |
|----------|------|
| **False Positive（FP，误检）** | 模型在真实标注中不存在目标的位置检测到了目标 |
| **False Negative（FN，漏检）** | 真实标注中的目标未被模型检测到 |

**按尺寸分类：**

错误按边界框面积分类：

| 尺寸 | 面积范围 | 典型目标 |
|------|----------|----------|
| **Small（S，小）** | < 32² 像素 | 远处行人、小型标志 |
| **Medium（M，中）** | 32²–96² 像素 | 近处行人、车辆 |
| **Large（L，大）** | > 96² 像素 | 特写目标、大型车辆 |

**按位置分析：**

错误还按图像中的位置（中心 vs. 边缘）进行分析，帮助识别模型是否在特定位置的目标上表现不佳。

**使用方法：**
1. 进入 **Analysis** 标签页 → **FP/FN Error Analysis**
2. 加载模型和带真实标注的评估数据集
3. 点击 **Run**
4. 查看按类型、尺寸和位置分类的错误分布
5. 利用这些洞察指导数据收集（例如，如果小目标的漏检率高，则收集更多小目标训练数据）
