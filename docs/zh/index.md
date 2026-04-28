# ssook 文档

> **ssook** — 集 AI 模型推理、评估、分析与数据管理于一体的桌面工具箱

🌐 [English](../index.md) | [한국어](../ko/index.md) | [日本語](../ja/index.md) | **中文**

---

## 技术指南（含原理说明）

| 文档 | 主题 |
|------|------|
| [模型优化](model-optimization.md) | Quantization（量化）：Dynamic/Static INT8、FP16、Mixed Precision（混合精度），Pruning（剪枝）：Weight/Channel，Graph Optimization（图优化） |
| [模型分析](model-analysis.md) | Model Diagnosis（模型诊断）、Profiler（性能分析器）：FLOPs、延迟、内存，Inspector（检查器）：图信息、EP 兼容性 |
| [评估指标](evaluation-metrics.md) | mAP、IoU、Precision/Recall/F1、分割指标、Confidence Optimizer（置信度优化器）、FP/FN Error Analysis（误检/漏检分析） |
| [Embedding 与 CLIP](embedding-and-clip.md) | t-SNE / UMAP / PCA 可视化、CLIP Zero-Shot（零样本）分类、Embedder 评估 |
| [Execution Providers（执行提供程序）](execution-providers.md) | 自动 EP 选择、venv 隔离、CUDA/DirectML/OpenVINO/CoreML 支持 |
| [跟踪与采样](tracking-and-sampling.md) | ByteTrack / SORT 目标跟踪、Smart Sampler（智能采样器）、Near-Duplicate Detection（近似重复检测，dHash） |

## 功能参考

| 文档 | 主题 |
|------|------|
| [通用功能](general-features.md) | Viewer（查看器）、Dataset Explorer（数据集浏览器）、Splitter（拆分器）、Format Converter（格式转换器）、Class Remapper（类别重映射）、Merger（合并器）、质量工具、Benchmark（基准测试）、Settings（设置） |

## 关于

| 文档 | 说明 |
|------|------|
| [项目介绍](ssook-introduction.md) | 项目背景与概述 |
