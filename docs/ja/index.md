# ssook ドキュメント

> **ssook** — AIモデルの推論、評価、分析、データ管理を一つにまとめたデスクトップツールキット

🌐 [English](../index.md) | [한국어](../ko/index.md) | **日本語** | [中文](../zh/index.md)

---

## テクニカルガイド（原理解説付き）

| ドキュメント | トピック |
|----------|--------|
| [Model Optimization（モデル最適化）](model-optimization.md) | Quantization（量子化）（Dynamic/Static INT8、FP16、Mixed Precision）、Pruning（枝刈り）（Weight/Channel）、Graph Optimization（グラフ最適化） |
| [Model Analysis（モデル分析）](model-analysis.md) | Model Diagnosis（モデル診断）、Profiler（FLOPs、レイテンシ、メモリ）、Inspector（グラフ情報、EP互換性） |
| [Evaluation Metrics（評価指標）](evaluation-metrics.md) | mAP、IoU、Precision/Recall/F1、Segmentation指標、Confidence Optimizer、FP/FNエラー分析 |
| [Embedding & CLIP](embedding-and-clip.md) | t-SNE / UMAP / PCA可視化、CLIP Zero-Shot分類、Embedder評価 |
| [Execution Providers（実行プロバイダ）](execution-providers.md) | 自動EP選択、venv分離、CUDA/DirectML/OpenVINO/CoreMLサポート |
| [Tracking & Sampling（トラッキング＆サンプリング）](tracking-and-sampling.md) | ByteTrack / SORTオブジェクトトラッキング、Smart Sampler、Near-Duplicate Detection（dHash） |

## 機能リファレンス

| ドキュメント | トピック |
|----------|--------|
| [General Features（一般機能）](general-features.md) | Viewer、Dataset Explorer、Splitter、Format Converter、Class Remapper、Merger、品質ツール、Benchmark、Settings |

## About

| ドキュメント | 説明 |
|----------|-------------|
| [Introduction（はじめに）](ssook-introduction.md) | プロジェクトの背景と概要 |
