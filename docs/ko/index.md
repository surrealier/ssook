# ssook 문서

> **ssook** — AI 모델 추론, 평가, 분석 및 데이터 관리를 위한 올인원 데스크톱 툴킷

🌐 [English](../index.md) | **한국어** | [日本語](../ja/index.md) | [中文](../zh/index.md)

---

## 기술 가이드 (원리 포함)

| 문서 | 주제 |
|------|------|
| [모델 최적화](model-optimization.md) | Quantization (양자화) — Dynamic/Static INT8, FP16, Mixed Precision, Pruning (가지치기) — Weight/Channel, Graph Optimization (그래프 최적화) |
| [모델 분석](model-analysis.md) | Model Diagnosis (모델 진단), Profiler (FLOPs, 지연시간, 메모리), Inspector (그래프 정보, EP 호환성) |
| [평가 지표](evaluation-metrics.md) | mAP, IoU, Precision/Recall/F1, Segmentation 지표, Confidence Optimizer, FP/FN 오류 분석 |
| [Embedding & CLIP](embedding-and-clip.md) | t-SNE / UMAP / PCA 시각화, CLIP Zero-Shot 분류, Embedder 평가 |
| [Execution Providers](execution-providers.md) | 자동 EP 선택, venv 격리, CUDA/DirectML/OpenVINO/CoreML 지원 |
| [트래킹 & 샘플링](tracking-and-sampling.md) | ByteTrack / SORT 객체 추적, Smart Sampler, Near-Duplicate Detection (유사 중복 탐지, dHash) |

## 기능 레퍼런스

| 문서 | 주제 |
|------|------|
| [일반 기능](general-features.md) | Viewer, Dataset Explorer, Splitter, Format Converter, Class Remapper, Merger, 품질 도구, Benchmark, 설정 |

## 소개

| 문서 | 설명 |
|------|------|
| [프로젝트 소개](../ssook-introduction.md) | 프로젝트 배경 및 개요 |
