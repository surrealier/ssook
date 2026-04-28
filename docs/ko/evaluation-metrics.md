# 평가 지표

[← 목차로 돌아가기](../index.md) | 🌐 [English](../evaluation-metrics.md) | **한국어** | [日本語](../ja/evaluation-metrics.md) | [中文](../zh/evaluation-metrics.md)

ssook은 Detection (객체 탐지), Classification (분류), Segmentation (분할), CLIP, Embedder의 다섯 가지 태스크 유형에 대해 각각 고유한 지표로 모델을 평가합니다. 이 문서에서는 각 지표의 계산 방법과 결과 해석 방법을 설명합니다.

---

## 목차

- [Detection 지표](#detection-지표)
  - [IoU (Intersection over Union)](#iou-intersection-over-union)
  - [Precision, Recall, F1](#precision-recall-f1)
  - [mAP@50](#map50)
  - [mAP@50:95](#map5095)
- [Segmentation 지표](#segmentation-지표)
- [Classification 지표](#classification-지표)
- [Embedder 지표](#embedder-지표)
- [Confidence Optimizer](#confidence-optimizer)
- [FP/FN 오류 분석](#fpfn-오류-분석)

---

## Detection 지표

### IoU (Intersection over Union)

IoU는 예측 바운딩 박스가 정답(Ground Truth) 박스와 얼마나 겹치는지를 측정합니다:

```
IoU = 겹치는 영역 / 합집합 영역
```

- **IoU = 1.0**: 완벽한 겹침
- **IoU = 0.5**: 표준 임계값 — IoU ≥ 0.5인 예측은 올바른 탐지(True Positive)로 간주됩니다
- **IoU = 0.0**: 전혀 겹치지 않음

ssook은 **벡터화된 행렬 연산**으로 IoU를 계산합니다: M개의 예측과 N개의 정답이 주어지면, 효율성을 위해 M×N개의 IoU 값을 동시에 계산합니다.

### Precision, Recall, F1

IoU를 사용하여 예측과 정답을 매칭한 후:

- **True Positive (TP)**: 예측이 정답과 매칭됨 (IoU ≥ 임계값, 올바른 클래스)
- **False Positive (FP)**: 예측이 어떤 정답과도 매칭되지 않음
- **False Negative (FN)**: 정답이 어떤 예측과도 매칭되지 않음

```
Precision = TP / (TP + FP)    — "전체 탐지 중 올바른 것의 비율"
Recall    = TP / (TP + FN)    — "전체 정답 중 찾아낸 것의 비율"
F1        = 2 × P × R / (P + R) — Precision과 Recall의 조화 평균
```

### mAP@50

IoU 임계값 0.5에서의 Mean Average Precision:

1. 각 클래스에 대해 모든 예측을 신뢰도(confidence) 순으로 정렬합니다 (높은 순).
2. 예측을 순서대로 확인합니다. 각 예측에 대해 IoU ≥ 0.5인 미매칭 정답이 있는지 확인합니다.
3. 각 단계에서 누적 Precision과 Recall을 계산합니다.
4. **101-point interpolation (101점 보간법)**을 사용하여 **AP** (Average Precision)를 계산합니다: Precision-Recall 곡선을 101개의 균등 간격 Recall 지점(0.00, 0.01, ..., 1.00)에서 샘플링하고 보간된 Precision 값의 평균을 구합니다.
5. **mAP@50** = 모든 클래스에 대한 AP의 평균.

### mAP@50:95

10개의 IoU 임계값(0.50, 0.55, 0.60, ..., 0.95)에 걸쳐 mAP를 평균하는 더 엄격한 지표입니다.

ssook은 IoU 행렬을 한 번만 계산하고 모든 임계값에 재사용하여 최적화합니다. 10번의 별도 평가를 실행하지 않습니다.

높은 IoU 임계값은 더 정밀한 위치 지정을 요구하므로, mAP@50:95는 항상 mAP@50보다 낮습니다.

---

## Segmentation 지표

시맨틱 세그멘테이션 모델 (출력: C×H×W 클래스 확률 맵)의 경우:

### IoU (클래스별)

```
IoU = |Pred ∩ GT| / |Pred ∪ GT|
```

픽셀 단위로 계산합니다: 예측과 정답이 모두 해당 클래스로 일치하는 픽셀 수(교집합)를 어느 쪽이든 해당 클래스로 예측한 픽셀 수(합집합)로 나눕니다.

### Dice Coefficient (다이스 계수, 클래스별)

```
Dice = 2 × |Pred ∩ GT| / (|Pred| + |GT|)
```

IoU와 유사하지만 교집합에 더 큰 가중치를 부여합니다. 동일한 예측에 대해 Dice는 항상 IoU 이상입니다.

### mIoU / mDice

정답 픽셀이 있는 모든 클래스에 대한 평균 IoU와 평균 Dice. 정답 픽셀이 없는 클래스는 평균에서 제외됩니다.

**사용 방법:**
1. **Evaluation** 탭으로 이동
2. 태스크 유형: **Segmentation** 선택
3. 세그멘테이션 모델과 정답 마스크 로드
4. 평가 실행 — 클래스별 IoU/Dice와 전체 mIoU/mDice가 표시됩니다

---

## Classification 지표

분류 모델 (출력: 클래스 확률)의 경우:

- **Accuracy (정확도)**: 올바르게 분류된 이미지의 비율
- **클래스별 Precision/Recall/F1**: 각 클래스에 대해 one-vs-rest 방식으로 계산
- **전체 P/R/F1**: 모든 클래스에 대한 매크로 평균

**사용 방법:**
1. **Evaluation** 탭으로 이동
2. 태스크 유형: **Classification** 선택
3. 분류 모델과 정답 레이블 로드
4. 평가 실행

---

## Embedder 지표

특징 추출 모델 (출력: 임베딩 벡터)의 경우:

### Cosine Similarity (코사인 유사도)

```
similarity = (A · B) / (‖A‖ × ‖B‖)
```

임베딩은 비교 전에 L2 정규화되므로, 코사인 유사도는 내적과 같습니다. 범위: -1 (반대) ~ 1 (동일).

### Retrieval@1

각 쿼리 이미지에 대해 코사인 유사도가 가장 높은 갤러리 이미지를 찾습니다. Retrieval@1 = 상위 1개 결과가 올바른 클래스 레이블을 가진 쿼리의 비율.

### Retrieval@K

Retrieval@1과 동일하지만, 상위 K개 결과 중 어디에든 올바른 클래스가 나타나면 정답으로 간주합니다.

**사용 방법:**
1. **Evaluation** 탭으로 이동
2. 태스크 유형: **Embedder** 선택
3. 특징 추출 ONNX 모델 로드
4. 데이터셋 디렉토리 설정 (폴더 구조: `class_name/image.jpg`)
5. 평가 실행 — Retrieval@1, Retrieval@K, 평균 코사인 유사도가 표시됩니다

---

## Confidence Optimizer

Confidence Optimizer는 임계값을 스위핑하고 F1을 측정하여 각 클래스에 대한 최적의 신뢰도 임계값을 찾습니다.

**동작 원리:**
1. 모든 평가 이미지에 대해 추론을 실행하여 신뢰도 점수가 포함된 예측을 얻습니다.
2. 각 클래스에 대해 신뢰도 임계값을 0.0에서 1.0까지 스위핑합니다.
3. 각 임계값에서 Precision, Recall, F1을 계산합니다.
4. 각 클래스에 대한 **PR 곡선** (Precision vs. Recall)을 그립니다.
5. 각 클래스에서 F1을 최대화하는 임계값을 식별합니다.

**클래스별 임계값이 중요한 이유:**
클래스마다 최적의 임계값이 다를 수 있습니다. 예시가 많은 "사람" 클래스는 0.3에서 가장 잘 작동할 수 있지만, 드문 "휠체어" 클래스는 오탐을 줄이기 위해 0.5가 필요할 수 있습니다.

**사용 방법:**
1. **Analysis** 탭 → **Confidence Optimizer**로 이동
2. 모델과 평가 데이터셋 로드
3. **Run** 클릭 — 클래스별 PR 곡선과 최적 임계값이 표시됩니다
4. 각 클래스의 F1 최대화 임계값이 강조 표시됩니다

---

## FP/FN 오류 분석

오류 분석기는 탐지 오류를 자동으로 분류하여 모델이 어디에서 왜 실패하는지 이해하는 데 도움을 줍니다.

**오류 분류:**

| 오류 유형 | 정의 |
|-----------|------|
| **False Positive (FP, 오탐)** | 정답에 객체가 없는 곳에서 모델이 객체를 탐지 |
| **False Negative (FN, 미탐)** | 정답 객체를 모델이 탐지하지 못함 |

**크기별 분류:**

오류는 바운딩 박스 면적에 따라 분류됩니다:

| 크기 | 면적 범위 | 대표적인 객체 |
|------|-----------|---------------|
| **Small (S)** | < 32² 픽셀 | 먼 거리의 사람, 작은 표지판 |
| **Medium (M)** | 32²~96² 픽셀 | 가까운 사람, 차량 |
| **Large (L)** | > 96² 픽셀 | 근접 객체, 대형 차량 |

**위치별 분석:**

오류는 이미지 내 위치(중앙 vs. 가장자리)별로도 분석되어, 모델이 특정 위치의 객체에서 어려움을 겪는지 파악하는 데 도움을 줍니다.

**사용 방법:**
1. **Analysis** 탭 → **FP/FN Error Analysis**로 이동
2. 모델과 정답이 포함된 평가 데이터셋 로드
3. **Run** 클릭
4. 유형, 크기, 위치별 오류 분석 결과 검토
5. 인사이트를 활용하여 데이터 수집 방향을 결정합니다 (예: 작은 객체의 FN 비율이 높으면, 작은 객체 학습 데이터를 더 수집)
