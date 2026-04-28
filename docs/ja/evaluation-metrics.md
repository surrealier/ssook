# Evaluation Metrics（評価指標）

[← インデックスに戻る](../index.md) | 🌐 [English](../evaluation-metrics.md) | [한국어](../ko/evaluation-metrics.md) | **日本語** | [中文](../zh/evaluation-metrics.md)

ssookは、Detection（検出）、Classification（分類）、Segmentation（セグメンテーション）、CLIP、Embedderの5つのタスクタイプにわたってモデルを評価します。それぞれ独自の指標セットを持ちます。本ドキュメントでは、各指標の計算方法と結果の解釈方法を説明します。

---

## 目次

- [Detection指標](#detection指標)
  - [IoU（Intersection over Union）](#iouintersection-over-union)
  - [Precision、Recall、F1](#precisionrecallf1)
  - [mAP@50](#map50)
  - [mAP@50:95](#map5095)
- [Segmentation指標](#segmentation指標)
- [Classification指標](#classification指標)
- [Embedder指標](#embedder指標)
- [Confidence Optimizer](#confidence-optimizer)
- [FP/FNエラー分析](#fpfnエラー分析)

---

## Detection指標

### IoU（Intersection over Union）

IoUは、予測バウンディングボックスがGround Truth（正解）ボックスとどの程度重なっているかを測定します：

```
IoU = 重なり面積 / 和集合面積
```

- **IoU = 1.0**：完全な重なり
- **IoU = 0.5**：標準的な閾値 — IoU ≥ 0.5の予測は正しい検出（True Positive）とみなされる
- **IoU = 0.0**：重なりなし

ssookは**ベクトル化された行列演算**を使用してIoUを計算します：M個の予測とN個のGround Truthが与えられた場合、効率のためにM×N個のIoU値をすべて同時に計算します。

### Precision、Recall、F1

IoUを使用して予測とGround Truthをマッチングした後：

- **True Positive（TP）**：予測がGround Truthと一致（IoU ≥ 閾値、正しいクラス）
- **False Positive（FP）**：予測がどのGround Truthとも一致しない
- **False Negative（FN）**：どの予測ともマッチしないGround Truth

```
Precision = TP / (TP + FP)    — 「全検出のうち、正しいものはいくつか？」
Recall    = TP / (TP + FN)    — 「全Ground Truthのうち、見つけられたものはいくつか？」
F1        = 2 × P × R / (P + R) — PrecisionとRecallの調和平均
```

### mAP@50

IoU閾値0.5でのMean Average Precision：

1. 各クラスについて、すべての予測を信頼度順（高い順）にソート。
2. 予測を順番に処理。各予測について、IoU ≥ 0.5の未マッチGround Truthと一致するか確認。
3. 各ステップで累積PrecisionとRecallを計算。
4. **101点補間**を使用して**AP**（Average Precision）を計算：Precision-Recall曲線を101個の等間隔Recall点（0.00、0.01、...、1.00）でサンプリングし、補間されたPrecision値を平均。
5. **mAP@50** = 全クラスのAPの平均。

### mAP@50:95

10個のIoU閾値（0.50、0.55、0.60、...、0.95）にわたるmAPの平均で、より厳密な指標です。

ssookはIoU行列を一度計算し、10回の個別評価を実行する代わりにすべての閾値で再利用することで最適化しています。

高いIoU閾値はより正確な位置特定を要求するため、mAP@50:95は常にmAP@50より低くなります。

---

## Segmentation指標

セマンティックセグメンテーションモデル（出力：C×H×Wクラス確率マップ）の場合：

### IoU（クラスごと）

```
IoU = |Pred ∩ GT| / |Pred ∪ GT|
```

ピクセル単位で計算：予測とGround Truthの両方がそのクラスで一致するピクセル数（交差）を、いずれかがそのクラスを予測するピクセル数（和集合）で割ります。

### Dice Coefficient（Dice係数、クラスごと）

```
Dice = 2 × |Pred ∩ GT| / (|Pred| + |GT|)
```

IoUに似ていますが、交差により大きな重みを与えます。同じ予測に対して、DiceはIoU以上の値になります。

### mIoU / mDice

Ground Truthピクセルを持つすべてのクラスにわたる平均IoUと平均Dice。Ground Truthピクセルのないクラスは平均から除外されます。

**使い方：**
1. **Evaluation** タブに移動
2. タスクタイプ：**Segmentation** を選択
3. セグメンテーションモデルとGround Truthマスクを読み込む
4. 評価を実行 — クラスごとのIoU/Diceと全体のmIoU/mDiceが表示されます

---

## Classification指標

分類モデル（出力：クラス確率）の場合：

- **Accuracy（精度）**：正しく分類された画像の割合
- **クラスごとのPrecision/Recall/F1**：各クラスについてone-vs-restアプローチで計算
- **全体のP/R/F1**：全クラスのマクロ平均

**使い方：**
1. **Evaluation** タブに移動
2. タスクタイプ：**Classification** を選択
3. 分類モデルとGround Truthラベルを読み込む
4. 評価を実行

---

## Embedder指標

特徴抽出モデル（出力：Embedding（埋め込み）ベクトル）の場合：

### Cosine Similarity（コサイン類似度）

```
similarity = (A · B) / (‖A‖ × ‖B‖)
```

Embeddingは比較前にL2正規化されるため、コサイン類似度はドット積と等しくなります。範囲：-1（正反対）〜 1（同一）。

### Retrieval@1

各クエリ画像について、コサイン類似度で最も類似したギャラリー画像を見つけます。Retrieval@1 = top-1の結果が正しいクラスラベルを持つクエリの割合。

### Retrieval@K

Retrieval@1と同様ですが、正しいクラスがtop-Kの結果のいずれかに含まれていれば正解とみなされます。

**使い方：**
1. **Evaluation** タブ → **Embedder** に移動
2. 特徴抽出ONNXモデルを読み込む
3. データセットディレクトリを設定（フォルダ構造：`class_name/image.jpg`）
4. 評価を実行 — Retrieval@1、Retrieval@K、平均コサイン類似度が表示されます

---

## Confidence Optimizer

Confidence Optimizerは、閾値をスイープしてF1を測定することで、各クラスの最適な信頼度閾値を見つけます。

**仕組み：**
1. すべての評価画像に対して推論を実行し、信頼度スコア付きの予測を取得。
2. 各クラスについて、信頼度閾値を0.0から1.0までスイープ。
3. 各閾値でPrecision、Recall、F1を計算。
4. 各クラスの**PR曲線**（Precision vs. Recall）をプロット。
5. 各クラスでF1を最大化する閾値を特定。

**クラスごとの閾値が重要な理由：**
クラスによって最適な閾値が異なる場合があります。多くのサンプルを持つ「person」クラスは0.3で最適に動作する一方、まれな「wheelchair」クラスはFalse Positiveを避けるために0.5が必要かもしれません。

**使い方：**
1. **Analysis** タブ → **Confidence Optimizer** に移動
2. モデルと評価データセットを読み込む
3. **Run** をクリック — PR曲線と最適閾値がクラスごとに表示されます
4. 各クラスのF1最大化閾値がハイライトされます

---

## FP/FNエラー分析

Error Analyzerは、モデルがどこでなぜ失敗するかを理解するために、検出エラーを自動的に分類します。

**エラー分類：**

| エラータイプ | 定義 |
|-----------|-----------|
| **False Positive（FP、偽陽性）** | Ground Truthに存在しない場所でモデルがオブジェクトを検出 |
| **False Negative（FN、偽陰性）** | モデルが検出しなかったGround Truthオブジェクト |

**サイズ別内訳：**

エラーはバウンディングボックスの面積で分類されます：

| サイズ | 面積範囲 | 典型的なオブジェクト |
|------|-----------|----------------|
| **Small（S）** | < 32²ピクセル | 遠くの人、小さな標識 |
| **Medium（M）** | 32²〜96²ピクセル | 近くの人、車両 |
| **Large（L）** | > 96²ピクセル | クローズアップのオブジェクト、大型車両 |

**位置ベースの分析：**

エラーは画像内の位置（中央 vs. 端）でも分析され、モデルが特定の位置のオブジェクトで苦戦しているかどうかを特定するのに役立ちます。

**使い方：**
1. **Analysis** タブ → **FP/FN Error Analysis** に移動
2. モデルとGround Truth付き評価データセットを読み込む
3. **Run** をクリック
4. タイプ、サイズ、位置別のエラー内訳を確認
5. インサイトを活用してデータ収集を改善（例：小さなオブジェクトのFN率が高い場合、小さなオブジェクトのトレーニングデータを追加収集）
