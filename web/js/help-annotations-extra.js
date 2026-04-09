/**
 * help-annotations-extra.js
 * 추가 탭(18개)에 대한 페이지별 도움말 어노테이션 데이터
 * Bilingual: { en, ko } for i18n support
 */
var _ANNOTATIONS_EXTRA = {

/* ── 1. model-compare ── */
'model-compare': [
  { page: { en: '📋 Tab Summary', ko: '📋 탭 요약' }, items: [
    { sel: '#page-title', text: { en: '⚔️ Model Compare Tab\nCompare two ONNX detection models\nside-by-side on the same images.\nNavigate with a slider to inspect\ndetection differences per image.', ko: '⚔️ 모델 비교 탭\n두 개의 ONNX 탐지 모델을 동일 이미지에서\n나란히 비교하는 탭입니다.\n슬라이더로 이미지를 탐색하며\n검출 결과 차이를 확인할 수 있습니다.' } },
  ]},
  { page: { en: '🔧 Model Settings', ko: '🔧 모델 설정' }, items: [
    { sel: '#cmp-a', text: { en: 'Select the ONNX file for Model A.\nOnly detection models are supported.\nYOLOv5/v8/v9/v11 and other architectures are auto-detected.', ko: 'Model A로 사용할 ONNX 파일을 선택합니다.\n탐지(Detection) 모델만 지원되며,\nYOLOv5/v8/v9/v11 등 다양한 구조를 자동 인식합니다.' } },
    { sel: '#cmp-b', text: { en: 'Select the ONNX file for Model B.\nUsing a model with the same class set as Model A is recommended.\nTry comparing different versions or training conditions.', ko: 'Model B로 사용할 ONNX 파일을 선택합니다.\nModel A와 동일한 클래스 구성의 모델을 권장합니다.\n서로 다른 버전이나 학습 조건의 모델을 비교해 보세요.' } },
    { sel: '#cmp-type-a', text: { en: 'Select the model type for Model A.\nChoose the correct type (YOLO, CenterNet, etc.) matching the output format.\nIncorrect selection may produce wrong results.', ko: 'Model A의 모델 유형을 선택합니다.\nYOLO 계열, CenterNet 등 출력 형식에 맞게 지정하세요.\n잘못 지정하면 결과가 올바르지 않을 수 있습니다.' } },
    { sel: '#cmp-type-b', text: { en: 'Select the model type for Model B.\nThe two models can have different types.', ko: 'Model B의 모델 유형을 선택합니다.\n두 모델의 유형이 달라도 비교 가능합니다.' } },
    { sel: '#cmp-img', text: { en: 'Specify the folder containing images for comparison.\nSupports common formats like JPG and PNG.\nMore images will take longer to process.', ko: '비교에 사용할 이미지가 들어 있는 폴더를 지정합니다.\nJPG, PNG 등 일반 이미지 형식을 지원합니다.\n이미지 수가 많을수록 비교가 오래 걸립니다.' } },
  ]},
  { page: { en: '📊 Comparison Results', ko: '📊 비교 결과' }, items: [
    { sel: '#cmp-stop', text: { en: 'Stop the comparison in progress.\nResults up to the point of stopping are preserved.', ko: '진행 중인 비교를 중단합니다.\n중단 시점까지의 결과는 유지됩니다.' } },
    { sel: '#cmp-prog', text: { en: 'Shows progress relative to the total number of images.', ko: '전체 이미지 대비 처리 진행률을 표시합니다.' } },
    { sel: '#cmp-slider', text: { en: 'Move the slider to browse images.\nYou can also use the left/right arrow keys.\nCompare the results of both models per image.', ko: '슬라이더를 움직여 이미지를 탐색합니다.\n좌우 화살표 키로도 이동할 수 있습니다.\n각 이미지별 두 모델의 결과를 비교해 보세요.' } },
    { sel: '#cmp-counter', text: { en: 'Shows the current image number and total count.', ko: '현재 보고 있는 이미지 번호와 전체 수를 표시합니다.' } },
    { sel: '#cmp-panel-a', text: { en: 'Inference results for Model A.\nShows bounding box count and inference time (ms),\nwith detections visualized on the image.', ko: 'Model A의 추론 결과입니다.\n바운딩 박스 수, 추론 시간(ms)이 표시되며,\n이미지 위에 탐지 결과가 시각화됩니다.' } },
    { sel: '#cmp-panel-b', text: { en: 'Inference results for Model B.\nCompare results on the same image side-by-side\nto intuitively see performance differences.', ko: 'Model B의 추론 결과입니다.\n같은 이미지에 대한 결과를 나란히 비교하여\n모델 간 성능 차이를 직관적으로 확인하세요.' } },
  ]},
],

/* ── 2. error-analyzer ── */
'error-analyzer': [
  { page: { en: '📋 Tab Summary', ko: '📋 탭 요약' }, items: [
    { sel: '#page-title', text: { en: '🎯 FP/FN Error Analysis Tab\nClassify model false positives (FP) and\nfalse negatives (FN) by size (S/M/L)\nand position (Top/Center/Bottom).', ko: '🎯 FP/FN 오류 분석 탭\n모델의 오탐(FP)과 미탐(FN)을\n크기별(S/M/L), 위치별(Top/Center/Bottom)로\n분류하여 분석하는 탭입니다.' } },
  ]},
  { page: { en: '🔧 Input Settings', ko: '🔧 입력 설정' }, items: [
    { sel: '#ea-model', text: { en: 'Select the ONNX detection model to analyze.\nOnly detection models are supported.\nThe model output must be in bounding box format.', ko: '분석할 ONNX 탐지 모델을 선택합니다.\n탐지(Detection) 모델만 지원됩니다.\n모델의 출력이 바운딩 박스 형식이어야 합니다.' } },
    { sel: '#ea-type', text: { en: 'Select the model type.\nChoose the correct type (YOLO, CenterNet, etc.) matching the output format.', ko: '모델 유형을 선택합니다.\nYOLO, CenterNet 등 출력 형식에 맞게 지정하세요.' } },
    { sel: '#ea-img', text: { en: 'Specify the image folder to analyze.\nFilenames must match the label files (only the extension differs).', ko: '분석할 이미지 폴더를 지정합니다.\n라벨 파일과 동일한 파일명(확장자만 다름)이어야 합니다.' } },
    { sel: '#ea-lbl', text: { en: 'Specify the Ground Truth label folder.\nOnly YOLO format (.txt) is supported.\nEach line: class_id cx cy w h (normalized coordinates)', ko: 'Ground Truth 라벨 폴더를 지정합니다.\nYOLO 형식(.txt)만 지원됩니다.\n각 줄: class_id cx cy w h (정규화 좌표)' } },
    { sel: '#ea-iou', text: { en: 'Set the IoU threshold.\n0.5: Standard criterion (COCO mAP@50)\n0.75: Strict criterion (emphasizes localization)\nHigher values produce more FP/FN.', ko: 'IoU 임계값을 설정합니다.\n0.5: 일반적 기준 (COCO mAP@50)\n0.75: 엄격한 기준 (위치 정확도 중시)\n값이 높을수록 FP/FN이 많아집니다.' } },
  ]},
  { page: { en: '📋 Result Interpretation', ko: '📋 결과 해석' }, items: [
    { sel: '#ea-stop', text: { en: 'Stop the analysis. Results up to the point of stopping are preserved.', ko: '분석을 중단합니다. 중단 시점까지의 결과는 유지됩니다.' } },
    { sel: '#ea-prog', text: { en: 'Analysis progress relative to total images.', ko: '전체 이미지 대비 분석 진행률입니다.' } },
    { sel: '#ea-results', text: { en: 'FP (false positive) and FN (false negative) analysis results.\nBy size: S(<32²px), M(<96²px), L(≥96²px)\nBy position: Top/Center/Bottom regions\nIf FP is high, try raising the confidence threshold.\nIf FN is high, consider retraining or data augmentation.', ko: 'FP(오탐)와 FN(미탐) 분석 결과입니다.\n크기별 분류: S(<32²px), M(<96²px), L(≥96²px)\n위치별 분류: Top/Center/Bottom 영역\nFP가 많으면 신뢰도 임계값을 올려 보세요.\nFN이 많으면 모델 재학습이나 데이터 보강을 고려하세요.' } },
  ]},
],

/* ── 3. conf-optimizer ── */
'conf-optimizer': [
  { page: { en: '📋 Tab Summary', ko: '📋 탭 요약' }, items: [
    { sel: '#page-title', text: { en: '📈 Confidence Optimizer Tab\nAutomatically find the optimal confidence\nthreshold per class that maximizes\nthe F1 score.', ko: '📈 신뢰도 최적화 탭\n클래스별로 F1 점수를 최대화하는\n최적의 Confidence 임계값을 자동으로\n탐색하는 탭입니다.' } },
  ]},
  { page: { en: '🔧 Input Settings', ko: '🔧 입력 설정' }, items: [
    { sel: '#co-model', text: { en: 'Select the ONNX detection model to optimize.\nSearches for the optimal confidence threshold per class.', ko: '최적화할 ONNX 탐지 모델을 선택합니다.\n각 클래스별로 최적의 신뢰도 임계값을 탐색합니다.' } },
    { sel: '#co-type', text: { en: 'Select the model type.', ko: '모델 유형을 선택합니다.' } },
    { sel: '#co-img', text: { en: 'Specify the image folder for evaluation.\nUsing a validation set is recommended.', ko: '평가용 이미지 폴더를 지정합니다.\n검증(val) 세트 사용을 권장합니다.' } },
    { sel: '#co-lbl', text: { en: 'Ground Truth label folder.\nYOLO format (.txt) files are required.', ko: 'Ground Truth 라벨 폴더입니다.\nYOLO 형식(.txt) 파일이 필요합니다.' } },
    { sel: '#co-step', text: { en: 'Confidence sweep step size.\n0.05: Fast but approximate (recommended)\n0.01: Precise but 20× slower\nSmaller values find the optimum more accurately but take longer.', ko: '신뢰도 탐색 간격입니다.\n0.05: 빠르지만 대략적 (권장)\n0.01: 정밀하지만 20배 느림\n값이 작을수록 최적점을 정확히 찾지만 시간이 오래 걸립니다.' } },
  ]},
  { page: { en: '📋 Result Interpretation', ko: '📋 결과 해석' }, items: [
    { sel: '#co-stop', text: { en: 'Stop the optimization.', ko: '최적화를 중단합니다.' } },
    { sel: '#co-prog', text: { en: 'Overall sweep progress.', ko: '전체 탐색 진행률입니다.' } },
    { sel: '#co-results', text: { en: 'Per-class optimal confidence threshold results.\nBest Threshold: Confidence value that maximizes F1\nF1: Harmonic mean of Precision and Recall (1.0 is best)\nPrecision: Ratio of true positives among detections\nRecall: Ratio of ground truths that were detected\nUse these values as per-class thresholds in deployment.', ko: '클래스별 최적 신뢰도 임계값 결과입니다.\nBest Threshold: F1 점수가 최대인 신뢰도 값\nF1: 정밀도와 재현율의 조화 평균 (1.0이 최고)\nPrecision: 탐지 중 실제 정답 비율\nRecall: 정답 중 탐지된 비율\n이 값을 실제 배포 시 클래스별 임계값으로 사용하세요.' } },
  ]},
],

/* ── 4. embedding-viewer ── */
'embedding-viewer': [
  { page: { en: '📋 Tab Summary', ko: '📋 탭 요약' }, items: [
    { sel: '#page-title', text: { en: '🗺️ Embedding Visualization Tab\nVisualize image embedding vectors as\n2D scatter plots using t-SNE/UMAP/PCA\nto check class separability.', ko: '🗺️ 임베딩 시각화 탭\n이미지 임베딩 벡터를 t-SNE/UMAP/PCA로\n2D 산점도로 시각화하여 클래스 분리도를\n확인하는 탭입니다.' } },
  ]},
  { page: { en: '🔧 Input Settings', ko: '🔧 입력 설정' }, items: [
    { sel: '#ev-model', text: { en: 'Select the ONNX model for embedding extraction.\nOutput shape: 1×D vector (D = embedding dimension)\nUse a classification model (ResNet, EfficientNet, etc.)\nwith the final FC layer removed.', ko: '임베딩 추출용 ONNX 모델을 선택합니다.\n출력 형태: 1×D 벡터 (D: 임베딩 차원)\nResNet, EfficientNet 등 분류 모델의\n마지막 FC 레이어 제거 버전을 사용하세요.' } },
    { sel: '#ev-img', text: { en: 'Specify the image folder.\nSubfolder names are used as class labels.\nExample: images/cat/, images/dog/\nAt least 10 images per class is recommended.', ko: '이미지 폴더를 지정합니다.\n하위 폴더명이 클래스 라벨로 사용됩니다.\n예: images/cat/, images/dog/\n클래스당 최소 10장 이상을 권장합니다.' } },
    { sel: '#ev-method', text: { en: 'Select the dimensionality reduction method.\nt-SNE: Preserves local structure, best for cluster visualization\nUMAP: Preserves global+local structure, fast (requires install)\nPCA: Fastest, captures only linear structure', ko: '차원 축소 방법을 선택합니다.\nt-SNE: 지역 구조 보존, 클러스터 시각화에 최적\nUMAP: 전역+지역 구조 보존, 빠름 (설치 필요)\nPCA: 가장 빠름, 선형 구조만 반영' } },
  ]},
  { page: { en: '📊 Visualization Results', ko: '📊 시각화 결과' }, items: [
    { sel: '#ev-stop', text: { en: 'Stop embedding extraction.', ko: '임베딩 추출을 중단합니다.' } },
    { sel: '#ev-prog', text: { en: 'Image processing progress.', ko: '이미지 처리 진행률입니다.' } },
    { sel: '#ev-plot', text: { en: '2D scatter plot. Each dot represents one image.\nSame color = same class\nWell-separated clusters indicate good class discrimination.\nOverlapping regions contain samples the model confuses.\nOutliers may indicate labeling errors — worth checking.', ko: '2D 산점도입니다. 각 점은 하나의 이미지를 나타냅니다.\n같은 색상 = 같은 클래스\n클러스터가 잘 분리되면 모델이 클래스를 잘 구분합니다.\n겹치는 영역의 이미지는 모델이 혼동하는 샘플입니다.\n이상치(outlier)는 라벨 오류일 수 있으니 확인해 보세요.' } },
  ]},
],

/* ── 5. segmentation ── */
'segmentation': [
  { page: { en: '📋 Tab Summary', ko: '📋 탭 요약' }, items: [
    { sel: '#page-title', text: { en: '🖼️ Segmentation Evaluation Tab\nCompare segmentation model predictions\nagainst GT masks and compute IoU, Dice,\nand other performance metrics.', ko: '🖼️ 세그멘테이션 평가 탭\n세그멘테이션 모델의 예측 마스크를\nGT 마스크와 비교하여 IoU, Dice 등\n성능 지표를 산출하는 탭입니다.' } },
  ]},
  { page: { en: '🔧 Input Settings', ko: '🔧 입력 설정' }, items: [
    { sel: '#seg-model', text: { en: 'Select the segmentation ONNX model.\nOutput shape: B×C×H×W (C = number of classes)\nSupports DeepLabV3, U-Net, FCN, YOLO-seg, etc.', ko: '세그멘테이션 ONNX 모델을 선택합니다.\n출력 형태: B×C×H×W (C=클래스 수)\nDeepLabV3, U-Net, FCN, YOLO-seg 등을 지원합니다.' } },
    { sel: '#seg-img', text: { en: 'Specify the image folder for evaluation.\nFilenames must match the GT masks.', ko: '평가할 이미지 폴더를 지정합니다.\nGT 마스크와 동일한 파일명이어야 합니다.' } },
    { sel: '#seg-lbl', text: { en: 'Ground Truth mask folder.\nPNG format, pixel value = class ID\n0 = background, 1 = first class, 2 = second class…\nMust have the same resolution as the images.', ko: 'Ground Truth 마스크 폴더입니다.\nPNG 형식, 픽셀 값 = 클래스 ID\n0 = 배경, 1 = 첫 번째 클래스, 2 = 두 번째 클래스...\n이미지와 동일한 해상도여야 합니다.' } },
  ]},
  { page: { en: '📋 Result Interpretation', ko: '📋 결과 해석' }, items: [
    { sel: '#seg-prog', text: { en: 'Evaluation progress relative to total images.', ko: '전체 이미지 대비 평가 진행률입니다.' } },
    { sel: '#seg-results', text: { en: 'Per-class segmentation evaluation results.\nIoU: Intersection/Union of prediction and GT (0–1, higher is better)\nDice: 2×Intersection/(Prediction+GT), more lenient than IoU\nmIoU: Mean IoU across all classes, ≥0.5 is acceptable\nFor medical imaging, aim for Dice ≥ 0.8.', ko: '클래스별 세그멘테이션 평가 결과입니다.\nIoU: 예측과 GT의 교집합/합집합 (0~1, 높을수록 좋음)\nDice: 2×교집합/(예측+GT), IoU보다 관대한 지표\nmIoU: 전체 클래스 IoU 평균, 0.5 이상이면 양호\n의료 영상에서는 Dice 0.8 이상을 목표로 합니다.' } },
  ]},
],

/* ── 6. clip ── */
'clip': [
  { page: { en: '📋 Tab Summary', ko: '📋 탭 요약' }, items: [
    { sel: '#page-title', text: { en: '🔤 CLIP Zero-Shot Tab\nUse CLIP image/text encoders to classify\nimages with only text labels,\nno training required.', ko: '🔤 CLIP Zero-Shot 탭\nCLIP 이미지/텍스트 인코더를 사용하여\n학습 없이 텍스트 라벨만으로\n이미지를 분류하는 탭입니다.' } },
  ]},
  { page: { en: '🔧 Model Settings', ko: '🔧 모델 설정' }, items: [
    { sel: '#clip-img-enc', text: { en: 'Select the CLIP image encoder ONNX file.\nConverts images into embedding vectors.\nMust be the same CLIP version as the text encoder.', ko: 'CLIP 이미지 인코더 ONNX 파일을 선택합니다.\n이미지를 임베딩 벡터로 변환하는 모델입니다.\n텍스트 인코더와 동일한 CLIP 버전이어야 합니다.' } },
    { sel: '#clip-txt-enc', text: { en: 'Select the CLIP text encoder ONNX file.\nConverts text into embedding vectors.\nBoth encoders are required; download them from the Settings tab.', ko: 'CLIP 텍스트 인코더 ONNX 파일을 선택합니다.\n텍스트를 임베딩 벡터로 변환하는 모델입니다.\n두 인코더 모두 필수이며, 설정 탭에서 다운로드 가능합니다.' } },
  ]},
  { page: { en: '📁 Data & Labels', ko: '📁 데이터 & 라벨' }, items: [
    { sel: '#clip-img', text: { en: 'Specify the image folder.\nSubfolder names are used as GT classes.\nExample: images/dog/, images/cat/\nFolder names do not need to match the class labels.', ko: '이미지 폴더를 지정합니다.\n하위 폴더명이 GT 클래스로 사용됩니다.\n예: images/dog/, images/cat/\n폴더명과 클래스 라벨이 일치하지 않아도 됩니다.' } },
    { sel: '#clip-labels', text: { en: 'Enter class labels separated by commas.\nExample: dog, cat, bird\nEnglish recommended (CLIP is trained on English)\nPrompt-style labels like "a photo of a dog" also work.', ko: '분류할 클래스 라벨을 쉼표로 구분하여 입력합니다.\n예: dog, cat, bird\n영어 권장 (CLIP은 영어 학습 모델)\n"a photo of a dog" 형태의 프롬프트도 가능합니다.' } },
  ]},
  { page: { en: '📊 Result Interpretation', ko: '📊 결과 해석' }, items: [
    { sel: '#clip-prog', text: { en: 'Image processing progress.', ko: '이미지 처리 진행률입니다.' } },
    { sel: '#clip-results', text: { en: 'Per-label zero-shot classification accuracy.\nTotal: Number of images for that class\nCorrect: Number correctly classified\nAccuracy: Correct / Total\nMisclassified images can be reviewed in the detail view.', ko: '라벨별 Zero-shot 분류 정확도입니다.\nTotal: 해당 클래스의 전체 이미지 수\nCorrect: 올바르게 분류된 수\nAccuracy: 정확도 (Correct/Total)\n오분류된 이미지는 상세 보기에서 확인할 수 있습니다.' } },
  ]},
],

/* ── 7. embedder ── */
'embedder': [
  { page: { en: '📋 Tab Summary', ko: '📋 탭 요약' }, items: [
    { sel: '#page-title', text: { en: '🧲 Embedder Evaluation Tab\nEvaluate embedding model retrieval performance.\nMeasure vector quality using Retrieval@1/@K\nand cosine similarity across same-class pairs.', ko: '🧲 임베더 평가 탭\n임베딩 모델의 검색 성능을 평가합니다.\nRetrieval@1/@K, 코사인 유사도로\n같은 클래스 간 벡터 품질을 측정합니다.' } },
  ]},
  { page: { en: '🔧 Input Settings', ko: '🔧 입력 설정' }, items: [
    { sel: '#emb-model', text: { en: 'Select the embedding model ONNX file.\nOutput shape: 1×D vector (feature vector)\nUse a classification backbone or metric learning model.', ko: '임베딩 모델 ONNX 파일을 선택합니다.\n출력 형태: 1×D 벡터 (특징 벡터)\n분류 모델의 backbone 또는 metric learning 모델을 사용하세요.' } },
    { sel: '#emb-img', text: { en: 'Specify the image folder.\nSubfolder names are used as class labels.\nExample: dataset/class_a/, dataset/class_b/\nAt least 5 images per class is recommended.', ko: '이미지 폴더를 지정합니다.\n하위 폴더명이 클래스 라벨로 사용됩니다.\n예: dataset/class_a/, dataset/class_b/\n클래스당 최소 5장 이상을 권장합니다.' } },
    { sel: '#emb-k', text: { en: 'Set the Top-K value.\nReturns the top K results during retrieval.\nUsed for Retrieval@K evaluation;\ntypically 5 or 10 is recommended.', ko: 'Top-K 값을 설정합니다.\n검색 시 상위 K개의 결과를 반환합니다.\nRetrieval@K 평가에 사용되며,\n일반적으로 5 또는 10을 권장합니다.' } },
  ]},
  { page: { en: '📋 Result Interpretation', ko: '📋 결과 해석' }, items: [
    { sel: '#emb-prog', text: { en: 'Embedding extraction and evaluation progress.', ko: '임베딩 추출 및 평가 진행률입니다.' } },
    { sel: '#emb-results', text: { en: 'Per-class retrieval performance results.\nRetrieval@1: Ratio where the most similar result is the same class\nRetrieval@K: Ratio where at least one of the top K is the same class\nAvg Cosine: Mean cosine similarity within the same class\nCloser to 1.0 means higher intra-class cohesion.', ko: '클래스별 검색 성능 평가 결과입니다.\nRetrieval@1: 가장 유사한 1개가 같은 클래스인 비율\nRetrieval@K: 상위 K개 중 같은 클래스가 있는 비율\nAvg Cosine: 같은 클래스 간 평균 코사인 유사도\n1.0에 가까울수록 클래스 내 응집도가 높습니다.' } },
  ]},
],

/* ── 8. converter ── */
'converter': [
  { page: { en: '📋 Tab Summary', ko: '📋 탭 요약' }, items: [
    { sel: '#page-title', text: { en: '🔄 Format Converter Tab\nConvert between YOLO (.txt), COCO (.json),\nand Pascal VOC (.xml) label formats.', ko: '🔄 형식 변환 탭\nYOLO(.txt), COCO(.json), Pascal VOC(.xml)\n라벨 형식을 상호 변환하는 탭입니다.' } },
  ]},
  { page: { en: '🔄 Conversion Settings', ko: '🔄 변환 설정' }, items: [
    { sel: '#conv-in', text: { en: 'Specify the folder containing label files to convert.\nMust contain files matching the source format.', ko: '변환할 라벨 파일이 있는 폴더를 지정합니다.\n소스 형식에 맞는 파일이 들어 있어야 합니다.' } },
    { sel: '#conv-out', text: { en: 'Output folder for converted files.\nExisting files will be overwritten — use caution.', ko: '변환된 파일이 저장될 출력 폴더입니다.\n기존 파일이 있으면 덮어쓰므로 주의하세요.' } },
    { sel: '#conv-from', text: { en: 'Select the source label format.\nYOLO: .txt (class cx cy w h, normalized coordinates)\nCOCO: .json (single JSON, absolute coordinates)\nPascal VOC: .xml (per-image XML, absolute coordinates)', ko: '원본 라벨 형식을 선택합니다.\nYOLO: .txt (class cx cy w h, 정규화 좌표)\nCOCO: .json (단일 JSON, 절대 좌표)\nPascal VOC: .xml (이미지별 XML, 절대 좌표)' } },
    { sel: '#conv-to', text: { en: 'Select the target format.\nChoose a format different from the source.\nCoordinate systems are converted automatically.', ko: '변환할 대상 형식을 선택합니다.\n소스와 다른 형식을 선택하세요.\n좌표계가 자동으로 변환됩니다.' } },
    { sel: '#conv-prog', text: { en: 'Conversion progress. Check the output folder when complete.', ko: '변환 진행률입니다. 완료 후 출력 폴더를 확인하세요.' } },
  ]},
],

/* ── 9. remapper ── */
'remapper': [
  { page: { en: '📋 Tab Summary', ko: '📋 탭 요약' }, items: [
    { sel: '#page-title', text: { en: '🏷️ Class Remapper Tab\nChange, merge, or delete class IDs\nin YOLO labels.\nAuto-reindexing keeps IDs contiguous.', ko: '🏷️ 클래스 재매핑 탭\nYOLO 라벨의 클래스 ID를 변경, 병합,\n삭제하는 탭입니다.\n자동 재인덱싱으로 연속 ID를 유지합니다.' } },
  ]},
  { page: { en: '🏷️ Remapping Settings', ko: '🏷️ 재매핑 설정' }, items: [
    { sel: '#remap-lbl', text: { en: 'Label folder whose class IDs will be changed.\nYOLO format (.txt) files are required.', ko: '클래스 ID를 변경할 라벨 폴더입니다.\nYOLO 형식(.txt) 파일이 필요합니다.' } },
    { sel: '#remap-out', text: { en: 'Output folder for remapped labels.\nOriginal files are not modified.', ko: '재매핑된 라벨이 저장될 출력 폴더입니다.\n원본 파일은 수정되지 않습니다.' } },
    { sel: '#remap-map', text: { en: 'Enter class mapping rules.\nFormat: old:new (one per line)\nExample: 0:1  → change class 0 to 1\nExample: 2:0, 3:0  → merge classes 2 & 3 into 0\nClasses not in the mapping are deleted.', ko: '클래스 매핑 규칙을 입력합니다.\n형식: old:new (한 줄에 하나씩)\n예: 0:1  → 클래스 0을 1로 변경\n예: 2:0, 3:0  → 클래스 2,3을 0으로 병합\n매핑에 없는 클래스는 삭제됩니다.' } },
    { sel: '#remap-reindex', text: { en: 'Enable auto-reindexing.\nRenumbers class IDs sequentially from 0 after mapping.\nExample: {0,3,7} → {0,1,2}\nUseful when training requires contiguous class IDs.', ko: '자동 재인덱싱을 활성화합니다.\n매핑 후 클래스 ID를 0부터 연속으로 재정렬합니다.\n예: {0,3,7} → {0,1,2}\n학습 시 클래스 수와 ID가 연속이어야 할 때 유용합니다.' } },
  ]},
],

/* ── 10. merger ── */
'merger': [
  { page: { en: '📋 Tab Summary', ko: '📋 탭 요약' }, items: [
    { sel: '#page-title', text: { en: '🔗 Dataset Merger Tab\nCombine multiple datasets into one.\nAutomatically removes duplicates\nusing dHash-based detection.', ko: '🔗 데이터셋 병합 탭\n여러 데이터셋을 하나로 합치는 탭입니다.\ndHash 기반 중복 탐지로\n동일 이미지를 자동 제거합니다.' } },
  ]},
  { page: { en: '🔗 Merge Settings', ko: '🔗 병합 설정' }, items: [
    { sel: '#merge-d1', text: { en: 'First dataset folder to merge.\nAn images/ and labels/ subfolder structure is recommended.\nUse the + button to add more datasets.', ko: '병합할 첫 번째 데이터셋 폴더입니다.\nimages/와 labels/ 하위 폴더 구조를 권장합니다.\n+ 버튼으로 추가 데이터셋을 더할 수 있습니다.' } },
    { sel: '#merge-out', text: { en: 'Output folder for the merged dataset.\nA suffix is added automatically on filename conflicts.', ko: '병합된 데이터셋이 저장될 출력 폴더입니다.\n파일명 충돌 시 자동으로 접미사가 추가됩니다.' } },
  ]},
],

/* ── 11. sampler ── */
'sampler': [
  { page: { en: '📋 Tab Summary', ko: '📋 탭 요약' }, items: [
    { sel: '#page-title', text: { en: '📊 Smart Sampler Tab\nExtract a target number of images from\na dataset using Random, Balanced,\nor Stratified sampling strategies.', ko: '📊 스마트 샘플러 탭\nRandom/Balanced/Stratified 전략으로\n데이터셋에서 원하는 수만큼\n이미지를 추출하는 탭입니다.' } },
  ]},
  { page: { en: '📊 Sampling Settings', ko: '📊 샘플링 설정' }, items: [
    { sel: '#samp-img', text: { en: 'Specify the image folder to sample from.', ko: '샘플링할 이미지 폴더를 지정합니다.' } },
    { sel: '#samp-lbl', text: { en: 'Specify the label folder.\nYOLO format (.txt); required for Stratified/Balanced strategies.', ko: '라벨 폴더를 지정합니다.\nYOLO 형식(.txt)이며, Stratified/Balanced 전략에 필요합니다.' } },
    { sel: '#samp-out', text: { en: 'Output folder for sampled data.', ko: '샘플링된 데이터가 저장될 출력 폴더입니다.' } },
    { sel: '#samp-strat', text: { en: 'Select the sampling strategy.\nRandom: Random selection (fastest)\nBalanced: Equal count per class (augments minority classes)\nStratified: Preserves original class distribution', ko: '샘플링 전략을 선택합니다.\nRandom: 무작위 추출 (가장 빠름)\nBalanced: 클래스별 동일 수량 추출 (소수 클래스 보강)\nStratified: 원본 클래스 비율 유지하며 추출' } },
    { sel: '#samp-n', text: { en: 'Target number of images to extract.\nIf larger than the source, all images are selected.', ko: '추출할 목표 이미지 수입니다.\n원본보다 큰 값을 지정하면 전체가 선택됩니다.' } },
    { sel: '#samp-seed', text: { en: 'Random seed value.\nSame seed = same results (reproducibility).\nLeave empty for different results each time.', ko: '랜덤 시드 값입니다.\n같은 시드 = 같은 결과 (재현성 보장)\n비워두면 매번 다른 결과가 나옵니다.' } },
    { sel: '#samp-lbl-chk', text: { en: 'When checked, label files are copied along with images.\nAlways enable this when building a training dataset.', ko: '체크하면 이미지와 함께 라벨 파일도 복사합니다.\n학습용 데이터셋 구성 시 반드시 체크하세요.' } },
  ]},
],

/* ── 12. anomaly ── */
'anomaly': [
  { page: { en: '📋 Tab Summary', ko: '📋 탭 요약' }, items: [
    { sel: '#page-title', text: { en: '🛡️ Label Anomaly Detector Tab\nAutomatically detect anomalies in YOLO labels\nsuch as out-of-bounds coordinates, tiny/huge boxes,\nabnormal aspect ratios, and excessive overlaps.', ko: '🛡️ 라벨 이상 탐지 탭\nYOLO 라벨에서 경계 밖 좌표, 극소/극대 박스,\n비정상 종횡비, 과도한 겹침 등\n이상 항목을 자동 검출하는 탭입니다.' } },
  ]},
  { page: { en: '🔧 Input Settings', ko: '🔧 입력 설정' }, items: [
    { sel: '#anom-img', text: { en: 'Specify the image folder to inspect.\nFilenames must match the label files.', ko: '검사할 이미지 폴더를 지정합니다.\n라벨 파일과 동일한 파일명이어야 합니다.' } },
    { sel: '#anom-lbl', text: { en: 'Label folder to inspect.\nOnly YOLO format (.txt) is supported.\nEach line: class_id cx cy w h (normalized coordinates)', ko: '검사할 라벨 폴더입니다.\nYOLO 형식(.txt)만 지원됩니다.\n각 줄: class_id cx cy w h (정규화 좌표)' } },
  ]},
  { page: { en: '📋 Result Interpretation', ko: '📋 결과 해석' }, items: [
    { sel: '#anom-prog', text: { en: 'Label inspection progress.', ko: '라벨 검사 진행률입니다.' } },
    { sel: '#anom-results', text: { en: 'Label anomaly detection results.\nOOB: Coordinates outside image bounds (exceeds 0–1)\nTiny Box: Very small box (<32²px), possible labeling error\nHuge Box: Abnormally large box (>80% of image)\nAspect Ratio: Extreme aspect ratio (>10:1)\nOverlap: Excessive overlap (IoU>0.9), suspected duplicate label\nFix high-severity items first.', ko: '라벨 이상 탐지 결과입니다.\nOOB: 좌표가 이미지 범위를 벗어남 (0~1 초과)\nTiny Box: 너무 작은 박스 (<32²px), 라벨 오류 가능\nHuge Box: 비정상적으로 큰 박스 (이미지 80% 이상)\nAspect Ratio: 극단적 종횡비 (>10:1)\nOverlap: 과도한 겹침 (IoU>0.9), 중복 라벨 의심\n심각도(Severity)가 높은 항목부터 우선 수정하세요.' } },
  ]},
],

/* ── 13. quality ── */
'quality': [
  { page: { en: '📋 Tab Summary', ko: '📋 탭 요약' }, items: [
    { sel: '#page-title', text: { en: '🖼️ Image Quality Checker Tab\nAutomatically measure image quality metrics\nsuch as blur, brightness, entropy, and aspect ratio\nto filter out problematic images.', ko: '🖼️ 이미지 품질 검사 탭\n흐림, 밝기, 엔트로피, 종횡비 등\n이미지 품질 지표를 자동 측정하여\n문제 이미지를 필터링하는 탭입니다.' } },
  ]},
  { page: { en: '🖼️ Input & Results', ko: '🖼️ 입력 & 결과' }, items: [
    { sel: '#qual-img', text: { en: 'Specify the image folder to check.\nSupports common formats like JPG and PNG.', ko: '품질 검사할 이미지 폴더를 지정합니다.\nJPG, PNG 등 일반 이미지 형식을 지원합니다.' } },
    { sel: '#qual-prog', text: { en: 'Image quality check progress.', ko: '이미지 품질 검사 진행률입니다.' } },
    { sel: '#qual-results', text: { en: 'Image quality check results.\nBlur: Blur score (lower = blurrier, <100 is concerning)\nBrightness: Brightness (0–255, watch for extremes)\nEntropy: Information content (low = monotonous image)\nAspect: Aspect ratio (extreme ratios hurt training)\nIssues: Summary of detected problems\nConsider excluding problematic images from training data.', ko: '이미지 품질 검사 결과입니다.\nBlur: 흐림 점수 (낮을수록 흐림, <100 주의)\nBrightness: 밝기 (0~255, 극단값 주의)\nEntropy: 정보량 (낮으면 단조로운 이미지)\nAspect: 종횡비 (극단적 비율은 학습에 불리)\nIssues: 발견된 문제 요약\n문제 이미지는 학습 데이터에서 제외를 고려하세요.' } },
  ]},
],

/* ── 14. duplicate ── */
'duplicate': [
  { page: { en: '📋 Tab Summary', ko: '📋 탭 요약' }, items: [
    { sel: '#page-title', text: { en: '👯 Near-Duplicate Detector Tab\nFind similar/duplicate image pairs\nwithin a folder using dHash perceptual hashing.', ko: '👯 유사 이미지 탐지 탭\ndHash 지각 해싱으로 폴더 내\n유사/중복 이미지 쌍을 찾는 탭입니다.' } },
  ]},
  { page: { en: '👯 Duplicate Detection Settings', ko: '👯 중복 탐지 설정' }, items: [
    { sel: '#dup-img', text: { en: 'Specify the image folder to check for duplicates.\nAll image pairs are compared, so large sets take longer.', ko: '중복 검사할 이미지 폴더를 지정합니다.\n모든 이미지 쌍을 비교하므로 수량이 많으면 시간이 걸립니다.' } },
    { sel: '#dup-thr', text: { en: 'Set the Hamming distance threshold.\n0: Detect only identical images\n10: Recommended (allows minor differences)\n20+: Loose criterion (includes similar images)\ndHash perceptual hashing ignores resize/compression differences.', ko: 'Hamming 거리 임계값을 설정합니다.\n0: 완전 동일한 이미지만 탐지\n10: 권장값 (약간의 차이 허용)\n20+: 느슨한 기준 (유사 이미지도 포함)\ndHash 기반 지각 해싱으로 리사이즈/압축 차이를 무시합니다.' } },
    { sel: '#dup-prog', text: { en: 'Duplicate detection progress.', ko: '중복 탐지 진행률입니다.' } },
    { sel: '#dup-results', text: { en: 'Duplicate group results.\nGroup: Duplicate group number\nImage A/B: Image pair flagged as duplicates\nDistance: Hamming distance (lower = more similar)\nKeep only one per group and remove the rest.', ko: '중복 그룹 결과입니다.\nGroup: 중복 그룹 번호\nImage A/B: 중복으로 판정된 이미지 쌍\nDistance: Hamming 거리 (낮을수록 유사)\n같은 그룹 내에서 하나만 남기고 제거하세요.' } },
  ]},
],

/* ── 15. leaky ── */
'leaky': [
  { page: { en: '📋 Tab Summary', ko: '📋 탭 요약' }, items: [
    { sel: '#page-title', text: { en: '🔍 Leaky Split Detector Tab\nCheck whether identical images exist\nacross Train/Val/Test splits\nto prevent data leakage.', ko: '🔍 교차 분할 중복 검출 탭\nTrain/Val/Test 간 동일 이미지가\n존재하는지 검사하여 데이터 누수를\n방지하는 탭입니다.' } },
  ]},
  { page: { en: '🔍 Split Verification', ko: '🔍 분할 검증' }, items: [
    { sel: '#leak-train', text: { en: 'Specify the Train set image folder.', ko: 'Train 세트 이미지 폴더를 지정합니다.' } },
    { sel: '#leak-val', text: { en: 'Specify the Validation set image folder.', ko: 'Validation 세트 이미지 폴더를 지정합니다.' } },
    { sel: '#leak-test', text: { en: 'Specify the Test set image folder.\nIf empty, only Train↔Val is checked.', ko: 'Test 세트 이미지 폴더를 지정합니다.\n비어 있으면 Train↔Val만 검사합니다.' } },
    { sel: '#leak-prog', text: { en: 'Cross-split duplicate check progress.', ko: '교차 분할 중복 검사 진행률입니다.' } },
    { sel: '#leak-results', text: { en: 'Cross-split duplicate detection results.\nSplit Pair: Compared split pair (e.g., Train↔Val)\nDuplicates: Number of duplicates found\nFiles: List of duplicate files\nData leakage seriously undermines evaluation reliability.\nIf duplicates are found, remove them from one split.', ko: '분할 간 중복 탐지 결과입니다.\nSplit Pair: 비교한 분할 쌍 (예: Train↔Val)\nDuplicates: 발견된 중복 수\nFiles: 중복 파일 목록\n데이터 누수(leakage)는 평가 신뢰도를 심각하게 훼손합니다.\n중복이 발견되면 반드시 한쪽에서 제거하세요.' } },
  ]},
],

/* ── 16. similarity ── */
'similarity': [
  { page: { en: '📋 Tab Summary', ko: '📋 탭 요약' }, items: [
    { sel: '#page-title', text: { en: '🔎 Similarity Search Tab\nSearch for the Top-K most similar images\nto a query image using dHash.\nNo model required — runs fast.', ko: '🔎 유사 이미지 검색 탭\n쿼리 이미지와 가장 유사한 Top-K 이미지를\ndHash 기반으로 검색하는 탭입니다.\n모델 없이 빠르게 동작합니다.' } },
  ]},
  { page: { en: '🔎 Similarity Search Settings', ko: '🔎 유사 검색 설정' }, items: [
    { sel: '#sim-img', text: { en: 'Specify the index (search target) image folder.\nAll images in this folder are searched for similarity.', ko: '검색 대상(인덱스) 이미지 폴더를 지정합니다.\n이 폴더의 모든 이미지에서 유사한 것을 찾습니다.' } },
    { sel: '#sim-query', text: { en: 'Select the query image.\nThe most similar images to this one will be retrieved.', ko: '쿼리 이미지를 선택합니다.\n이 이미지와 가장 유사한 이미지를 검색합니다.' } },
    { sel: '#sim-k', text: { en: 'Number of top results to return.\nExample: 10 shows the 10 most similar images.', ko: '반환할 상위 결과 수입니다.\n예: 10이면 가장 유사한 10개를 표시합니다.' } },
    { sel: '#sim-prog', text: { en: 'Similarity computation progress.', ko: '유사도 계산 진행률입니다.' } },
    { sel: '#sim-results', text: { en: 'Similarity search results.\nRank: Similarity rank (1 = most similar)\nImage: Retrieved image filename\nDistance: dHash Hamming distance (lower = more similar)\nRuns fast using dHash without requiring a model.', ko: '유사 이미지 검색 결과입니다.\nRank: 유사도 순위 (1이 가장 유사)\nImage: 검색된 이미지 파일명\nDistance: dHash Hamming 거리 (낮을수록 유사)\n모델 없이 dHash 기반으로 동작하므로 빠릅니다.' } },
  ]},
],

/* ── 17. batch ── */
'batch': [
  { page: { en: '📋 Tab Summary', ko: '📋 탭 요약' }, items: [
    { sel: '#page-title', text: { en: '📦 Batch Inference Tab\nRun a model on all images in a folder\nand export results as YOLO txt, JSON,\nor CSV format.', ko: '📦 일괄 추론 탭\n폴더 내 모든 이미지에 모델을 적용하여\nYOLO txt, JSON, CSV 형식으로\n결과를 내보내는 탭입니다.' } },
  ]},
  { page: { en: '📦 Batch Inference Settings', ko: '📦 일괄 추론 설정' }, items: [
    { sel: '#bat-model', text: { en: 'Select the ONNX model for inference.\nSupports detection models; applied to all images in the folder.', ko: '추론에 사용할 ONNX 모델을 선택합니다.\n탐지 모델을 지원하며, 폴더 내 모든 이미지에 적용됩니다.' } },
    { sel: '#bat-img', text: { en: 'Specify the image folder for inference.\nSupports common formats like JPG and PNG.', ko: '추론할 이미지 폴더를 지정합니다.\nJPG, PNG 등 일반 이미지 형식을 지원합니다.' } },
    { sel: '#bat-out', text: { en: 'Output folder for inference results.', ko: '추론 결과가 저장될 출력 폴더입니다.' } },
    { sel: '#bat-fmt', text: { en: 'Select the output format.\nYOLO txt: Per-image .txt (class cx cy w h)\nJSON: All results in a single JSON file\nCSV: Spreadsheet-compatible format', ko: '출력 형식을 선택합니다.\nYOLO txt: 이미지별 .txt (class cx cy w h)\nJSON: 전체 결과를 하나의 JSON 파일로\nCSV: 스프레드시트 호환 형식' } },
    { sel: '#bat-vis', text: { en: 'When checked, saves images with bounding boxes drawn.\nUseful for reviewing results but uses more disk space.', ko: '체크하면 바운딩 박스가 그려진 이미지를 함께 저장합니다.\n결과 확인에 유용하지만 디스크 공간을 더 사용합니다.' } },
    { sel: '#bat-prog', text: { en: 'Batch inference progress.', ko: '일괄 추론 진행률입니다.' } },
  ]},
],

/* ── 18. augmentation ── */
'augmentation': [
  { page: { en: '📋 Tab Summary', ko: '📋 탭 요약' }, items: [
    { sel: '#page-title', text: { en: '🎨 Augmentation Preview Tab\nPreview data augmentations like Mosaic, Flip,\nRotate, and Brightness before applying them.', ko: '🎨 증강 미리보기 탭\nMosaic, Flip, Rotate, Brightness 등\n데이터 증강을 적용 전에 미리보기로\n확인하는 탭입니다.' } },
  ]},
  { page: { en: '🎨 Augmentation Settings', ko: '🎨 증강 설정' }, items: [
    { sel: '#aug-img', text: { en: 'Specify the source image folder for augmentation.', ko: '증강할 원본 이미지 폴더를 지정합니다.' } },
    { sel: '#aug-lbl', text: { en: 'Specify the label folder (optional).\nYOLO format (.txt); coordinates are transformed along with the image.\nBounding boxes are automatically adjusted for Flip/Rotate.', ko: '라벨 폴더를 지정합니다 (선택사항).\nYOLO 형식(.txt)이며, 증강 시 좌표도 함께 변환됩니다.\nFlip/Rotate 시 바운딩 박스가 자동으로 조정됩니다.' } },
    { sel: '#aug-type', text: { en: 'Select the augmentation type.\nMosaic 2×2: Combine 4 images into one\nFlip: Horizontal/vertical flip\nRotate: Rotation (90°/180°/270°)\nBrightness: Brightness adjustment\nPreview the result before applying.', ko: '증강 유형을 선택합니다.\nMosaic 2×2: 4장을 합쳐 하나의 이미지 생성\nFlip: 좌우/상하 반전\nRotate: 회전 (90°/180°/270°)\nBrightness: 밝기 조절\n적용 전 미리보기로 결과를 확인하세요.' } },
    { sel: '#aug-orig', text: { en: 'Original image preview.\nView the image before augmentation.', ko: '원본 이미지 미리보기입니다.\n증강 전 상태를 확인할 수 있습니다.' } },
    { sel: '#aug-result', text: { en: 'Augmented result preview.\nReview the result before applying;\nif satisfied, press the save button to apply.', ko: '증강 결과 미리보기입니다.\n실제 적용 전에 결과를 확인하고,\n만족스러우면 저장 버튼을 눌러 적용하세요.' } },
  ]},
],

};