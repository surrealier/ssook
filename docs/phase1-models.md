# ssook — Phase 1 ONNX Model Download Guide

> Phase 1에서 추가된 태스크별 추천 ONNX 모델과 다운로드 링크입니다.

---

## 1. Pose Estimation

| Model | Size | Keypoints | Download |
|-------|------|-----------|----------|
| **YOLOv8n-pose** | ~6 MB | 17 (COCO) | [GitHub](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.onnx) |
| **YOLOv8s-pose** | ~23 MB | 17 (COCO) | [GitHub](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-pose.onnx) |
| **YOLOv8m-pose** | ~52 MB | 17 (COCO) | [GitHub](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-pose.onnx) |
| **YOLO11n-pose** | ~6 MB | 17 (COCO) | [GitHub](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.onnx) |
| **YOLO11s-pose** | ~19 MB | 17 (COCO) | [GitHub](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.onnx) |

**사용법**: ssook → Pose Estimation 탭 → Model Type: `YOLO-Pose (v8/v11)` 선택 → 모델 로드

**Export (직접 변환)**:
```bash
pip install ultralytics
yolo export model=yolov8n-pose.pt format=onnx imgsz=640
```

---

## 2. Instance Segmentation

| Model | Size | Classes | Download |
|-------|------|---------|----------|
| **YOLOv8n-seg** | ~7 MB | 80 (COCO) | [GitHub](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.onnx) |
| **YOLOv8s-seg** | ~24 MB | 80 (COCO) | [GitHub](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-seg.onnx) |
| **YOLOv8m-seg** | ~53 MB | 80 (COCO) | [GitHub](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-seg.onnx) |
| **YOLO11n-seg** | ~6 MB | 80 (COCO) | [GitHub](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.onnx) |
| **YOLO11s-seg** | ~19 MB | 80 (COCO) | [GitHub](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.onnx) |

**사용법**: ssook → Instance Seg 탭 → Model Type: `YOLO-Seg Instance (v8/v11)` 선택

**Export**:
```bash
yolo export model=yolov8n-seg.pt format=onnx imgsz=640
```

---

## 3. Object Tracking

트래킹은 별도 ONNX 모델이 필요하지 않습니다. 기존 Detection 모델 + ByteTrack/SORT 트래커 조합으로 동작합니다.

**추천 Detection 모델 (트래킹용)**:

| Model | Size | FPS (CPU) | Download |
|-------|------|-----------|----------|
| **YOLOv8n** | ~6 MB | ~45 | [GitHub](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx) |
| **YOLOv8s** | ~22 MB | ~30 | [GitHub](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.onnx) |
| **YOLO11n** | ~5 MB | ~50 | [GitHub](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx) |
| **YOLO11s** | ~18 MB | ~35 | [GitHub](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.onnx) |

**사용법**: ssook → Tracking 탭 → ByteTrack 또는 SORT 선택 → 트래커 생성

---

## 4. VLM (Vision-Language Model)

| Model | Size | Task | Download |
|-------|------|------|----------|
| **CLIP ViT-B/32 (Image)** | ~350 MB | Zero-shot | [HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/onnx/model.onnx) |
| **CLIP ViT-B/32 (Text)** | ~250 MB | Zero-shot | [HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/onnx/model.onnx) |
| **BLIP-2 (ONNX)** | ~1.5 GB | VQA/Caption | [HuggingFace](https://huggingface.co/Xenova/blip-image-captioning-base/tree/main/onnx) |
| **GIT-base (ONNX)** | ~700 MB | Captioning | [HuggingFace](https://huggingface.co/Xenova/git-base/tree/main/onnx) |

> VLM 기능은 현재 기본 프레임워크가 구현되어 있으며, 향후 업데이트에서 완전한 VQA/Captioning 파이프라인이 추가됩니다.

---

## 5. ONNX Model Inspector / Profiler

인스펙터와 프로파일러는 **모든 ONNX 모델**에서 동작합니다. 별도 모델 다운로드가 필요하지 않습니다.

**테스트용 추천 모델**:

| Model | Size | Purpose | Download |
|-------|------|---------|----------|
| **YOLOv8n** | ~6 MB | 빠른 테스트 | [GitHub](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx) |
| **ResNet-50** | ~98 MB | Classification | [ONNX Zoo](https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx) |
| **MobileNetV2** | ~14 MB | 경량 모델 | [ONNX Zoo](https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx) |
| **EfficientNet-Lite4** | ~50 MB | 효율 모델 | [ONNX Zoo](https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx) |

---

## Quick Download Script

```bash
# 모델 디렉토리 생성
mkdir -p Models

# Pose Estimation
wget -O Models/yolov8n-pose.onnx https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.onnx
wget -O Models/yolov8s-pose.onnx https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-pose.onnx

# Instance Segmentation
wget -O Models/yolov8n-seg.onnx https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.onnx
wget -O Models/yolov8s-seg.onnx https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-seg.onnx

# Detection (for Tracking)
wget -O Models/yolov8n.onnx https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx

# Inspector/Profiler test
wget -O Models/resnet50-v2-7.onnx https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx
```

---

## 참고

- 모든 모델은 `Models/` 디렉토리에 저장하면 ssook에서 자동으로 인식합니다.
- GPU 가속이 필요한 경우 `pip install onnxruntime-gpu`를 설치하세요.
- 모델 변환은 `ultralytics` 패키지의 `yolo export` 명령을 사용하세요.
