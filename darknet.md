# ARGOS_JIHACHUL_v4.8 — DarkCenterNet Inference Pipeline

## Model Specification

| Item | Value |
|------|-------|
| Architecture | DarkCenterNet (CenterNet variant) |
| Format | ONNX |
| Size | ~19.5 MB |
| Input | `(1, 3, 736, 1280)` float32, [0,1] normalized |
| Output | `(1, 100, 13)` — max 100 detections, 13 attributes each |
| NMS | Not required (CenterNet is NMS-free by design) |

## Classes

| ID | Name |
|----|------|
| 0 | person |
| 1 | face |
| 2 | red-sign |
| 3 | wheelchair |
| 4 | cane |

## Output Tensor Layout (13 columns)

| Index | Name | Type | Range | Description |
|-------|------|------|-------|-------------|
| 0 | tlx | bbox | [0, 1] | Top-left X (normalized) |
| 1 | tly | bbox | [0, 1] | Top-left Y (normalized) |
| 2 | w | bbox | [0, 1] | Width (normalized) |
| 3 | h | bbox | [0, 1] | Height (normalized) |
| 4 | conf | score | [0, 1] | Detection confidence |
| 5 | class_id | int | {0,1,2,3,4} | Object class |
| 6 | fall | **multi-label** | [0, 1] | 넘어짐 |
| 7 | crawl | **multi-label** | [0, 1] | 기어감 |
| 8 | jump | **multi-label** | [0, 1] | 점프 |
| 9 | front | **multi-label** | [0, 1] | 정면 방향 |
| 10 | back | **multi-label** | [0, 1] | 후면 방향 |
| 11 | side | **multi-label** | [0, 1] | 측면 방향 |
| 12 | no-mask | **multi-label** | [0, 1] | 마스크 미착용 |

### Multi-Label 특성

col[6:13]은 **상호 배타적이지 않은 독립적 이진 분류** (sigmoid 출력)입니다.

- `argmax`를 사용하면 안 됨 — 여러 속성이 동시에 활성화될 수 있음
- 각 속성별 threshold (기본 0.3)로 개별 판단
- 예: person이 `side=0.41, no-mask=0.67`이면 → "측면 방향 + 마스크 미착용"

실제 관측된 값 범위:
- `fall/crawl/jump`: 대부분 매우 낮음 (< 0.05), 해당 행동 시에만 활성화
- `front/back/side`: 방향 속성, 하나가 높으면 나머지는 낮음 (soft exclusive)
- `no-mask`: 기본값이 높음 (0.3~0.99), 마스크 착용 시 낮아짐

## Preprocessing Pipeline

```
원본 프레임 (H_orig, W_orig, 3) BGR
    │
    ├─ cv2.resize(frame, (1280, 736))    ← 단순 resize, letterbox 없음!
    ├─ BGR → RGB                          ← [..., ::-1]
    ├─ HWC → CHW                          ← .transpose(2, 0, 1)
    ├─ float32 / 255.0                    ← [0, 1] 정규화
    │
    └─→ (1, 3, 736, 1280) float32
```

**핵심 차이점 (vs YOLO):**
- Letterbox를 사용하지 않음 — 단순 resize로 aspect ratio가 변형됨
- 이는 CenterNet 아키텍처의 특성 (heatmap 기반이라 aspect ratio 변형에 강건)

## Postprocessing Pipeline

```
모델 출력 (1, 100, 13)
    │
    ├─ confidence 필터링: col[4] > threshold
    ├─ 좌표 변환 (정규화 → 원본 픽셀):
    │     x1 = col[0] * orig_width
    │     y1 = col[1] * orig_height
    │     x2 = (col[0] + col[2]) * orig_width
    │     y2 = (col[1] + col[3]) * orig_height
    ├─ class_id 추출: col[5]
    ├─ multi-label 속성 추출: col[6:13]
    │
    └─→ boxes (xyxy pixels), scores, class_ids, attrs (N, 7)
```

**NMS 불필요:** CenterNet은 heatmap peak detection 기반이라 구조적으로 중복 검출이 발생하지 않음.

## Usage

```bash
# 기본 실행 (화면 표시)
python darknet.py --video Videos/stroller_01.mp4

# 결과 비디오 저장
python darknet.py --video Videos/wheelchair_01.mp4 --output result.mp4 --no-show

# Confidence / Attribute threshold 조절
python darknet.py --video Videos/no_mask_01.mp4 --conf 0.3 --attr-thresh 0.4

# 다른 모델 사용
python darknet.py --model Models/other_darknet.onnx --video input.mp4
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` / `ESC` | 종료 |

## Dependencies

```
pip install onnxruntime-gpu opencv-python numpy
```

GPU 없이 CPU만 사용할 경우:
```
pip install onnxruntime opencv-python numpy
```

## Integration with ssook

ssook에서 이 모델을 사용하려면:
1. Settings → Model Type: **CenterNet (Darknet)** 선택
2. Model path: `Models/ARGOS_JIHACHUL_v4.8.onnx` 지정
3. Viewer 탭에서 비디오 로드 후 Play

ssook 내부적으로 `core/inference.py`의 `preprocess_darknet()` + `postprocess_darknet()`이 동일한 파이프라인을 수행합니다.
