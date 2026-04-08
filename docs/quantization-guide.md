# ONNX 모델 양자화 가이드

ssook에서 사용하는 ONNX 모델의 추론 속도를 개선하기 위한 양자화 방법을 안내합니다.
특히 **GPU가 없는 노트북 환경**에서 큰 효과를 볼 수 있습니다.

---

## 양자화란?

모델 가중치를 FP32 → INT8 또는 FP16으로 변환하여:
- 추론 속도 **2~4배 향상** (CPU 환경)
- 모델 파일 크기 **50~75% 감소**
- 메모리 사용량 대폭 감소

---

## 방법 1: 동적 양자화 (Dynamic Quantization)

가장 간단한 방법. 캘리브레이션 데이터 불필요.

```bash
pip install onnxruntime-tools
# 또는
pip install onnxruntime>=1.17.0
```

```python
import onnxruntime.quantization as quant

model_path = "your_model.onnx"
output_path = "your_model_int8.onnx"

quant.quantize_dynamic(
    model_input=model_path,
    model_output=output_path,
    weight_type=quant.QuantType.QUInt8,
)
print(f"양자화 완료: {output_path}")
```

**장점**: 간단, 캘리브레이션 불필요
**단점**: 정적 양자화 대비 정확도/속도 이점 적음

---

## 방법 2: 정적 양자화 (Static Quantization)

캘리브레이션 데이터를 사용하여 더 정확한 양자화 수행.

```python
import numpy as np
import cv2
import glob
import onnxruntime.quantization as quant
from onnxruntime.quantization import CalibrationDataReader

class YOLOCalibrationReader(CalibrationDataReader):
    """YOLO 모델용 캘리브레이션 데이터 리더"""

    def __init__(self, image_dir, input_size=640, max_images=100):
        self.images = glob.glob(f"{image_dir}/*.jpg")[:max_images]
        self.input_size = input_size
        self.index = 0

    def get_next(self):
        if self.index >= len(self.images):
            return None
        img = cv2.imread(self.images[self.index])
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        self.index += 1
        return {"images": tensor}  # input name에 맞게 수정

model_path = "your_model.onnx"
output_path = "your_model_int8_static.onnx"
calibration_dir = "path/to/calibration/images"

reader = YOLOCalibrationReader(calibration_dir)

quant.quantize_static(
    model_input=model_path,
    model_output=output_path,
    calibration_data_reader=reader,
    quant_format=quant.QuantFormat.QDQ,
    per_channel=True,
    weight_type=quant.QuantType.QInt8,
    activation_type=quant.QuantType.QUInt8,
)
print(f"정적 양자화 완료: {output_path}")
```

**장점**: 최고 속도, 정확도 손실 최소
**단점**: 캘리브레이션 이미지 필요 (50~200장 권장)

---

## 방법 3: FP16 변환

INT8보다 정확도 손실이 적고, GPU 환경에서 특히 효과적.

```python
import onnx
from onnxconverter_common import float16

model = onnx.load("your_model.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "your_model_fp16.onnx")
```

```bash
pip install onnxconverter-common
```

---

## 양자화 후 ssook에서 사용

양자화된 모델은 기존 ONNX 모델과 동일하게 사용 가능합니다.
ssook의 Settings에서 양자화된 `.onnx` 파일을 선택하면 됩니다.

---

## 성능 비교 예시 (CPU, i7-1165G7 노트북)

| 모델 | 크기 | 추론 시간 | mAP@50 |
|------|------|-----------|--------|
| YOLOv8n FP32 | 12.2 MB | ~45ms | 37.3 |
| YOLOv8n INT8 (동적) | 3.4 MB | ~25ms | 36.8 |
| YOLOv8n INT8 (정적) | 3.2 MB | ~18ms | 37.0 |
| YOLOv8n FP16 | 6.1 MB | ~38ms | 37.3 |

> 수치는 환경에 따라 다를 수 있습니다. 양자화 후 반드시 ssook의 Evaluation 탭에서 정확도를 검증하세요.

---

## 주의사항

1. **입력 이름 확인**: 캘리브레이션 리더의 딕셔너리 키가 모델의 입력 이름과 일치해야 합니다.
   ssook의 Inference Analysis 탭에서 모델 입력 이름을 확인할 수 있습니다.

2. **ONNX opset 버전**: opset 11 이상 권장. 낮은 버전은 양자화 실패 가능.

3. **정확도 검증**: 양자화 후 mAP가 1~2% 이상 하락하면 캘리브레이션 데이터를 늘리거나
   동적 양자화로 전환하세요.

4. **CenterNet/DETR 모델**: 일부 커스텀 연산이 INT8을 지원하지 않을 수 있습니다.
   이 경우 FP16 변환을 권장합니다.
