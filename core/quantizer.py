"""ONNX Model Quantizer — Dynamic / Static INT8 / FP16 conversion."""
import glob
import os

import cv2
import numpy as np
import onnxruntime as ort

try:
    from onnxruntime.quantization import CalibrationDataReader as _CalibBase
except ImportError:
    _CalibBase = object


def _get_input_meta(model_path: str):
    """Return list of (name, shape, dtype_str) for model inputs."""
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    meta = []
    for inp in sess.get_inputs():
        shape = []
        for d in inp.shape:
            shape.append(d if isinstance(d, int) and d > 0 else None)
        meta.append({"name": inp.name, "shape": shape, "type": inp.type})
    return meta


def _resolve_shape(shape, default_hw=640):
    """Resolve dynamic dims to concrete values for calibration."""
    resolved = []
    ndim = len(shape)
    for i, d in enumerate(shape):
        if d is not None and d > 0:
            resolved.append(d)
        elif ndim == 4:
            resolved.append([1, 3, default_hw, default_hw][i])
        elif ndim == 3:
            resolved.append([1, 64, 256][min(i, 2)])
        else:
            resolved.append(1 if i == 0 else 256)
    return resolved


class _AutoCalibrationReader(_CalibBase):
    """CalibrationDataReader that auto-preprocesses images to match model input."""

    def __init__(self, model_path: str, image_dir: str, max_images: int = 100):
        self.meta = _get_input_meta(model_path)
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.images = []
        for e in exts:
            self.images.extend(glob.glob(os.path.join(image_dir, e)))
        self.images = sorted(self.images)[:max_images]
        self.index = 0

        inp = self.meta[0]
        self.input_name = inp["name"]
        self.shape = _resolve_shape(inp["shape"])
        self.is_float16 = "float16" in (inp["type"] or "")

    def get_next(self):
        if self.index >= len(self.images):
            return None
        img = cv2.imread(self.images[self.index])
        if img is None:
            self.index += 1
            return self.get_next()
        h, w = self.shape[2], self.shape[3]
        img = cv2.resize(img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        batch = self.shape[0]
        if batch > 1:
            tensor = np.repeat(tensor, batch, axis=0)
        if self.is_float16:
            tensor = tensor.astype(np.float16)
        self.index += 1
        return {self.input_name: tensor}

    def rewind(self):
        self.index = 0


def quantize_dynamic(model_path: str, output_path: str,
                     weight_type: str = "uint8") -> dict:
    """Dynamic quantization — no calibration data needed."""
    from onnxruntime.quantization import quantize_dynamic as _qd, QuantType

    wt = QuantType.QUInt8 if weight_type == "uint8" else QuantType.QInt8
    _qd(model_path, output_path, weight_type=wt)

    orig_size = os.path.getsize(model_path)
    new_size = os.path.getsize(output_path)
    return {
        "method": "dynamic",
        "weight_type": weight_type,
        "original_size_mb": round(orig_size / 1024 / 1024, 2),
        "quantized_size_mb": round(new_size / 1024 / 1024, 2),
        "compression_ratio": round(orig_size / max(new_size, 1), 2),
        "output_path": output_path,
    }


def quantize_static(model_path: str, output_path: str,
                    calibration_dir: str, max_images: int = 100,
                    per_channel: bool = True,
                    weight_type: str = "int8",
                    activation_type: str = "uint8",
                    quant_format: str = "QDQ",
                    on_progress=None) -> dict:
    """Static quantization with calibration data."""
    from onnxruntime.quantization import (
        quantize_static as _qs, QuantType, QuantFormat,
    )

    reader = _AutoCalibrationReader(model_path, calibration_dir, max_images)
    total = len(reader.images)
    if total == 0:
        raise ValueError("No calibration images found in the directory")

    # Wrap reader to report progress
    orig_get_next = reader.get_next
    def _tracked_get_next():
        result = orig_get_next()
        if on_progress and result is not None:
            on_progress(reader.index, total)
        return result
    reader.get_next = _tracked_get_next

    wt = QuantType.QInt8 if weight_type == "int8" else QuantType.QUInt8
    at = QuantType.QUInt8 if activation_type == "uint8" else QuantType.QInt8
    qf = QuantFormat.QDQ if quant_format == "QDQ" else QuantFormat.QOperator

    _qs(model_path, output_path,
        calibration_data_reader=reader,
        quant_format=qf,
        per_channel=per_channel,
        weight_type=wt,
        activation_type=at)

    orig_size = os.path.getsize(model_path)
    new_size = os.path.getsize(output_path)
    return {
        "method": "static",
        "weight_type": weight_type,
        "activation_type": activation_type,
        "quant_format": quant_format,
        "per_channel": per_channel,
        "calibration_images": total,
        "original_size_mb": round(orig_size / 1024 / 1024, 2),
        "quantized_size_mb": round(new_size / 1024 / 1024, 2),
        "compression_ratio": round(orig_size / max(new_size, 1), 2),
        "output_path": output_path,
    }


def convert_fp16(model_path: str, output_path: str) -> dict:
    """Convert FP32 model to FP16."""
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(model_path)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, output_path)

    orig_size = os.path.getsize(model_path)
    new_size = os.path.getsize(output_path)
    return {
        "method": "fp16",
        "original_size_mb": round(orig_size / 1024 / 1024, 2),
        "quantized_size_mb": round(new_size / 1024 / 1024, 2),
        "compression_ratio": round(orig_size / max(new_size, 1), 2),
        "output_path": output_path,
    }
