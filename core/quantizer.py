"""ONNX Model Quantizer — Dynamic / Static INT8 / FP16 conversion."""
import glob
import logging
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

_log = logging.getLogger(__name__)

# Below this agreement an optimized model is assumed to have lost meaningful
# accuracy and is flagged degraded so the caller can warn / discard. 0.9 is a
# deliberately loose default: destructive INT8/pruning should clear it easily on
# a behaving model, while a wrecked one (random-ish outputs) falls well under.
DEFAULT_AGREEMENT_THRESHOLD = 0.9

try:
    from onnxruntime.quantization import CalibrationDataReader as _CalibBase
except ImportError:
    _CalibBase = object


@dataclass
class PreprocessSpec:
    """Explicit calibration preprocessing, mirroring the deployment pipeline.

    Static INT8 ranges are only valid if calibration sees the SAME input
    distribution the model is fed at inference. The auto-guess (letterbox iff
    H==W, RGB, plain /255) is wrong for classifier/embedder pipelines (BGR,
    plain resize, ImageNet mean/std), so callers can pass this to pin it down.

    mean/std are applied in the [0,1] tensor space, per channel, AFTER the
    optional /255 scale: tensor = (img/scale - mean) / std. Defaults reproduce
    the historical behaviour (plain /255, no mean/std shift).
    """

    use_letterbox: bool = True
    color_order: str = "rgb"  # "rgb" or "bgr"
    scale: float = 255.0  # divide pixel values by this (255 -> [0,1]); 1.0 to skip
    mean: tuple = (0.0, 0.0, 0.0)  # per-channel mean in scaled space
    std: tuple = (1.0, 1.0, 1.0)  # per-channel std in scaled space
    layout: str = "nchw"  # "nchw" or "nhwc"

    def as_dict(self) -> dict:
        return {
            "use_letterbox": self.use_letterbox,
            "color_order": self.color_order,
            "scale": self.scale,
            "mean": list(self.mean),
            "std": list(self.std),
            "layout": self.layout,
        }


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
    """CalibrationDataReader that preprocesses images to match model input.

    Calibration must mirror deployment preprocessing or the INT8 ranges are
    calibrated on a different input distribution and accuracy silently drops.
    A caller can pass an explicit `spec` (PreprocessSpec) to pin the pipeline;
    when omitted, the reader auto-guesses (letterbox for square detection
    inputs, RGB, /255) and records the assumptions it made in `.spec` so the
    result can report them — the choice is never silent.
    """

    def __init__(self, model_path: str, image_dir: str, max_images: int = 100,
                 use_letterbox: Optional[bool] = None,
                 spec: Optional[PreprocessSpec] = None):
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

        if spec is not None:
            # Explicit override wins; a use_letterbox kwarg is ignored so the
            # spec is the single source of truth.
            self.spec = spec
            self.is_auto_guessed = False
        else:
            # Auto-guess: letterbox for square detection inputs (H == W), plain
            # resize otherwise. An explicit use_letterbox flag overrides only
            # the letterbox heuristic; everything else stays at the historical
            # default (RGB, /255, NCHW).
            h, w = self.shape[2], self.shape[3]
            guessed_letterbox = (h == w) if use_letterbox is None else use_letterbox
            self.spec = PreprocessSpec(use_letterbox=guessed_letterbox)
            self.is_auto_guessed = True

        self.mean = np.asarray(self.spec.mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(self.spec.std, dtype=np.float32).reshape(3, 1, 1)

    def get_next(self):
        if self.index >= len(self.images):
            return None
        img = cv2.imread(self.images[self.index])
        if img is None:
            self.index += 1
            return self.get_next()
        h, w = self.shape[2], self.shape[3]
        if self.spec.use_letterbox:
            # Reuse deployment letterbox so calibration matches inference.
            from core.inference import letterbox
            img, _ratio, _pad = letterbox(img, (h, w))
        else:
            img = cv2.resize(img, (w, h))
        if self.spec.color_order == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # CHW float, scaled then normalized: (img/scale - mean) / std.
        tensor = img.transpose(2, 0, 1).astype(np.float32)
        if self.spec.scale != 1.0:
            tensor = tensor / self.spec.scale
        tensor = (tensor - self.mean) / self.std
        tensor = np.expand_dims(tensor, axis=0)
        if self.spec.layout == "nhwc":
            tensor = tensor.transpose(0, 2, 3, 1)
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
                    use_letterbox: Optional[bool] = None,
                    preprocess: Optional[PreprocessSpec] = None,
                    on_progress=None) -> dict:
    """Static quantization with calibration data.

    `preprocess` (PreprocessSpec) pins the calibration pipeline for non-default
    deployments (BGR, plain resize, mean/std). When omitted, the reader
    auto-guesses and the returned dict reports the assumptions used under
    `preprocess` / `preprocess_auto_guessed`, so a miscalibration is visible.
    `use_letterbox` is kept for backward compatibility and only nudges the
    auto-guess; an explicit `preprocess` overrides it.
    """
    from onnxruntime.quantization import (
        quantize_static as _qs, QuantType, QuantFormat,
    )

    reader = _AutoCalibrationReader(model_path, calibration_dir, max_images,
                                    use_letterbox=use_letterbox,
                                    spec=preprocess)
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
        # Surface the preprocessing actually used so a wrong auto-guess is not
        # silent — the UI/result body can show it for confirmation.
        "preprocess": reader.spec.as_dict(),
        "preprocess_auto_guessed": reader.is_auto_guessed,
        "original_size_mb": round(orig_size / 1024 / 1024, 2),
        "quantized_size_mb": round(new_size / 1024 / 1024, 2),
        "compression_ratio": round(orig_size / max(new_size, 1), 2),
        "output_path": output_path,
    }


# Above this source size we always round-trip through external data so the
# protobuf 2GB serialize limit can't abort a large fp32 -> fp16 conversion.
_EXTERNAL_DATA_THRESHOLD_BYTES = 1.8 * 1024 * 1024 * 1024


def _load_float16_converter():
    """Return convert_float_to_float16, preferring the documented package.

    onnxconverter_common is the canonical dependency (see core/deployment
    targets), but it is optional and absent in some envs. onnxruntime ships an
    equivalent converter with the same keep_io_types contract, so fall back to
    it rather than failing — both keep model I/O in fp32 when asked.
    """
    try:
        from onnxconverter_common import float16
        return float16.convert_float_to_float16
    except ImportError:
        from onnxruntime.transformers import float16
        return float16.convert_float_to_float16


def convert_fp16(model_path: str, output_path: str) -> dict:
    """Convert an FP32 model's weights/compute to FP16, keeping FP32 I/O.

    keep_io_types=True leaves the graph inputs/outputs as float32 so every
    downstream consumer (benchmark hand-off, profiler, eval) can keep feeding
    fp32 uint8/255 tensors — converting the I/O would break them with a type
    mismatch. External data is loaded and re-emitted as a sidecar for large
    models so the protobuf 2GB serialize limit doesn't abort the conversion.
    """
    import onnx

    convert = _load_float16_converter()

    # Load weights inline so the converter can rewrite initializers; ONNX
    # resolves the sidecar relative to the model's directory automatically.
    model = onnx.load(model_path, load_external_data=True)
    model_fp16 = convert(model, keep_io_types=True)

    # Decide whether to emit external data. fp16 roughly halves weight bytes,
    # but a source already near/over the 2GB protobuf limit (or with existing
    # external data) must stay external to serialize at all.
    orig_size = os.path.getsize(model_path)
    has_external = any(
        ext.key == "location"
        for init in model_fp16.graph.initializer
        for ext in init.external_data
    )
    use_external = has_external or orig_size >= _EXTERNAL_DATA_THRESHOLD_BYTES

    if use_external:
        data_name = os.path.basename(output_path) + ".data"
        # Drop any stale sidecar so save doesn't append to a previous run.
        stale = os.path.join(os.path.dirname(output_path) or ".", data_name)
        if os.path.exists(stale):
            os.remove(stale)
        onnx.save(model_fp16, output_path, save_as_external_data=True,
                  all_tensors_to_one_file=True, location=data_name)
    else:
        onnx.save(model_fp16, output_path)

    orig_size = os.path.getsize(model_path)
    new_size = os.path.getsize(output_path)
    if use_external:
        sidecar = os.path.join(os.path.dirname(output_path) or ".",
                               os.path.basename(output_path) + ".data")
        if os.path.exists(sidecar):
            new_size += os.path.getsize(sidecar)
    return {
        "method": "fp16",
        "external_data": use_external,
        "original_size_mb": round(orig_size / 1024 / 1024, 2),
        "quantized_size_mb": round(new_size / 1024 / 1024, 2),
        "compression_ratio": round(orig_size / max(new_size, 1), 2),
        "output_path": output_path,
    }


# ── Post-optimization accuracy validation ───────────────
# Destructive optimizations (INT8 quant, pruning) can silently wreck a model:
# the artifact still loads and runs, but its outputs diverge from the original.
# Without ground-truth labels we can't measure true accuracy, so we measure
# *agreement* — how closely the optimized model reproduces the original model's
# own outputs on the same inputs. High agreement => the optimization preserved
# behaviour; low agreement => it changed what the model does, regardless of
# whether the original was "correct". This is a generic proxy that works for any
# task (detection, classification, embedding) without GT.


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two flattened output tensors, in [-1, 1].

    Zero-norm vectors (all-zero outputs) are treated as perfectly agreeing when
    both are zero, else as fully divergent — avoids a 0/0 NaN that would poison
    the mean.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    # Differing element counts (e.g. a pruned model changed an output's size)
    # can't be aligned for a dot product — that is itself full divergence.
    if a.shape != b.shape:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == 0.0 and nb == 0.0 else 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_agreement_metrics(orig_outputs: list, opt_outputs: list) -> dict:
    """Compare one inference's outputs from the original vs optimized model.

    Pure numpy — no onnxruntime — so the metric math is unit-testable on plain
    arrays. Both inputs are lists of output tensors (the value `session.run`
    returns). Returns per-sample stats:
      - cosine: mean cosine similarity across the model's output tensors
        (a generic numeric 'how aligned are the raw outputs' in [-1, 1]).
      - max_abs_diff: largest absolute elementwise difference across outputs
        (catches blow-ups even when cosine stays high).
      - top1_match: True iff argmax of the first output agrees — meaningful for
        classifiers, harmless (just one signal) otherwise.
    """
    if not orig_outputs or not opt_outputs:
        return {"cosine": 0.0, "max_abs_diff": float("inf"), "top1_match": False}

    n = min(len(orig_outputs), len(opt_outputs))
    cosines: list[float] = []
    max_abs = 0.0
    for i in range(n):
        a = np.asarray(orig_outputs[i], dtype=np.float64)
        b = np.asarray(opt_outputs[i], dtype=np.float64)
        cosines.append(_cosine_similarity(a, b))
        if a.shape == b.shape and a.size:
            max_abs = max(max_abs, float(np.max(np.abs(a - b))))
        else:
            # Shape divergence is itself a strong degradation signal.
            max_abs = float("inf")

    a0 = np.asarray(orig_outputs[0]).ravel()
    b0 = np.asarray(opt_outputs[0]).ravel()
    top1_match = bool(a0.size and b0.size and int(np.argmax(a0)) == int(np.argmax(b0)))

    return {
        "cosine": float(np.mean(cosines)) if cosines else 0.0,
        "max_abs_diff": max_abs,
        "top1_match": top1_match,
    }


def _calibration_image_paths(image_dir: str, max_images: int) -> list[str]:
    """Collect up to `max_images` image paths, mirroring the calib reader."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths: list[str] = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(image_dir, e)))
    return sorted(paths)[:max_images]


def validate_optimization(original_path: str, optimized_path: str,
                          sample_dir: str, max_images: int = 16,
                          threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
                          preprocess: Optional[PreprocessSpec] = None,
                          use_letterbox: Optional[bool] = None) -> dict:
    """Run original + optimized ONNX on the SAME N images and score agreement.

    Both models are run under onnxruntime CPU on identical preprocessed inputs
    (reusing `_AutoCalibrationReader` so the feed matches the deployment/calib
    pipeline). The per-image agreement is averaged into a single `agreement` in
    [0, 1] (cosine remapped from [-1, 1] so a "% similar" reads naturally). When
    `agreement < threshold` the result is marked `degraded` — the caller should
    warn or discard rather than silently presenting the optimized model as good.
    The original artifact is never touched, so its path stays available.

    Returns: {validated, agreement, threshold, degraded, n_images,
              max_abs_diff, top1_agreement, original_path}. On any failure this
      returns {validated: False, error: ...} — validation must never block or
      mask the optimization itself; the caller catches/reports.
    """
    image_paths = _calibration_image_paths(sample_dir, max_images)
    if not image_paths:
        return {"validated": False, "error": "No sample images found",
                "n_images": 0, "threshold": threshold,
                "original_path": original_path}

    # Reuse the calibration reader to produce inputs that match how the model is
    # actually fed; both sessions consume the exact same tensors.
    reader = _AutoCalibrationReader(original_path, sample_dir, max_images,
                                    use_letterbox=use_letterbox, spec=preprocess)

    sess_opts = ort.SessionOptions()
    orig_sess = ort.InferenceSession(original_path, sess_opts,
                                     providers=["CPUExecutionProvider"])
    opt_sess = ort.InferenceSession(optimized_path, sess_opts,
                                    providers=["CPUExecutionProvider"])

    agreements: list[float] = []
    max_abs_diff = 0.0
    top1_matches = 0
    n_scored = 0

    reader.rewind()
    while True:
        feed = reader.get_next()
        if feed is None:
            break
        try:
            orig_out = orig_sess.run(None, feed)
            # The optimized model may expect a different input dtype (e.g. an
            # fp16 graph keeps fp32 I/O, but a re-typed input would differ);
            # feed by the optimized model's own input name to stay robust.
            opt_feed = {opt_sess.get_inputs()[0].name: list(feed.values())[0]}
            opt_out = opt_sess.run(None, opt_feed)
        except Exception as run_err:  # noqa: BLE001 — per-image failure is non-fatal
            _log.warning("validate_optimization: inference failed on a sample: %s", run_err)
            continue

        m = compute_agreement_metrics(orig_out, opt_out)
        # Remap cosine [-1, 1] -> [0, 1] so 'agreement' reads as a similarity %.
        agreements.append((m["cosine"] + 1.0) / 2.0)
        if m["max_abs_diff"] != float("inf"):
            max_abs_diff = max(max_abs_diff, m["max_abs_diff"])
        else:
            max_abs_diff = float("inf")
        if m["top1_match"]:
            top1_matches += 1
        n_scored += 1

    if n_scored == 0:
        return {"validated": False, "error": "No samples could be scored",
                "n_images": 0, "threshold": threshold,
                "original_path": original_path}

    agreement = float(np.mean(agreements))
    degraded = agreement < threshold
    if degraded:
        _log.warning(
            "validate_optimization: agreement %.4f < threshold %.4f over %d images "
            "— optimized model may be degraded (%s)",
            agreement, threshold, n_scored, optimized_path)
    return {
        "validated": True,
        "agreement": round(agreement, 4),
        "threshold": threshold,
        "degraded": degraded,
        "n_images": n_scored,
        "max_abs_diff": (None if max_abs_diff == float("inf")
                         else round(max_abs_diff, 6)),
        "top1_agreement": round(top1_matches / n_scored, 4),
        # Keep the untouched original available so the caller can fall back to it.
        "original_path": original_path,
    }
