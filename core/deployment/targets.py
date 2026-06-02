"""Built-in export targets, keyed to the runtimes the goal calls out:
onnxruntime (CPU / CUDA / DirectML), TensorRT, OpenVINO, CoreML, plus
TorchScript. Importing this module registers them into the singleton
``registry``.

Source dispatch (per target):
* ``.pt`` checkpoint  -> Ultralytics exporter (opset/imgsz/batch/dynamic/half).
* ``.onnx``           -> precision via ``core.quantizer``; engine via TensorRT
                         API; IR via ``openvino.convert_model``.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
from typing import Callable, Optional

from core.deployment.registry import BaseExportTarget, registry
from core.deployment import exporters as _ex


def _have(module: str) -> bool:
    """Cheap availability probe — does the module exist without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _need_ultralytics_for_pt() -> tuple[bool, str]:
    if not _have("ultralytics"):
        return False, "PyTorch (.pt) source needs ultralytics (pip install ultralytics)"
    return True, ""


def _apply_onnx_precision_from_onnx(
    src_onnx: str,
    output_path: str,
    opts: dict,
    on_progress: Optional[Callable[[int, int, str], None]],
) -> dict:
    """Take an existing ONNX and write `output_path` at the requested precision."""
    precision = opts.get("precision", "fp32")
    if precision == "fp16":
        from core.quantizer import convert_fp16  # optional onnxconverter_common
        try:
            r = convert_fp16(src_onnx, output_path)
        except ImportError:
            # Match the module's friendly "needs X" convention instead of a raw import error.
            return {"error": "FP16 from an existing ONNX needs onnxconverter_common "
                             "(pip install onnxconverter-common)", "output_path": output_path}
        r.setdefault("output_path", output_path)
        return r
    if precision == "int8":
        from core.quantizer import quantize_dynamic, quantize_static
        calib = opts.get("calibration_dir")
        if calib:
            # quantize_static reports progress as on_progress(cur, tot) (2 args);
            # adapt our 3-arg (done, total, msg) callback to that contract.
            calib_cb = None
            if on_progress:
                def calib_cb(cur: int, tot: int) -> None:
                    on_progress(cur, tot, f"Calibrating {cur}/{tot}...")
            r = quantize_static(src_onnx, output_path, calib,
                                max_images=int(opts.get("max_images", 100)),
                                on_progress=calib_cb)
        else:
            r = quantize_dynamic(src_onnx, output_path)
        r.setdefault("output_path", output_path)
        return r
    # fp32 — copy through (opset re-spec on an existing ONNX is best-effort).
    if opts.get("opset"):
        try:
            import onnx
            from onnx import version_converter
            model = onnx.load(src_onnx)
            converted = version_converter.convert_version(model, int(opts["opset"]))
            onnx.save(converted, output_path)
            return {"output_path": output_path, "opset": int(opts["opset"])}
        except Exception:
            # version_converter is brittle for real graphs; fall back to copy.
            pass
    if os.path.abspath(src_onnx) != os.path.abspath(output_path):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        shutil.copy2(src_onnx, output_path)
    return {"output_path": output_path}


def _produce_onnx(
    model_path: str,
    output_path: str,
    opts: dict,
    on_progress: Optional[Callable[[int, int, str], None]],
) -> dict:
    """Produce an ONNX artifact at output_path from a .pt or .onnx source."""
    precision = opts.get("precision", "fp32")
    if _ex.is_torch_checkpoint(model_path):
        ok, reason = _need_ultralytics_for_pt()
        if not ok:
            return {"error": reason, "output_path": output_path}
        if on_progress:
            on_progress(0, 1, "Exporting ONNX from checkpoint (ultralytics)...")
        export_opts = dict(opts)
        # fp16 ONNX straight from the checkpoint avoids the onnxconverter_common
        # dependency; int8 always routes through our calibrating quantizer below.
        if precision == "fp16":
            export_opts["half"] = True
        produced = _ex.ultralytics_export(model_path, "onnx", export_opts)
        if precision == "int8":
            if on_progress:
                on_progress(0, 1, "Quantizing exported ONNX to INT8...")
            # finally so a quant failure still drops the interim fp32 export.
            try:
                result = _apply_onnx_precision_from_onnx(produced, output_path, opts, on_progress)
            finally:
                if os.path.abspath(produced) != os.path.abspath(output_path):
                    try:
                        os.remove(produced)
                    except OSError:
                        pass
            return result
        final = _ex.finalize_artifact(produced, output_path)
        return {"output_path": final}
    # ONNX source.
    if on_progress:
        on_progress(0, 1, f"Producing ONNX ({precision})...")
    return _apply_onnx_precision_from_onnx(model_path, output_path, opts, on_progress)


# ── ONNX Runtime family (CPU / CUDA / DirectML) ───────────────────────────
class _OnnxRuntimeTarget(BaseExportTarget):
    """Shared base — all three produce the SAME .onnx; they differ only in the
    execution provider + provider options written into the deploy manifest."""

    category = "onnxruntime"
    output_ext = ".onnx"
    _provider_options: dict = {}

    def is_available(self) -> tuple[bool, str]:
        ok = super().is_available()
        if not ok[0]:
            return ok
        # .onnx source always works; .pt source needs ultralytics — but we
        # can't see the source here, so report available and let export() give
        # the friendly message if ultralytics is missing for a .pt.
        return True, ""

    def export(self, model_path, output_path, opts, on_progress=None, is_stopped=None) -> dict:
        result = _produce_onnx(model_path, output_path, opts, on_progress)
        if "error" in result:
            return {"target": self.name, **result}
        return {
            "target": self.name,
            "provider": self.provider,
            "provider_options": self._provider_options,
            "output_path": result.get("output_path", output_path),
            "output_size_mb": _ex.file_size_mb(result.get("output_path", output_path)),
            "precision": opts.get("precision", "fp32"),
            **{k: v for k, v in result.items() if k not in ("output_path",)},
        }


class OnnxCpuTarget(_OnnxRuntimeTarget):
    name = "onnxruntime_cpu"
    label = "ONNX Runtime · CPU"
    description = "Portable ONNX for CPUExecutionProvider (most compatible)."
    provider = "CPUExecutionProvider"


class OnnxCudaTarget(_OnnxRuntimeTarget):
    name = "onnxruntime_cuda"
    label = "ONNX Runtime · CUDA (onnxruntime-gpu)"
    description = "ONNX tuned for NVIDIA CUDAExecutionProvider; FP16 recommended."
    provider = "CUDAExecutionProvider"
    platforms = ("win32", "linux")
    _provider_options = {"device_id": 0}


class OnnxDirectMLTarget(_OnnxRuntimeTarget):
    name = "onnxruntime_directml"
    label = "ONNX Runtime · DirectML"
    description = "ONNX for Windows DmlExecutionProvider (AMD/Intel/NVIDIA GPU)."
    provider = "DmlExecutionProvider"
    platforms = ("win32",)
    # DirectML requires disabling the memory pattern optimizer (see
    # core.model_loader._create_session) — record it so the runtime config is correct.
    _provider_options = {"device_id": 0}


# ── TensorRT ───────────────────────────────────────────────────────────────
class TensorRTTarget(BaseExportTarget):
    name = "tensorrt"
    label = "TensorRT engine"
    category = "tensorrt"
    description = "Serialized NVIDIA TensorRT .engine (CUDA GPU required to build)."
    provider = "TensorrtExecutionProvider"
    output_ext = ".engine"
    platforms = ("win32", "linux")

    def is_available(self) -> tuple[bool, str]:
        ok = super().is_available()
        if not ok[0]:
            return ok
        if not _have("tensorrt"):
            return False, "needs tensorrt (pip install tensorrt) + a CUDA GPU"
        return True, ""

    def export(self, model_path, output_path, opts, on_progress=None, is_stopped=None) -> dict:
        if _ex.is_torch_checkpoint(model_path):
            ok, reason = _need_ultralytics_for_pt()
            if not ok:
                return {"target": self.name, "error": reason, "output_path": output_path}
            if on_progress:
                on_progress(0, 1, "Building TensorRT engine via ultralytics...")
            produced = _ex.ultralytics_export(model_path, "engine", opts)
            final = _ex.finalize_artifact(produced, output_path)
            res = {"output_path": final, "output_size_mb": _ex.file_size_mb(final)}
        else:
            res = _ex.build_tensorrt_engine(model_path, output_path, opts, on_progress)
        return {
            "target": self.name, "provider": self.provider,
            "precision": opts.get("precision", "fp32"), **res,
        }


# ── OpenVINO ──────────────────────────────────────────────────────────────
class OpenVINOTarget(BaseExportTarget):
    name = "openvino"
    label = "OpenVINO IR"
    category = "openvino"
    description = "OpenVINO Intermediate Representation (.xml + .bin) for Intel CPU/iGPU."
    provider = "OpenVINOExecutionProvider"
    output_ext = ".xml"
    platforms = ("win32", "linux")

    def is_available(self) -> tuple[bool, str]:
        ok = super().is_available()
        if not ok[0]:
            return ok
        if not _have("openvino"):
            return False, "needs openvino (pip install openvino) — run from the openvino env"
        return True, ""

    def export(self, model_path, output_path, opts, on_progress=None, is_stopped=None) -> dict:
        if _ex.is_torch_checkpoint(model_path):
            ok, reason = _need_ultralytics_for_pt()
            if not ok:
                return {"target": self.name, "error": reason, "output_path": output_path}
            if on_progress:
                on_progress(0, 1, "Exporting OpenVINO IR via ultralytics...")
            produced = _ex.ultralytics_export(model_path, "openvino", opts)
            final = _ex.finalize_artifact(produced, output_path)
            res = {"output_path": final, "output_size_mb": _ex.dir_size_mb(final)
                   if os.path.isdir(final) else _ex.file_size_mb(final)}
        else:
            if on_progress:
                on_progress(0, 1, "Converting ONNX to OpenVINO IR...")
            res = _ex.convert_openvino(model_path, output_path, opts)
        return {
            "target": self.name, "provider": self.provider,
            "precision": opts.get("precision", "fp32"), **res,
        }


# ── CoreML (macOS build only) ─────────────────────────────────────────────
class CoreMLTarget(BaseExportTarget):
    name = "coreml"
    label = "CoreML package"
    category = "coreml"
    description = "Apple CoreML .mlpackage (macOS build only)."
    provider = "CoreMLExecutionProvider"
    output_ext = ".mlpackage"
    platforms = ("darwin",)

    def is_available(self) -> tuple[bool, str]:
        ok = super().is_available()
        if not ok[0]:
            return ok
        if not _have("coremltools"):
            return False, "needs coremltools (pip install coremltools)"
        return True, ""

    def export(self, model_path, output_path, opts, on_progress=None, is_stopped=None) -> dict:
        if not _ex.is_torch_checkpoint(model_path):
            return {"target": self.name,
                    "error": "CoreML export currently requires a PyTorch (.pt) source",
                    "output_path": output_path}
        ok, reason = _need_ultralytics_for_pt()
        if not ok:
            return {"target": self.name, "error": reason, "output_path": output_path}
        if on_progress:
            on_progress(0, 1, "Exporting CoreML package via ultralytics...")
        produced = _ex.ultralytics_export(model_path, "coreml", opts)
        final = _ex.finalize_artifact(produced, output_path)
        size = _ex.dir_size_mb(final) if os.path.isdir(final) else _ex.file_size_mb(final)
        return {"target": self.name, "provider": self.provider,
                "output_path": final, "output_size_mb": size,
                "precision": opts.get("precision", "fp32")}


# ── TorchScript ───────────────────────────────────────────────────────────
class TorchScriptTarget(BaseExportTarget):
    name = "torchscript"
    label = "TorchScript"
    category = "torchscript"
    description = "Serialized TorchScript module (.torchscript) for libtorch deployment."
    provider = None
    output_ext = ".torchscript"

    def is_available(self) -> tuple[bool, str]:
        ok = super().is_available()
        if not ok[0]:
            return ok
        if not _have("torch"):
            return False, "needs torch (pip install torch)"
        return True, ""

    def export(self, model_path, output_path, opts, on_progress=None, is_stopped=None) -> dict:
        if not _ex.is_torch_checkpoint(model_path):
            return {"target": self.name,
                    "error": "TorchScript export requires a PyTorch (.pt) source",
                    "output_path": output_path}
        ok, reason = _need_ultralytics_for_pt()
        if not ok:
            return {"target": self.name, "error": reason, "output_path": output_path}
        if on_progress:
            on_progress(0, 1, "Exporting TorchScript via ultralytics...")
        produced = _ex.ultralytics_export(model_path, "torchscript", opts)
        final = _ex.finalize_artifact(produced, output_path)
        return {"target": self.name, "provider": self.provider,
                "output_path": final, "output_size_mb": _ex.file_size_mb(final),
                "precision": opts.get("precision", "fp32")}


def _register_builtins() -> None:
    for target in (
        OnnxCpuTarget(), OnnxCudaTarget(), OnnxDirectMLTarget(),
        TensorRTTarget(), OpenVINOTarget(), CoreMLTarget(), TorchScriptTarget(),
    ):
        try:
            registry.register(target)
        except ValueError:
            # Already registered (module re-import) — harmless.
            pass


_register_builtins()
