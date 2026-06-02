"""Deployment / model-export subsystem.

Mirrors the shape of ``core.optimizer_registry`` (a plugin registry behind a
uniform interface) but for *runtime-specific export targets* rather than
ONNX->ONNX transforms. Each target turns a trained model (a PyTorch
checkpoint or an existing .onnx) into a deployable artifact for one execution
provider / runtime — ONNX (for onnxruntime CPU/CUDA/DirectML/CoreML), a
TensorRT engine, an OpenVINO IR, a CoreML package, or TorchScript — with
configurable opset / batch / input-size / precision.

Importing this package populates the global ``registry`` with the built-in
targets. Heavy backends (torch, ultralytics, tensorrt, openvino, coremltools)
are imported lazily inside each target so the package import stays cheap and
never fails in the ONNX-runtime-only base environment.
"""
from core.deployment.registry import (
    BaseExportTarget,
    ExportTargetRegistry,
    registry,
)
from core.deployment import targets as _targets  # noqa: F401  populates registry

__all__ = ["BaseExportTarget", "ExportTargetRegistry", "registry"]
