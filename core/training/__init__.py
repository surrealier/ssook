"""Training subsystem.

A plugin registry of trainers (parallel to ``core.optimizer_registry`` and
``core.deployment.registry``) that fine-tune / train vision models on the
app's datasets. Built-ins:

* Ultralytics YOLO — detection / segmentation / pose / classification.
* timm / torchvision — image classification from an ImageFolder dataset.

The trained checkpoint feeds straight into the Deployment module
(``core.deployment``) for export to a runtime artifact.

torch / ultralytics / timm are optional, lazy-imported inside each trainer, so
importing this package is cheap and safe in the ONNX-runtime-only base env.
"""
from core.training.registry import BaseTrainer, TrainerRegistry, registry
from core.training import trainers as _trainers  # noqa: F401  populates registry

__all__ = ["BaseTrainer", "TrainerRegistry", "registry"]
