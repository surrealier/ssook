"""Optimizer Registry — unified interface for all ONNX optimization techniques."""
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    name: str = ""
    category: str = ""       # quantization | pruning | graph_optimization
    description: str = ""

    @abstractmethod
    def can_apply(self, model_path: str) -> bool: ...

    @abstractmethod
    def apply(self, model_path: str, output_path: str, **kwargs) -> dict: ...

    def to_dict(self):
        return {"name": self.name, "category": self.category, "description": self.description}


class OptimizerRegistry:
    def __init__(self):
        self._optimizers: dict[str, BaseOptimizer] = {}

    def register(self, opt: BaseOptimizer):
        if opt.name in self._optimizers:
            raise ValueError(f"Optimizer '{opt.name}' already registered")
        self._optimizers[opt.name] = opt

    def get(self, name: str) -> BaseOptimizer | None:
        return self._optimizers.get(name)

    def list_by_category(self, category: str) -> list[BaseOptimizer]:
        return [o for o in self._optimizers.values() if o.category == category]

    def list_all(self) -> list[BaseOptimizer]:
        return list(self._optimizers.values())


# ── Built-in quantizer wrappers ─────────────────────────

class _DynamicINT8(BaseOptimizer):
    name = "dynamic_int8"
    category = "quantization"
    description = "Dynamic INT8 quantization — no calibration data needed"

    def can_apply(self, model_path):
        return model_path.endswith(".onnx")

    def apply(self, model_path, output_path, **kw):
        from core.quantizer import quantize_dynamic
        return quantize_dynamic(model_path, output_path, kw.get("weight_type", "uint8"))


class _StaticINT8(BaseOptimizer):
    name = "static_int8"
    category = "quantization"
    description = "Static INT8 quantization with calibration data"

    def can_apply(self, model_path):
        return model_path.endswith(".onnx")

    def apply(self, model_path, output_path, **kw):
        from core.quantizer import quantize_static
        return quantize_static(
            model_path, output_path,
            calibration_dir=kw.get("calibration_dir", ""),
            max_images=kw.get("max_images", 100),
            per_channel=kw.get("per_channel", True),
            weight_type=kw.get("weight_type", "int8"),
            activation_type=kw.get("activation_type", "uint8"),
            quant_format=kw.get("quant_format", "QDQ"),
            on_progress=kw.get("on_progress"),
        )


class _FP16(BaseOptimizer):
    name = "fp16"
    category = "quantization"
    description = "FP16 conversion — lower precision with minimal accuracy loss"

    def can_apply(self, model_path):
        return model_path.endswith(".onnx")

    def apply(self, model_path, output_path, **kw):
        from core.quantizer import convert_fp16
        return convert_fp16(model_path, output_path)


# ── Global registry singleton ───────────────────────────

registry = OptimizerRegistry()
registry.register(_DynamicINT8())
registry.register(_StaticINT8())
registry.register(_FP16())

# Auto-register optimizers from submodules
def _auto_register():
    try:
        from core.optimizers.mixed_precision import MixedPrecisionOptimizer
        registry.register(MixedPrecisionOptimizer())
    except Exception:
        pass
    try:
        from core.optimizers.weight_pruning import WeightPruningOptimizer
        registry.register(WeightPruningOptimizer())
    except Exception:
        pass
    try:
        from core.optimizers.channel_pruning import ChannelPruningOptimizer
        registry.register(ChannelPruningOptimizer())
    except Exception:
        pass
    try:
        from core.optimizers.graph_optimizer import (
            ORTGraphOptimizer, ONNXSimplifier, DeadNodeEliminator)
        registry.register(ORTGraphOptimizer())
        registry.register(ONNXSimplifier())
        registry.register(DeadNodeEliminator())
    except Exception:
        pass

_auto_register()
