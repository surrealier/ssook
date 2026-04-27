"""Weight Pruning — magnitude-based unstructured pruning on ONNX models."""
import os
import numpy as np
from core.optimizer_registry import BaseOptimizer

_TARGET_OPS = {"Conv", "MatMul", "Gemm", "ConvTranspose"}


class WeightPruningOptimizer(BaseOptimizer):
    name = "weight_pruning"
    category = "pruning"
    description = "Magnitude-based weight pruning — zero out smallest weights"

    def can_apply(self, model_path):
        return model_path.endswith(".onnx")

    def apply(self, model_path, output_path, **kw):
        import onnx
        from onnx import numpy_helper

        sparsity = kw.get("sparsity_ratio", 0.3)
        target_ops = set(kw.get("target_op_types", _TARGET_OPS))

        model = onnx.load(model_path)
        init_map = {i.name: i for i in model.graph.initializer}

        # Find weight initializer names used by target ops
        weight_names = set()
        for node in model.graph.node:
            if node.op_type in target_ops:
                for inp in node.input:
                    if inp in init_map:
                        weight_names.add(inp)
                        break  # first initializer input is typically the weight

        total_params = 0
        zeroed_params = 0
        layer_stats = []

        for wname in weight_names:
            init = init_map[wname]
            arr = numpy_helper.to_array(init).copy()
            n = arr.size
            total_params += n

            if sparsity <= 0 or n == 0:
                layer_stats.append({"name": wname, "size": n, "sparsity": 0.0})
                continue

            threshold = np.percentile(np.abs(arr), sparsity * 100)
            mask = np.abs(arr) < threshold
            arr[mask] = 0.0
            z = int(np.sum(arr == 0))
            zeroed_params += z
            layer_stats.append({"name": wname, "size": n, "sparsity": round(z / n, 4)})

            new_init = numpy_helper.from_array(arr, wname)
            # Replace initializer in-place
            for i, orig in enumerate(model.graph.initializer):
                if orig.name == wname:
                    model.graph.initializer[i].CopyFrom(new_init)
                    break

        onnx.save(model, output_path)
        orig_size = os.path.getsize(model_path)
        new_size = os.path.getsize(output_path)
        return {
            "method": "weight_pruning",
            "sparsity_ratio": sparsity,
            "overall_sparsity": round(zeroed_params / max(total_params, 1), 4),
            "total_params": total_params,
            "zeroed_params": zeroed_params,
            "layer_stats": layer_stats,
            "original_size_mb": round(orig_size / 1024 / 1024, 2),
            "output_size_mb": round(new_size / 1024 / 1024, 2),
            "output_path": output_path,
        }
