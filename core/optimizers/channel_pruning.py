"""Channel Pruning — L1-norm based structured pruning on Conv layers."""
import os
import numpy as np
from core.optimizer_registry import BaseOptimizer


class ChannelPruningOptimizer(BaseOptimizer):
    name = "channel_pruning"
    category = "pruning"
    description = "L1-norm channel pruning — remove least important output channels"

    def can_apply(self, model_path):
        return model_path.endswith(".onnx")

    def apply(self, model_path, output_path, **kw):
        import onnx
        from onnx import numpy_helper, helper, TensorProto

        ratio = kw.get("pruning_ratio", 0.25)
        min_ch = kw.get("min_channels", 4)

        model = onnx.load(model_path)
        init_map = {i.name: i for i in model.graph.initializer}

        total_before = 0
        total_after = 0
        total_removed = 0

        # Process each Conv node
        for node in model.graph.node:
            if node.op_type != "Conv" or len(node.input) < 2:
                continue
            w_name = node.input[1]
            if w_name not in init_map:
                continue

            w_arr = numpy_helper.to_array(init_map[w_name]).copy()
            if w_arr.ndim != 4:
                continue

            out_ch = w_arr.shape[0]
            total_before += out_ch
            n_prune = int(out_ch * ratio)
            keep = max(out_ch - n_prune, min_ch)
            if keep >= out_ch:
                total_after += out_ch
                continue

            # L1-norm per output channel
            norms = np.sum(np.abs(w_arr.reshape(out_ch, -1)), axis=1)
            keep_idx = np.argsort(norms)[-keep:]
            keep_idx = np.sort(keep_idx)

            # Prune weight
            new_w = w_arr[keep_idx]
            new_init = numpy_helper.from_array(new_w, w_name)
            for i, orig in enumerate(model.graph.initializer):
                if orig.name == w_name:
                    model.graph.initializer[i].CopyFrom(new_init)
                    break

            # Prune bias if present
            if len(node.input) >= 3 and node.input[2] in init_map:
                b_name = node.input[2]
                b_arr = numpy_helper.to_array(init_map[b_name])
                new_b = b_arr[keep_idx]
                new_b_init = numpy_helper.from_array(new_b, b_name)
                for i, orig in enumerate(model.graph.initializer):
                    if orig.name == b_name:
                        model.graph.initializer[i].CopyFrom(new_b_init)
                        break

            # Prune downstream BN if connected
            conv_out = node.output[0]
            for other in model.graph.node:
                if other.op_type == "BatchNormalization" and other.input[0] == conv_out:
                    for idx in range(1, min(len(other.input), 5)):
                        bn_name = other.input[idx]
                        if bn_name in init_map:
                            bn_arr = numpy_helper.to_array(init_map[bn_name])
                            if bn_arr.shape[0] == out_ch:
                                new_bn = bn_arr[keep_idx]
                                new_bn_init = numpy_helper.from_array(new_bn, bn_name)
                                for i, orig in enumerate(model.graph.initializer):
                                    if orig.name == bn_name:
                                        model.graph.initializer[i].CopyFrom(new_bn_init)
                                        break

            # Prune next Conv's input channels if directly connected
            for other in model.graph.node:
                if other.op_type == "Conv" and other != node:
                    # Check if this conv takes output of pruned conv (possibly through BN/Relu)
                    connected = False
                    check_name = conv_out
                    for mid in model.graph.node:
                        if mid.op_type in ("BatchNormalization", "Relu", "LeakyRelu", "Clip") \
                                and mid.input[0] == check_name:
                            check_name = mid.output[0]
                    if other.input[0] == check_name and other.input[1] in init_map:
                        next_w_name = other.input[1]
                        next_w = numpy_helper.to_array(init_map[next_w_name])
                        if next_w.ndim == 4 and next_w.shape[1] == out_ch:
                            new_next_w = next_w[:, keep_idx, :, :]
                            new_next_init = numpy_helper.from_array(new_next_w, next_w_name)
                            for i, orig in enumerate(model.graph.initializer):
                                if orig.name == next_w_name:
                                    model.graph.initializer[i].CopyFrom(new_next_init)
                                    break

            removed = out_ch - keep
            total_after += keep
            total_removed += removed

        onnx.save(model, output_path)
        orig_size = os.path.getsize(model_path)
        new_size = os.path.getsize(output_path)
        return {
            "method": "channel_pruning",
            "pruning_ratio": ratio,
            "channels_before": total_before,
            "channels_after": total_after,
            "channels_removed": total_removed,
            "original_size_mb": round(orig_size / 1024 / 1024, 2),
            "output_size_mb": round(new_size / 1024 / 1024, 2),
            "output_path": output_path,
        }
