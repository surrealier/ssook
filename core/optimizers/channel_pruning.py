"""Channel Pruning — L1-norm based structured pruning on Conv layers."""
import os
import shutil
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
        from onnx import numpy_helper

        ratio = kw.get("pruning_ratio", 0.25)
        min_ch = kw.get("min_channels", 4)

        model = onnx.load(model_path)
        init_map = {i.name: i for i in model.graph.initializer}

        # Topology maps for skip decisions: how many nodes consume each tensor,
        # and which tensors are graph outputs. Pruning a Conv's output channels
        # only stays structurally valid if exactly one Conv (optionally via
        # BN/Relu/Clip) consumes its output and the Conv is not a graph head.
        consumers: dict[str, list] = {}
        for node in model.graph.node:
            for inp in node.input:
                consumers.setdefault(inp, []).append(node)
        graph_output_names = {o.name for o in model.graph.output}

        total_before = 0
        total_after = 0
        total_removed = 0
        skipped_layers: list[dict] = []

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
            conv_out = node.output[0]

            # Grouped/depthwise Conv: pruning output channels desyncs the
            # `group` attribute and breaks ORT load. Skip.
            group = 1
            for attr in node.attribute:
                if attr.name == "group":
                    group = attr.i
            if group != 1:
                skipped_layers.append({"name": w_name, "reason": "grouped_conv"})
                continue

            # Terminal Conv that feeds a graph output *directly* (no activation
            # buffer in between) is typically a detection head whose output
            # channels carry semantics (num_classes). Pruning corrupts results.
            if conv_out in graph_output_names:
                skipped_layers.append({"name": w_name, "reason": "terminal_conv"})
                continue

            # Walk the single downstream chain through channel-preserving
            # elementwise/activation ops. We can safely prune the output
            # channels when the chain either (a) reaches a single Conv whose
            # input channels we then repair, or (b) terminates at a graph
            # output (output dim just shrinks; no repair needed). Any Concat/
            # Add/Mul/Split/Resize, fan-out, or multi-consumer node would need
            # sibling-dimension fixups we don't perform — skip those.
            check_name = conv_out
            next_conv = None
            skip_reason = None
            while True:
                outs = consumers.get(check_name, [])
                if len(outs) == 0:
                    # Tensor consumed by nobody — must be a graph output (case b).
                    break
                if len(outs) > 1:
                    skip_reason = "multiple_consumers"
                    break
                nxt = outs[0]
                if nxt.op_type == "Conv":
                    if nxt.input[0] == check_name and len(nxt.input) >= 2 \
                            and nxt.input[1] in init_map:
                        next_conv = nxt
                    else:
                        skip_reason = "unsupported_consumer"
                    break
                if nxt.op_type in ("BatchNormalization", "Relu", "LeakyRelu", "Clip"):
                    check_name = nxt.output[0]
                    continue
                # Concat/Add/Mul/Split/Resize/... — channel mismatch with siblings.
                skip_reason = "unsupported_consumer"
                break

            if skip_reason is not None:
                skipped_layers.append({"name": w_name, "reason": skip_reason})
                continue

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

            # Prune the downstream Conv's input channels to match (only when a
            # single Conv consumes this output; a graph-output tail needs none).
            if next_conv is not None:
                next_w_name = next_conv.input[1]
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

        # Validate: shape-infer + a real ORT load. A pruned graph with a
        # channel mismatch we missed must not ship — copy the original back
        # and report the failure so the UI does not claim success.
        validation_error = self._validate(output_path)
        if validation_error is not None:
            shutil.copyfile(model_path, output_path)
            return {
                "method": "channel_pruning",
                "error": f"Pruned model failed validation: {validation_error}",
                "rolled_back": True,
                "skipped_layers": skipped_layers,
                "output_path": output_path,
            }

        orig_size = os.path.getsize(model_path)
        new_size = os.path.getsize(output_path)
        return {
            "method": "channel_pruning",
            "pruning_ratio": ratio,
            "channels_before": total_before,
            "channels_after": total_after,
            "channels_removed": total_removed,
            "skipped_layers": skipped_layers,
            "original_size_mb": round(orig_size / 1024 / 1024, 2),
            "output_size_mb": round(new_size / 1024 / 1024, 2),
            "output_path": output_path,
        }

    @staticmethod
    def _validate(model_path: str):
        """Return None if the pruned model shape-infers and loads, else an error string."""
        try:
            import onnx
            m = onnx.load(model_path)
            onnx.shape_inference.infer_shapes(m, strict_mode=True)
        except Exception as e:
            return str(e)
        try:
            import onnxruntime as ort
            ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        except Exception as e:
            return str(e)
        return None
