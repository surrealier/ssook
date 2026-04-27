"""Graph Optimization — ORT optimizer, ONNX Simplifier, Dead Node Elimination."""
import os
from core.optimizer_registry import BaseOptimizer


class ORTGraphOptimizer(BaseOptimizer):
    name = "ort_graph_optimizer"
    category = "graph_optimization"
    description = "ONNX Runtime graph optimization — op fusion, constant folding"

    def can_apply(self, model_path):
        return model_path.endswith(".onnx")

    def apply(self, model_path, output_path, **kw):
        import onnxruntime as ort
        level = kw.get("level", "all")  # all | basic | extended
        level_map = {
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        }
        opts = ort.SessionOptions()
        opts.graph_optimization_level = level_map.get(level, level_map["all"])
        opts.optimized_model_filepath = output_path
        ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])

        orig = os.path.getsize(model_path)
        new = os.path.getsize(output_path) if os.path.isfile(output_path) else orig
        return {
            "method": "ort_graph_optimizer",
            "level": level,
            "original_size_mb": round(orig / 1024 / 1024, 2),
            "output_size_mb": round(new / 1024 / 1024, 2),
            "output_path": output_path,
        }


class ONNXSimplifier(BaseOptimizer):
    name = "onnx_simplifier"
    category = "graph_optimization"
    description = "ONNX Simplifier — constant folding, redundant op removal"

    def can_apply(self, model_path):
        return model_path.endswith(".onnx")

    def apply(self, model_path, output_path, **kw):
        try:
            import onnx
            import onnxsim
            model = onnx.load(model_path)
            nodes_before = len(model.graph.node)
            simplified, ok = onnxsim.simplify(model)
            if ok:
                onnx.save(simplified, output_path)
                nodes_after = len(simplified.graph.node)
            else:
                import shutil
                shutil.copy2(model_path, output_path)
                nodes_after = nodes_before
        except ImportError:
            import shutil
            shutil.copy2(model_path, output_path)
            return {
                "method": "onnx_simplifier",
                "error": "onnxsim not installed (pip install onnxsim)",
                "output_path": output_path,
            }
        except Exception as e:
            import shutil
            shutil.copy2(model_path, output_path)
            return {"method": "onnx_simplifier", "error": str(e), "output_path": output_path}

        orig = os.path.getsize(model_path)
        new = os.path.getsize(output_path)
        return {
            "method": "onnx_simplifier",
            "nodes_before": nodes_before,
            "nodes_after": nodes_after,
            "nodes_removed": nodes_before - nodes_after,
            "original_size_mb": round(orig / 1024 / 1024, 2),
            "output_size_mb": round(new / 1024 / 1024, 2),
            "output_path": output_path,
        }


class DeadNodeEliminator(BaseOptimizer):
    name = "dead_node_eliminator"
    category = "graph_optimization"
    description = "Remove unreachable nodes from the ONNX graph"

    def can_apply(self, model_path):
        return model_path.endswith(".onnx")

    def apply(self, model_path, output_path, **kw):
        import onnx
        model = onnx.load(model_path)
        nodes_before = len(model.graph.node)

        # BFS from outputs to find reachable nodes
        output_names = {o.name for o in model.graph.output}
        needed = set(output_names)
        node_list = list(model.graph.node)

        # Build output→node map
        out_to_node = {}
        for node in node_list:
            for o in node.output:
                out_to_node[o] = node

        visited_nodes = set()
        queue = list(output_names)
        while queue:
            name = queue.pop()
            if name in out_to_node:
                node = out_to_node[name]
                node_id = id(node)
                if node_id not in visited_nodes:
                    visited_nodes.add(node_id)
                    for inp in node.input:
                        queue.append(inp)

        # Remove unreachable nodes
        keep = [n for n in node_list if id(n) in visited_nodes]
        del model.graph.node[:]
        model.graph.node.extend(keep)

        # Remove unused initializers
        used_inputs = set()
        for n in keep:
            used_inputs.update(n.input)
        init_to_remove = [i for i in model.graph.initializer if i.name not in used_inputs]
        for i in init_to_remove:
            model.graph.initializer.remove(i)

        nodes_after = len(model.graph.node)
        onnx.save(model, output_path)
        orig = os.path.getsize(model_path)
        new = os.path.getsize(output_path)
        return {
            "method": "dead_node_eliminator",
            "nodes_before": nodes_before,
            "nodes_after": nodes_after,
            "nodes_removed": nodes_before - nodes_after,
            "original_size_mb": round(orig / 1024 / 1024, 2),
            "output_size_mb": round(new / 1024 / 1024, 2),
            "output_path": output_path,
        }
