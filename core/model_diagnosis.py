"""Model Diagnosis Engine — analyze, diagnose, and recommend optimizations."""
import os
import numpy as np

# ── Architecture detection patterns ─────────────────────

_ARCH_PATTERNS = {
    "yolo": {"Conv", "Concat", "Resize", "Sigmoid"},
    "rfdetr": {"MultiHeadAttention", "LayerNormalization", "Conv", "Gather"},
    "eva02": {"LayerNormalization", "Attention", "Gelu", "MatMul"},
    "transformer": {"LayerNormalization", "Attention", "MatMul", "Softmax"},
}

_QUANTIZABLE_OPS = {
    "Conv", "MatMul", "Gemm", "ConvTranspose", "Add", "Mul",
    "Relu", "Clip", "Sigmoid", "MaxPool", "AveragePool",
    "GlobalAveragePool", "Concat", "Reshape", "Transpose", "Flatten",
}

_MEMORY_BOUND_OPS = {
    "Reshape", "Transpose", "Concat", "Flatten", "Squeeze",
    "Unsqueeze", "Gather", "Slice", "Split", "Pad",
}

_FUSABLE_PATTERNS = [
    ("Conv", "BatchNormalization"),
    ("Conv", "Relu"),
    ("Conv", "Clip"),
    ("MatMul", "Add"),
    ("Conv", "Add", "Relu"),
]


class ModelDiagnosisEngine:
    def diagnose(self, model_path: str) -> dict:
        import onnx
        from onnx import numpy_helper

        model = onnx.load(model_path, load_external_data=False)
        graph = model.graph
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        # Basic info
        init_map = {}
        total_params = 0
        for init in graph.initializer:
            arr = numpy_helper.to_array(init)
            init_map[init.name] = arr
            total_params += arr.size

        op_counts = {}
        for node in graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        all_ops = set(op_counts.keys())
        num_nodes = len(graph.node)

        # Architecture detection
        architecture = self._detect_architecture(all_ops, op_counts)

        # Weight analysis
        weight_analysis = self._analyze_weights(graph, init_map)

        # Quantization analysis
        quant_analysis = self._analyze_quantization(graph, init_map, op_counts)

        # Pruning analysis
        pruning_analysis = self._analyze_pruning(graph, init_map)

        # Graph efficiency
        graph_analysis = self._analyze_graph_efficiency(graph, op_counts, num_nodes)

        # Op summary
        op_summary = [{"op_type": k, "count": v} for k, v in
                      sorted(op_counts.items(), key=lambda x: -x[1])]

        # Generate findings
        findings = self._generate_findings(
            file_size_mb, total_params, quant_analysis,
            pruning_analysis, graph_analysis, architecture, weight_analysis)

        # Health score (0-100)
        critical = sum(1 for f in findings if f["severity"] == "critical")
        warnings = sum(1 for f in findings if f["severity"] == "warning")
        health_score = max(0, 100 - critical * 20 - warnings * 5)

        return {
            "summary": {
                "file_size_mb": round(file_size_mb, 2),
                "total_params": total_params,
                "num_nodes": num_nodes,
                "opset_version": model.opset_import[0].version if model.opset_import else 0,
                "health_score": health_score,
            },
            "architecture": architecture,
            "findings": findings,
            "weight_analysis": weight_analysis,
            "quantization_analysis": quant_analysis,
            "pruning_analysis": pruning_analysis,
            "graph_analysis": graph_analysis,
            "op_summary": op_summary,
        }

    def _detect_architecture(self, ops: set, op_counts: dict) -> str:
        scores = {}
        for arch, pattern_ops in _ARCH_PATTERNS.items():
            overlap = len(ops & pattern_ops)
            scores[arch] = overlap / len(pattern_ops)
        best = max(scores, key=scores.get)
        if scores[best] >= 0.5:
            return best
        has_conv = "Conv" in ops
        has_attn = any(k in ops for k in ("Attention", "MultiHeadAttention", "LayerNormalization"))
        if has_conv and has_attn:
            return "hybrid"
        if has_conv:
            return "cnn"
        if has_attn:
            return "transformer"
        return "unknown"

    def _analyze_weights(self, graph, init_map: dict) -> list[dict]:
        results = []
        target_inits = set()
        for node in graph.node:
            if node.op_type in ("Conv", "MatMul", "Gemm", "ConvTranspose"):
                for inp in node.input:
                    if inp in init_map:
                        target_inits.add(inp)
                        break
        for name in sorted(target_inits):
            arr = init_map[name].astype(np.float64)
            flat = arr.flatten()
            std = float(np.std(flat))
            results.append({
                "name": name,
                "shape": list(arr.shape),
                "size": arr.size,
                "min": round(float(np.min(flat)), 6),
                "max": round(float(np.max(flat)), 6),
                "mean": round(float(np.mean(flat)), 6),
                "std": round(std, 6),
                "sparsity": round(float(np.mean(np.abs(flat) < 1e-7)), 4),
                "outlier_ratio": round(float(np.mean(np.abs(flat) > 3 * std)), 4) if std > 0 else 0,
            })
        return results

    def _analyze_quantization(self, graph, init_map, op_counts) -> dict:
        total = sum(op_counts.values())
        quant_count = sum(op_counts.get(op, 0) for op in _QUANTIZABLE_OPS)
        ratio = quant_count / max(total, 1)

        # Per-node sensitivity
        sensitive_nodes = []
        for node in graph.node:
            if node.op_type not in ("Conv", "MatMul", "Gemm"):
                continue
            for inp in node.input:
                if inp in init_map:
                    w = init_map[inp].flatten().astype(np.float64)
                    if w.size == 0:
                        continue
                    rng = float(np.max(np.abs(w)))
                    std = float(np.std(w))
                    outlier = float(np.mean(np.abs(w) > 3 * std)) if std > 0 else 0
                    score = rng * (1 + outlier * 10)
                    sensitive_nodes.append({
                        "name": node.name or inp,
                        "op_type": node.op_type,
                        "sensitivity": round(score, 4),
                    })
                    break
        sensitive_nodes.sort(key=lambda x: -x["sensitivity"])

        non_quant = sorted(set(op_counts.keys()) - _QUANTIZABLE_OPS -
                           {"BatchNormalization", "LayerNormalization", "Softmax",
                            "Dropout", "Identity", "Shape", "Cast", "Constant",
                            "ConstantOfShape"})
        return {
            "quantizable_ratio": round(ratio, 3),
            "quantizable_ops": quant_count,
            "total_ops": total,
            "sensitive_nodes": sensitive_nodes[:20],
            "non_quantizable_ops": non_quant,
        }

    def _analyze_pruning(self, graph, init_map) -> dict:
        conv_stats = []
        total_near_zero = 0
        total_weights = 0
        for node in graph.node:
            if node.op_type != "Conv" or len(node.input) < 2:
                continue
            w_name = node.input[1]
            if w_name not in init_map:
                continue
            w = init_map[w_name]
            if w.ndim != 4:
                continue
            out_ch = w.shape[0]
            norms = np.sum(np.abs(w.reshape(out_ch, -1)), axis=1)
            sparsity = float(np.mean(np.abs(w) < 1e-7))
            total_near_zero += int(np.sum(np.abs(w) < 1e-7))
            total_weights += w.size
            # Channel importance distribution
            norm_std = float(np.std(norms))
            norm_mean = float(np.mean(norms))
            low_importance = int(np.sum(norms < norm_mean * 0.1))
            conv_stats.append({
                "name": w_name,
                "out_channels": out_ch,
                "channel_norm_mean": round(norm_mean, 4),
                "channel_norm_std": round(norm_std, 4),
                "low_importance_channels": low_importance,
                "sparsity": round(sparsity, 4),
            })
        return {
            "conv_layers": conv_stats,
            "overall_sparsity": round(total_near_zero / max(total_weights, 1), 4),
            "total_conv_channels": sum(c["out_channels"] for c in conv_stats),
            "prunable_channels": sum(c["low_importance_channels"] for c in conv_stats),
        }

    def _analyze_graph_efficiency(self, graph, op_counts, num_nodes) -> dict:
        memory_bound = sum(op_counts.get(op, 0) for op in _MEMORY_BOUND_OPS)
        memory_ratio = memory_bound / max(num_nodes, 1)

        # Detect fusable patterns
        node_list = list(graph.node)
        out_to_node = {}
        for n in node_list:
            for o in n.output:
                out_to_node[o] = n
        fusable_count = 0
        for i, node in enumerate(node_list):
            for pattern in _FUSABLE_PATTERNS:
                if node.op_type == pattern[0] and len(pattern) >= 2:
                    # Check if next node matches
                    for o in node.output:
                        for j, other in enumerate(node_list):
                            if other.op_type == pattern[1] and o in other.input:
                                fusable_count += 1
                                break

        return {
            "memory_bound_ops": memory_bound,
            "memory_bound_ratio": round(memory_ratio, 3),
            "fusable_patterns": fusable_count,
            "total_nodes": num_nodes,
        }

    def _generate_findings(self, size_mb, params, quant, pruning, graph, arch, weights) -> list:
        findings = []

        # Size findings
        if size_mb > 100:
            findings.append({
                "severity": "critical", "category": "size",
                "message": f"Model is very large ({size_mb:.0f} MB). Quantization and pruning strongly recommended.",
            })
        elif size_mb > 30:
            findings.append({
                "severity": "warning", "category": "size",
                "message": f"Model is {size_mb:.0f} MB. Consider quantization for deployment.",
            })

        # Quantization findings
        qr = quant["quantizable_ratio"]
        if qr > 0.7:
            findings.append({
                "severity": "info", "category": "quantization",
                "message": f"{qr*100:.0f}% of ops are quantizable. INT8 quantization can yield ~2-3x speedup.",
            })
        if quant["non_quantizable_ops"]:
            findings.append({
                "severity": "warning", "category": "quantization",
                "message": f"Non-quantizable ops detected: {', '.join(quant['non_quantizable_ops'][:5])}. Consider mixed precision.",
            })

        # Pruning findings
        if pruning["prunable_channels"] > 0:
            ratio = pruning["prunable_channels"] / max(pruning["total_conv_channels"], 1)
            if ratio > 0.1:
                findings.append({
                    "severity": "info", "category": "pruning",
                    "message": f"{pruning['prunable_channels']} low-importance channels detected ({ratio*100:.0f}%). Channel pruning recommended.",
                })
        if pruning["overall_sparsity"] > 0.1:
            findings.append({
                "severity": "info", "category": "pruning",
                "message": f"Existing weight sparsity is {pruning['overall_sparsity']*100:.1f}%. Weight pruning can increase this further.",
            })

        # Graph efficiency findings
        if graph["fusable_patterns"] > 0:
            findings.append({
                "severity": "info", "category": "graph",
                "message": f"{graph['fusable_patterns']} fusable op patterns found. Graph optimization can reduce latency.",
            })
        if graph["memory_bound_ratio"] > 0.3:
            findings.append({
                "severity": "warning", "category": "graph",
                "message": f"{graph['memory_bound_ratio']*100:.0f}% of ops are memory-bound. Consider graph optimization.",
            })

        # Weight distribution findings
        for w in weights:
            if w["outlier_ratio"] > 0.05:
                findings.append({
                    "severity": "warning", "category": "weights",
                    "message": f"Layer '{w['name']}' has {w['outlier_ratio']*100:.1f}% outliers. May cause quantization accuracy loss.",
                })

        # Architecture-specific
        if arch == "transformer":
            findings.append({
                "severity": "info", "category": "architecture",
                "message": "Transformer architecture detected. Token Merging (ToMe) and Knowledge Distillation may further improve efficiency.",
            })

        if not findings:
            findings.append({
                "severity": "info", "category": "general",
                "message": "Model appears well-optimized. Minor improvements may still be possible.",
            })

        return findings


# ── Recommendation Engine ───────────────────────────────

class RecommendationEngine:
    def recommend(self, diagnosis: dict) -> list[dict]:
        recs = []
        summary = diagnosis["summary"]
        quant = diagnosis["quantization_analysis"]
        pruning = diagnosis["pruning_analysis"]
        graph = diagnosis["graph_analysis"]
        arch = diagnosis["architecture"]
        size_mb = summary["file_size_mb"]

        # Graph optimization — almost always beneficial, low risk
        if graph["fusable_patterns"] > 0:
            recs.append({
                "method": "ort_graph_optimizer",
                "reason": f"{graph['fusable_patterns']} fusable patterns detected. Op fusion reduces kernel launch overhead.",
                "expected_impact": "5-15% latency reduction, no accuracy loss",
                "executable": True,
                "priority": 1,
                "pipeline_config": {"optimizer": "ort_graph_optimizer", "params": {"level": "all"}},
            })

        # Dead node elimination
        recs.append({
            "method": "dead_node_eliminator",
            "reason": "Remove unreachable nodes to clean up the graph before other optimizations.",
            "expected_impact": "Cleaner graph, slight size reduction",
            "executable": True,
            "priority": 0,
            "pipeline_config": {"optimizer": "dead_node_eliminator", "params": {}},
        })

        # Quantization
        qr = quant["quantizable_ratio"]
        if qr > 0.5:
            if quant["non_quantizable_ops"]:
                recs.append({
                    "method": "mixed_precision",
                    "reason": f"{qr*100:.0f}% quantizable but {len(quant['non_quantizable_ops'])} non-quantizable op types. Mixed precision preserves accuracy on sensitive layers.",
                    "expected_impact": "2-3x speedup, <1% accuracy loss",
                    "executable": True,
                    "priority": 3,
                    "pipeline_config": {"optimizer": "mixed_precision", "params": {"exclude_pct": 20}},
                })
            else:
                recs.append({
                    "method": "static_int8",
                    "reason": f"{qr*100:.0f}% of ops are quantizable. Full INT8 quantization recommended.",
                    "expected_impact": "2-4x speedup, ~75% size reduction",
                    "executable": True,
                    "priority": 3,
                    "pipeline_config": {"optimizer": "static_int8", "params": {}},
                })

            recs.append({
                "method": "dynamic_int8",
                "reason": "Quick quantization without calibration data. Good starting point.",
                "expected_impact": "1.5-2x speedup, ~50% size reduction",
                "executable": True,
                "priority": 2,
                "pipeline_config": {"optimizer": "dynamic_int8", "params": {}},
            })

        # FP16
        if size_mb > 10:
            recs.append({
                "method": "fp16",
                "reason": "FP16 conversion halves model size with minimal accuracy loss. Effective on GPU.",
                "expected_impact": "~50% size reduction, GPU speedup",
                "executable": True,
                "priority": 2,
                "pipeline_config": {"optimizer": "fp16", "params": {}},
            })

        # Channel pruning
        if pruning["prunable_channels"] > 0:
            ratio = pruning["prunable_channels"] / max(pruning["total_conv_channels"], 1)
            if ratio > 0.05:
                recs.append({
                    "method": "channel_pruning",
                    "reason": f"{pruning['prunable_channels']} low-importance channels ({ratio*100:.0f}%). Removing them reduces FLOPs and model size.",
                    "expected_impact": f"~{ratio*100:.0f}% FLOPs reduction",
                    "executable": True,
                    "priority": 2,
                    "pipeline_config": {"optimizer": "channel_pruning", "params": {"pruning_ratio": min(ratio, 0.3)}},
                })

        # Weight pruning
        if pruning["overall_sparsity"] < 0.5 and summary["total_params"] > 100000:
            recs.append({
                "method": "weight_pruning",
                "reason": f"Current sparsity is {pruning['overall_sparsity']*100:.1f}%. Magnitude pruning can increase sparsity for faster sparse inference.",
                "expected_impact": "10-30% size reduction with sparse runtime support",
                "executable": True,
                "priority": 2,
                "pipeline_config": {"optimizer": "weight_pruning", "params": {"sparsity_ratio": 0.3}},
            })

        # Non-executable recommendations (future work)
        if arch in ("transformer", "hybrid", "eva02"):
            recs.append({
                "method": "token_merging",
                "reason": "Transformer architecture detected. Token Merging (ToMe) reduces token count in attention layers for significant speedup.",
                "expected_impact": "20-40% speedup with <1% accuracy loss",
                "executable": False,
                "explanation": "Token Merging requires PyTorch model modification during inference. Not yet supported for ONNX-only workflow. See: https://github.com/facebookresearch/ToMe",
                "priority": 4,
            })
            recs.append({
                "method": "knowledge_distillation",
                "reason": "A smaller student model can be trained to mimic this model, achieving similar accuracy at fraction of the cost.",
                "expected_impact": "2-10x speedup depending on student architecture",
                "executable": False,
                "explanation": "Knowledge Distillation requires training a student model in PyTorch/TensorFlow. Export the student to ONNX after training. See: https://arxiv.org/abs/1503.02531",
                "priority": 5,
            })

        if summary["total_params"] > 1_000_000:
            recs.append({
                "method": "lora_finetuning",
                "reason": "Large model with many parameters. LoRA can efficiently fine-tune for specific tasks with minimal additional parameters.",
                "expected_impact": "Task-specific accuracy improvement with <1% parameter overhead",
                "executable": False,
                "explanation": "LoRA requires PyTorch training. After fine-tuning, merge LoRA weights and export to ONNX. See: https://arxiv.org/abs/2106.09685",
                "priority": 5,
            })

        recs.sort(key=lambda x: x.get("priority", 99))
        return recs
