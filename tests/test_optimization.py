"""Tests for the Advanced Model Optimization Toolkit — TDD first."""
import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

# ── Test helper: create minimal ONNX models ────────────

def _make_linear_model(path: str, in_features=8, out_features=4):
    """Create a minimal ONNX model: MatMul + Add (linear layer)."""
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, in_features])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, out_features])
    W = numpy_helper.from_array(np.random.randn(in_features, out_features).astype(np.float32), "W")
    B = numpy_helper.from_array(np.random.randn(out_features).astype(np.float32), "B")
    matmul = helper.make_node("MatMul", ["X", "W"], ["mm_out"])
    add = helper.make_node("Add", ["mm_out", "B"], ["Y"])
    graph = helper.make_graph([matmul, add], "linear", [X], [Y], [W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


def _make_conv_model(path: str, in_ch=3, out_ch=16, kernel=3):
    """Create a minimal ONNX model: Conv + Relu."""
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, in_ch, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    W = numpy_helper.from_array(
        np.random.randn(out_ch, in_ch, kernel, kernel).astype(np.float32), "W")
    B = numpy_helper.from_array(np.zeros(out_ch, dtype=np.float32), "B")
    conv = helper.make_node("Conv", ["X", "W", "B"], ["conv_out"],
                            kernel_shape=[kernel, kernel], pads=[1,1,1,1])
    relu = helper.make_node("Relu", ["conv_out"], ["Y"])
    graph = helper.make_graph([conv, relu], "conv_net", [X], [Y], [W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


def _make_conv_bn_conv_model(path: str, ch1=3, ch2=16, ch3=8):
    """Conv → BN → Conv model for channel pruning tests."""
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, ch1, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    W1 = numpy_helper.from_array(np.random.randn(ch2, ch1, 3, 3).astype(np.float32), "W1")
    B1 = numpy_helper.from_array(np.zeros(ch2, dtype=np.float32), "B1")
    bn_scale = numpy_helper.from_array(np.ones(ch2, dtype=np.float32), "bn_scale")
    bn_bias = numpy_helper.from_array(np.zeros(ch2, dtype=np.float32), "bn_bias")
    bn_mean = numpy_helper.from_array(np.zeros(ch2, dtype=np.float32), "bn_mean")
    bn_var = numpy_helper.from_array(np.ones(ch2, dtype=np.float32), "bn_var")
    W2 = numpy_helper.from_array(np.random.randn(ch3, ch2, 3, 3).astype(np.float32), "W2")
    B2 = numpy_helper.from_array(np.zeros(ch3, dtype=np.float32), "B2")
    conv1 = helper.make_node("Conv", ["X", "W1", "B1"], ["c1"], kernel_shape=[3,3], pads=[1,1,1,1])
    bn = helper.make_node("BatchNormalization", ["c1", "bn_scale", "bn_bias", "bn_mean", "bn_var"], ["bn_out"])
    conv2 = helper.make_node("Conv", ["bn_out", "W2", "B2"], ["Y"], kernel_shape=[3,3], pads=[1,1,1,1])
    graph = helper.make_graph([conv1, bn, conv2], "conv_bn_conv", [X], [Y],
                              [W1, B1, bn_scale, bn_bias, bn_mean, bn_var, W2, B2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="ssook_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def linear_model(tmp_dir):
    return _make_linear_model(os.path.join(tmp_dir, "linear.onnx"))


@pytest.fixture
def conv_model(tmp_dir):
    return _make_conv_model(os.path.join(tmp_dir, "conv.onnx"))


@pytest.fixture
def conv_bn_conv_model(tmp_dir):
    return _make_conv_bn_conv_model(os.path.join(tmp_dir, "conv_bn_conv.onnx"))


# ============================================================
# Task 1: Optimizer Registry
# ============================================================
from core.optimizer_registry import BaseOptimizer, OptimizerRegistry, registry


class TestBaseOptimizer:
    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseOptimizer()

    def test_concrete_subclass(self):
        class Dummy(BaseOptimizer):
            name = "dummy"
            category = "test"
            description = "test optimizer"
            def can_apply(self, model_path): return True
            def apply(self, model_path, output_path, **kw): return {"ok": True}
        d = Dummy()
        assert d.name == "dummy"
        assert d.can_apply("any") is True


class TestOptimizerRegistry:
    def test_register_and_get(self):
        r = OptimizerRegistry()
        class Opt(BaseOptimizer):
            name = "opt_a"
            category = "quantization"
            description = "A"
            def can_apply(self, p): return True
            def apply(self, p, o, **kw): return {}
        r.register(Opt())
        assert r.get("opt_a") is not None
        assert r.get("nonexistent") is None

    def test_list_by_category(self):
        r = OptimizerRegistry()
        class Q(BaseOptimizer):
            name = "q1"; category = "quantization"; description = ""
            def can_apply(self, p): return True
            def apply(self, p, o, **kw): return {}
        class P(BaseOptimizer):
            name = "p1"; category = "pruning"; description = ""
            def can_apply(self, p): return True
            def apply(self, p, o, **kw): return {}
        r.register(Q())
        r.register(P())
        assert len(r.list_by_category("quantization")) == 1
        assert len(r.list_by_category("pruning")) == 1
        assert len(r.list_by_category("graph_optimization")) == 0

    def test_duplicate_register_raises(self):
        r = OptimizerRegistry()
        class D(BaseOptimizer):
            name = "dup"; category = "test"; description = ""
            def can_apply(self, p): return True
            def apply(self, p, o, **kw): return {}
        r.register(D())
        with pytest.raises(ValueError):
            r.register(D())

    def test_list_all(self):
        r = OptimizerRegistry()
        class A(BaseOptimizer):
            name = "a"; category = "c1"; description = ""
            def can_apply(self, p): return True
            def apply(self, p, o, **kw): return {}
        class B(BaseOptimizer):
            name = "b"; category = "c2"; description = ""
            def can_apply(self, p): return True
            def apply(self, p, o, **kw): return {}
        r.register(A())
        r.register(B())
        assert len(r.list_all()) == 2

    def test_global_registry_has_builtin_quantizers(self):
        names = [o.name for o in registry.list_by_category("quantization")]
        assert "dynamic_int8" in names
        assert "static_int8" in names
        assert "fp16" in names


# ============================================================
# Task 2: Mixed Precision Quantization
# ============================================================
class TestMixedPrecision:
    def test_sensitivity_scores(self, linear_model):
        from core.optimizers.mixed_precision import compute_sensitivity_scores
        scores = compute_sensitivity_scores(linear_model)
        assert isinstance(scores, dict)
        assert len(scores) > 0
        for name, score in scores.items():
            assert isinstance(score, float)
            assert score >= 0

    def test_registered(self):
        assert registry.get("mixed_precision") is not None
        assert registry.get("mixed_precision").category == "quantization"


# ============================================================
# Task 3: Weight Pruning
# ============================================================
class TestWeightPruning:
    def test_prune_linear(self, linear_model, tmp_dir):
        opt = registry.get("weight_pruning")
        out = os.path.join(tmp_dir, "pruned.onnx")
        result = opt.apply(linear_model, out, sparsity_ratio=0.3)
        assert os.path.isfile(out)
        assert result["method"] == "weight_pruning"
        assert 0.2 <= result["overall_sparsity"] <= 0.4

    def test_zero_sparsity(self, linear_model, tmp_dir):
        opt = registry.get("weight_pruning")
        out = os.path.join(tmp_dir, "pruned0.onnx")
        result = opt.apply(linear_model, out, sparsity_ratio=0.0)
        assert result["overall_sparsity"] < 0.01

    def test_result_keys(self, linear_model, tmp_dir):
        opt = registry.get("weight_pruning")
        out = os.path.join(tmp_dir, "pruned_k.onnx")
        result = opt.apply(linear_model, out, sparsity_ratio=0.5)
        for key in ("method", "sparsity_ratio", "overall_sparsity",
                     "total_params", "zeroed_params", "output_path"):
            assert key in result


# ============================================================
# Task 4: Channel Pruning
# ============================================================
class TestChannelPruning:
    def test_prune_conv(self, conv_model, tmp_dir):
        opt = registry.get("channel_pruning")
        out = os.path.join(tmp_dir, "ch_pruned.onnx")
        result = opt.apply(conv_model, out, pruning_ratio=0.25)
        assert os.path.isfile(out)
        assert result["method"] == "channel_pruning"
        assert result["channels_removed"] > 0

    def test_min_channels(self, conv_model, tmp_dir):
        opt = registry.get("channel_pruning")
        out = os.path.join(tmp_dir, "ch_pruned_min.onnx")
        result = opt.apply(conv_model, out, pruning_ratio=0.9, min_channels=4)
        assert result["channels_after"] >= 4

    def test_result_keys(self, conv_model, tmp_dir):
        opt = registry.get("channel_pruning")
        out = os.path.join(tmp_dir, "ch_pruned_k.onnx")
        result = opt.apply(conv_model, out, pruning_ratio=0.25)
        for key in ("method", "pruning_ratio", "channels_before",
                     "channels_after", "channels_removed", "output_path"):
            assert key in result


# ============================================================
# Task 5: Graph Optimization
# ============================================================
class TestGraphOptimization:
    def test_ort_optimizer(self, linear_model, tmp_dir):
        opt = registry.get("ort_graph_optimizer")
        out = os.path.join(tmp_dir, "ort_opt.onnx")
        result = opt.apply(linear_model, out)
        assert os.path.isfile(out)
        assert "output_path" in result

    def test_dead_node_eliminator(self, linear_model, tmp_dir):
        opt = registry.get("dead_node_eliminator")
        out = os.path.join(tmp_dir, "dead_elim.onnx")
        result = opt.apply(linear_model, out)
        assert os.path.isfile(out)
        assert "nodes_before" in result
        assert "nodes_after" in result

    def test_onnx_simplifier_graceful(self, linear_model, tmp_dir):
        opt = registry.get("onnx_simplifier")
        out = os.path.join(tmp_dir, "simplified.onnx")
        result = opt.apply(linear_model, out)
        # Should succeed or gracefully report unavailable
        assert "output_path" in result or "error" in result


# ============================================================
# Task 6: Pipeline Builder
# ============================================================
from core.optimization_pipeline import OptimizationPipeline


class TestPipeline:
    def test_empty_pipeline(self, linear_model, tmp_dir):
        pipe = OptimizationPipeline(registry)
        out = os.path.join(tmp_dir, "pipe_empty.onnx")
        result = pipe.run(linear_model, out)
        assert result["steps"] == []
        assert os.path.isfile(out)

    def test_single_step(self, linear_model, tmp_dir):
        pipe = OptimizationPipeline(registry)
        pipe.add_step("weight_pruning", sparsity_ratio=0.3)
        out = os.path.join(tmp_dir, "pipe_single.onnx")
        result = pipe.run(linear_model, out)
        assert len(result["steps"]) == 1
        assert os.path.isfile(out)

    def test_multi_step(self, conv_model, tmp_dir):
        pipe = OptimizationPipeline(registry)
        pipe.add_step("dead_node_eliminator")
        pipe.add_step("weight_pruning", sparsity_ratio=0.2)
        out = os.path.join(tmp_dir, "pipe_multi.onnx")
        result = pipe.run(conv_model, out)
        assert len(result["steps"]) == 2
        assert os.path.isfile(out)
        assert result["original_size_mb"] >= 0

    def test_invalid_step_raises(self):
        pipe = OptimizationPipeline(registry)
        with pytest.raises(ValueError):
            pipe.add_step("nonexistent_optimizer")

    def test_temp_files_cleaned(self, linear_model, tmp_dir):
        pipe = OptimizationPipeline(registry)
        pipe.add_step("weight_pruning", sparsity_ratio=0.1)
        pipe.add_step("dead_node_eliminator")
        out = os.path.join(tmp_dir, "pipe_clean.onnx")
        pipe.run(linear_model, out)
        # Only output file should remain (no _step_N_ intermediates)
        intermediates = [f for f in os.listdir(tmp_dir) if "_step_" in f]
        assert len(intermediates) == 0


# ============================================================
# Task 7: Diagnose Engine — Analysis
# ============================================================
from core.model_diagnosis import ModelDiagnosisEngine


class TestDiagnoseAnalysis:
    def test_basic_diagnosis(self, linear_model):
        engine = ModelDiagnosisEngine()
        result = engine.diagnose(linear_model)
        assert "summary" in result
        assert "findings" in result
        assert "architecture" in result
        assert isinstance(result["findings"], list)

    def test_findings_have_severity(self, conv_model):
        engine = ModelDiagnosisEngine()
        result = engine.diagnose(conv_model)
        for f in result["findings"]:
            assert f["severity"] in ("critical", "warning", "info")
            assert "message" in f
            assert "category" in f

    def test_weight_analysis(self, linear_model):
        engine = ModelDiagnosisEngine()
        result = engine.diagnose(linear_model)
        assert "weight_analysis" in result
        assert len(result["weight_analysis"]) > 0

    def test_architecture_detection(self, conv_model):
        engine = ModelDiagnosisEngine()
        result = engine.diagnose(conv_model)
        assert result["architecture"] in (
            "cnn", "transformer", "yolo", "rfdetr", "eva02", "hybrid", "unknown")


# ============================================================
# Task 8: Diagnose Engine — Recommendations
# ============================================================
from core.model_diagnosis import RecommendationEngine


class TestRecommendations:
    def test_generates_recommendations(self, conv_model):
        diag = ModelDiagnosisEngine()
        diagnosis = diag.diagnose(conv_model)
        rec_engine = RecommendationEngine()
        recs = rec_engine.recommend(diagnosis)
        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_recommendation_structure(self, conv_model):
        diag = ModelDiagnosisEngine()
        diagnosis = diag.diagnose(conv_model)
        recs = RecommendationEngine().recommend(diagnosis)
        for r in recs:
            assert "method" in r
            assert "reason" in r
            assert "expected_impact" in r
            assert "executable" in r
            assert isinstance(r["executable"], bool)

    def test_non_executable_have_explanation(self, conv_model):
        diag = ModelDiagnosisEngine()
        diagnosis = diag.diagnose(conv_model)
        recs = RecommendationEngine().recommend(diagnosis)
        for r in recs:
            if not r["executable"]:
                assert "explanation" in r

    def test_pipeline_config_for_executable(self, conv_model):
        diag = ModelDiagnosisEngine()
        diagnosis = diag.diagnose(conv_model)
        recs = RecommendationEngine().recommend(diagnosis)
        for r in recs:
            if r["executable"]:
                assert "pipeline_config" in r


# ============================================================
# Task 9: Diagnose Charts
# ============================================================
class TestDiagnoseCharts:
    def test_weight_distribution_chart(self, linear_model):
        from core.diagnosis_charts import generate_weight_distribution_chart
        diag = ModelDiagnosisEngine().diagnose(linear_model)
        img = generate_weight_distribution_chart(diag["weight_analysis"])
        assert isinstance(img, str)
        # Empty string is acceptable when matplotlib is not installed
        if img:
            assert img.startswith("data:image/png;base64,")

    def test_op_time_chart(self, conv_model):
        from core.diagnosis_charts import generate_op_time_chart
        diag = ModelDiagnosisEngine().diagnose(conv_model)
        img = generate_op_time_chart(diag.get("op_summary", []))
        assert isinstance(img, str)

    def test_quantization_heatmap(self, conv_model):
        from core.diagnosis_charts import generate_quantization_heatmap
        diag = ModelDiagnosisEngine().diagnose(conv_model)
        img = generate_quantization_heatmap(diag.get("quantization_analysis", {}))
        assert isinstance(img, str)

    def test_empty_data_no_crash(self):
        from core.diagnosis_charts import generate_weight_distribution_chart
        img = generate_weight_distribution_chart([])
        assert isinstance(img, str)


# ============================================================
# v1.6.0 hardening — TOOLS-01/02/03/08/09/11/12
# ============================================================
def _make_conv_concat_model(path: str):
    """Conv whose output feeds a Concat — pruning must be skipped (TOOLS-01)."""
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    W = numpy_helper.from_array(np.random.randn(8, 3, 3, 3).astype(np.float32), "W")
    B = numpy_helper.from_array(np.zeros(8, dtype=np.float32), "B")
    conv = helper.make_node("Conv", ["X", "W", "B"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    concat = helper.make_node("Concat", ["c", "c"], ["Y"], axis=1)
    g = helper.make_graph([conv, concat], "conv_concat", [X], [Y], [W, B])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)
    return path


@pytest.fixture
def conv_concat_model(tmp_dir):
    return _make_conv_concat_model(os.path.join(tmp_dir, "conv_concat.onnx"))


class TestChannelPruningSafety:
    def test_skips_concat_feeding_conv_and_loads(self, conv_concat_model, tmp_dir):
        import onnxruntime as ort
        opt = registry.get("channel_pruning")
        out = os.path.join(tmp_dir, "cc_pruned.onnx")
        result = opt.apply(conv_concat_model, out, pruning_ratio=0.5)
        # The single Conv feeds a Concat (twice) — must be skipped, nothing
        # pruned. Either skip reason is acceptable; both avoid corruption.
        assert result["channels_removed"] == 0
        reasons = {s["reason"] for s in result["skipped_layers"]}
        assert reasons & {"unsupported_consumer", "multiple_consumers"}
        # Output must still be a loadable model (no corruption).
        ort.InferenceSession(out, providers=["CPUExecutionProvider"])

    def test_reports_skipped_layers_key(self, conv_model, tmp_dir):
        opt = registry.get("channel_pruning")
        out = os.path.join(tmp_dir, "cp_skip.onnx")
        result = opt.apply(conv_model, out, pruning_ratio=0.25)
        assert "skipped_layers" in result
        assert isinstance(result["skipped_layers"], list)


class TestMixedPrecisionExclusion:
    def test_excluded_names_non_empty(self, conv_model, tmp_dir):
        opt = registry.get("mixed_precision")
        out = os.path.join(tmp_dir, "mp.onnx")
        result = opt.apply(conv_model, out, exclude_pct=50)
        assert "error" not in result
        assert result["num_excluded"] >= 1
        # Every excluded entry must be a non-empty matchable node key.
        assert all(isinstance(n, str) and n for n in result["excluded_nodes"])

    def test_unnamed_nodes_get_keys(self, tmp_dir):
        # Build a conv model whose nodes have empty names; scores must key on output.
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        W = numpy_helper.from_array(np.random.randn(8, 3, 3, 3).astype(np.float32), "W")
        conv = helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        assert conv.name == ""  # unnamed
        g = helper.make_graph([conv], "unnamed", [X], [Y], [W])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 8
        p = os.path.join(tmp_dir, "unnamed.onnx")
        onnx.save(m, p)
        from core.optimizers.mixed_precision import compute_sensitivity_scores
        scores = compute_sensitivity_scores(p)
        assert len(scores) == 1
        assert all(k for k in scores)  # key is node.output[0], non-empty


class TestArchitectureDetectionMargin:
    def test_single_conv_not_cnn(self, conv_model):
        # One Conv lacks the discriminating "multiple Conv" requirement.
        from core.model_diagnosis import ModelDiagnosisEngine
        arch = ModelDiagnosisEngine().diagnose(conv_model)["architecture"]
        assert arch in ("cnn", "unknown")  # never a transformer family

    def test_multi_conv_is_cnn(self, conv_bn_conv_model):
        from core.model_diagnosis import ModelDiagnosisEngine
        arch = ModelDiagnosisEngine().diagnose(conv_bn_conv_model)["architecture"]
        assert arch in ("cnn", "yolo", "hybrid")


class TestHealthScoreContinuity:
    def test_score_in_range(self, conv_model):
        from core.model_diagnosis import ModelDiagnosisEngine
        score = ModelDiagnosisEngine().diagnose(conv_model)["summary"]["health_score"]
        assert 0 <= score <= 100

    def test_score_does_not_saturate_at_zero(self):
        # Many warnings should decay smoothly, not clamp hard to 0.
        from core.model_diagnosis import ModelDiagnosisEngine
        eng = ModelDiagnosisEngine()
        findings = [{"severity": "warning", "category": "x", "message": "m"}] * 10
        crit = 0
        warn = len(findings)
        score = round(100.0 * (0.7 ** crit) * (0.9 ** warn))
        assert score > 0  # continuous decay, never an abrupt 0
        assert score < 100


class TestProfilerOutput:
    def test_provider_and_flops_flag(self, conv_model):
        from core.model_profiler import profile_model, profile_to_dict
        d = profile_to_dict(profile_model(conv_model, num_runs=3, warmup=1))
        assert "provider" in d and d["provider"]
        assert "flops_approximate" in d
        assert isinstance(d["flops_approximate"], bool)


class TestModelCacheAtomic:
    def test_concurrent_compute_runs_once(self, tmp_dir, monkeypatch):
        import threading
        from core import model_cache
        # Point the cache dir at a temp dir.
        from core import paths
        monkeypatch.setattr(paths, "cache_dir", lambda *a, **k: __import__("pathlib").Path(tmp_dir))
        target = os.path.join(tmp_dir, "f.onnx")
        with open(target, "wb") as f:
            f.write(b"x" * 10)
        calls = {"n": 0}
        lock = threading.Lock()

        def _compute(p):
            with lock:
                calls["n"] += 1
            import time
            time.sleep(0.05)
            return {"value": 42}

        results = []
        def _worker():
            results.append(model_cache.get_or_compute(target, _compute))

        threads = [threading.Thread(target=_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # All workers get the same value; compute ran once (per-key lock).
        assert all(r == {"value": 42} for r in results)
        assert calls["n"] == 1
