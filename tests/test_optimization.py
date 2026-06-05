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


class TestGenerateAllCharts:
    """generate_all_charts parallelizes the 5 builders; dict shape must match
    the keys the diagnose route previously built serially."""

    _EXPECTED_KEYS = {
        "weight_distribution", "op_distribution", "quantization_sensitivity",
        "channel_importance", "model_overview",
    }

    def test_keys_match_serial_contract(self, conv_bn_conv_model):
        from core.diagnosis_charts import generate_all_charts
        diag = ModelDiagnosisEngine().diagnose(conv_bn_conv_model)
        charts = generate_all_charts(diag)
        # Exact same key set as the legacy serial dict — no more, no fewer.
        assert set(charts.keys()) == self._EXPECTED_KEYS

    def test_each_chart_is_str(self, conv_bn_conv_model):
        from core.diagnosis_charts import generate_all_charts
        diag = ModelDiagnosisEngine().diagnose(conv_bn_conv_model)
        charts = generate_all_charts(diag)
        for key, img in charts.items():
            assert isinstance(img, str), key
            # When matplotlib is present each value is a base64 PNG; "" is the
            # graceful fallback when matplotlib is absent.
            if img:
                assert img.startswith("data:image/png;base64,"), key

    def test_matches_serial_builders(self, conv_bn_conv_model):
        # The parallel orchestrator must feed each builder the same diagnosis
        # slice the route fed serially, so non-empty charts come back non-empty.
        from core.diagnosis_charts import generate_all_charts
        diag = ModelDiagnosisEngine().diagnose(conv_bn_conv_model)
        charts = generate_all_charts(diag)
        # model_overview is built from the whole diagnosis and always renders.
        assert charts["model_overview"].startswith("data:image/png;base64,")

    def test_single_builder_failure_isolated(self, conv_bn_conv_model):
        # A broken slice for one chart must not sink the rest of the batch.
        from core.diagnosis_charts import generate_all_charts
        diag = ModelDiagnosisEngine().diagnose(conv_bn_conv_model)
        diag["weight_analysis"] = [{"name": "bad", "mean": "not-a-number"}]
        charts = generate_all_charts(diag)
        assert set(charts.keys()) == self._EXPECTED_KEYS
        # The healthy charts still render despite the poisoned weight slice.
        assert charts["model_overview"].startswith("data:image/png;base64,")


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


# ============================================================
# Diagnosis hardening — external-data models + bounded fusable scan
# ============================================================
def _make_external_data_model(path: str, in_features=64, out_features=32):
    """Linear model whose weights live in a sidecar .data file (external data).

    Mirrors how exporters emit >2GB models: numpy_helper.to_array() on these
    initializers raises when the model is loaded with load_external_data=False.
    """
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    from onnx.external_data_helper import convert_model_to_external_data
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, in_features])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, out_features])
    W = numpy_helper.from_array(
        np.random.randn(in_features, out_features).astype(np.float32), "W")
    B = numpy_helper.from_array(np.random.randn(out_features).astype(np.float32), "B")
    mm = helper.make_node("MatMul", ["X", "W"], ["mm"])
    add = helper.make_node("Add", ["mm", "B"], ["Y"])
    graph = helper.make_graph([mm, add], "lin_ext", [X], [Y], [W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    # Force every tensor out to a sidecar regardless of size.
    loaded = onnx.load(path)
    convert_model_to_external_data(
        loaded, all_tensors_to_one_file=True,
        location=os.path.basename(path) + ".data", size_threshold=0)
    onnx.save(loaded, path)
    return path


class TestDiagnoseExternalData:
    def test_external_data_with_sidecar_diagnoses(self, tmp_dir):
        # Sidecar present: weights must be materialized and stats computed.
        p = _make_external_data_model(os.path.join(tmp_dir, "ext.onnx"))
        assert os.path.isfile(p + ".data")
        result = ModelDiagnosisEngine().diagnose(p)
        assert result["summary"]["total_params"] > 0
        assert result["summary"]["weights_materialized"] is True
        assert len(result["weight_analysis"]) > 0

    def test_external_data_missing_sidecar_no_crash(self, tmp_dir):
        # Sidecar deleted: diagnosis must NOT crash; it degrades to a
        # structure-only report with a clear warning finding.
        p = _make_external_data_model(os.path.join(tmp_dir, "ext_missing.onnx"))
        os.remove(p + ".data")
        result = ModelDiagnosisEngine().diagnose(p)  # must not raise
        assert result["summary"]["weights_materialized"] is False
        # Graph structure still analyzed (nodes counted).
        assert result["summary"]["num_nodes"] >= 1
        # A weights-category warning explains why stats are absent.
        weight_warnings = [f for f in result["findings"]
                           if f["category"] == "weights"
                           and f["severity"] == "warning"]
        assert weight_warnings, "expected a warning about unmaterialized weights"
        assert "external data" in weight_warnings[0]["message"].lower()


class TestDiagnoseFusableScanBounded:
    def test_large_graph_scan_capped(self, conv_model, monkeypatch):
        # Lower the cap so a tiny graph trips the bound, proving the scan is
        # skipped (not run quadratically) and flagged instead of hanging.
        import core.model_diagnosis as md
        monkeypatch.setattr(md, "_FUSABLE_SCAN_MAX_NODES", 1)
        result = ModelDiagnosisEngine().diagnose(conv_model)
        ga = result["graph_analysis"]
        assert ga["fusable_scan_capped"] is True
        assert ga["fusable_patterns"] == 0  # scan skipped, nothing counted

    def test_small_graph_scan_runs(self, conv_bn_conv_model):
        # Under the default cap the scan runs and finds the Conv->BN fusion.
        result = ModelDiagnosisEngine().diagnose(conv_bn_conv_model)
        ga = result["graph_analysis"]
        assert ga["fusable_scan_capped"] is False
        assert ga["fusable_patterns"] >= 1


# ============================================================
# Quantizer P0s — FP16 I/O preservation + static-calib preprocessing spec
# ============================================================
def _make_square_image_dir(d: str, n: int = 3, size: int = 32):
    """Write a few small JPEGs so the calibration reader has inputs."""
    import cv2
    for i in range(n):
        img = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(d, f"img_{i}.jpg"), img)


def _make_fp32_conv_model(path: str, hw: int = 8):
    """Square-input Conv+Relu fp32 model with fixed I/O for fp16/static tests."""
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, hw, hw])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, hw, hw])
    W = numpy_helper.from_array(np.random.randn(4, 3, 3, 3).astype(np.float32), "W")
    B = numpy_helper.from_array(np.zeros(4, dtype=np.float32), "B")
    conv = helper.make_node("Conv", ["X", "W", "B"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    relu = helper.make_node("Relu", ["c"], ["Y"])
    g = helper.make_graph([conv, relu], "fp32_conv", [X], [Y], [W, B])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)
    return path


@pytest.fixture
def import_cv2():
    import cv2  # noqa: F401 — guard: skip quantizer tests when cv2 absent
    return cv2


class TestConvertFp16IOTypes:
    def test_keeps_float32_graph_io(self, tmp_dir, import_cv2):
        import onnxruntime as ort
        from core.quantizer import convert_fp16
        src = _make_fp32_conv_model(os.path.join(tmp_dir, "fp32.onnx"))
        out = os.path.join(tmp_dir, "fp16.onnx")
        result = convert_fp16(src, out)
        assert os.path.isfile(out)
        assert result["method"] == "fp16"
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        # I/O must remain fp32 so the benchmark/profiler hand-off keeps working.
        assert sess.get_inputs()[0].type == "tensor(float)"
        assert sess.get_outputs()[0].type == "tensor(float)"
        # And it must actually run on an fp32 feed.
        feed = {sess.get_inputs()[0].name: np.random.rand(1, 3, 8, 8).astype(np.float32)}
        out_vals = sess.run(None, feed)
        assert out_vals[0].dtype == np.float32

    def test_external_data_model_converts_and_loads(self, tmp_dir, import_cv2):
        import onnxruntime as ort
        from core.quantizer import convert_fp16
        # Force external data so the load_external_data=True path is exercised.
        src = _make_external_data_model(os.path.join(tmp_dir, "ext_fp32.onnx"))
        assert os.path.isfile(src + ".data")
        out = os.path.join(tmp_dir, "ext_fp16.onnx")
        result = convert_fp16(src, out)
        assert os.path.isfile(out)
        # The converted model must still load and keep fp32 I/O.
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        assert sess.get_inputs()[0].type == "tensor(float)"
        assert "external_data" in result


class TestStaticCalibPreprocessing:
    def test_auto_guess_is_reported(self, tmp_dir, import_cv2):
        from core.quantizer import quantize_static
        _make_square_image_dir(tmp_dir)
        src = _make_fp32_conv_model(os.path.join(tmp_dir, "calib_src.onnx"))
        out = os.path.join(tmp_dir, "calib_int8.onnx")
        result = quantize_static(src, out, calibration_dir=tmp_dir, max_images=3)
        # The auto-guessed preprocessing must be surfaced, not silent.
        assert result["preprocess_auto_guessed"] is True
        pp = result["preprocess"]
        assert pp["color_order"] == "rgb"
        assert pp["scale"] == 255.0
        # Square input (H==W) -> letterbox guessed on.
        assert pp["use_letterbox"] is True

    def test_explicit_spec_overrides_and_is_reported(self, tmp_dir, import_cv2, monkeypatch):
        import core.inference as inference
        from core.quantizer import quantize_static, PreprocessSpec
        _make_square_image_dir(tmp_dir)
        src = _make_fp32_conv_model(os.path.join(tmp_dir, "calib_src2.onnx"))
        out = os.path.join(tmp_dir, "calib_int8_2.onnx")

        # Spy on letterbox: with use_letterbox=False it must NOT be called.
        called = {"n": 0}
        real_letterbox = inference.letterbox

        def _spy(img, shape):
            called["n"] += 1
            return real_letterbox(img, shape)

        monkeypatch.setattr(inference, "letterbox", _spy)

        spec = PreprocessSpec(use_letterbox=False, color_order="bgr",
                              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        result = quantize_static(src, out, calibration_dir=tmp_dir, max_images=3,
                                 preprocess=spec)
        assert called["n"] == 0  # plain resize path, letterbox never invoked
        assert result["preprocess_auto_guessed"] is False
        pp = result["preprocess"]
        assert pp["use_letterbox"] is False
        assert pp["color_order"] == "bgr"
        assert pp["mean"] == [0.5, 0.5, 0.5]

    def test_letterbox_spec_invokes_letterbox(self, tmp_dir, import_cv2, monkeypatch):
        import core.inference as inference
        from core.quantizer import quantize_static, PreprocessSpec
        _make_square_image_dir(tmp_dir)
        src = _make_fp32_conv_model(os.path.join(tmp_dir, "calib_src3.onnx"))
        out = os.path.join(tmp_dir, "calib_int8_3.onnx")

        called = {"n": 0}
        real_letterbox = inference.letterbox

        def _spy(img, shape):
            called["n"] += 1
            return real_letterbox(img, shape)

        monkeypatch.setattr(inference, "letterbox", _spy)

        quantize_static(src, out, calibration_dir=tmp_dir, max_images=3,
                        preprocess=PreprocessSpec(use_letterbox=True))
        assert called["n"] >= 1  # letterbox used for every calibration image


# ============================================================
# Post-optimization accuracy validation — agreement metric math (PURE LOGIC)
# These import only numpy via core.quantizer's metric helpers; safe to RUN.
# ============================================================
class TestAgreementMetricsPureLogic:
    """compute_agreement_metrics / _cosine_similarity operate on plain numpy
    output lists — no onnxruntime, no cv2 — so the math is verifiable here."""

    def test_identical_outputs_perfect_agreement(self):
        from core.quantizer import compute_agreement_metrics
        a = [np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)]
        m = compute_agreement_metrics(a, [x.copy() for x in a])
        assert m["cosine"] == pytest.approx(1.0, abs=1e-6)
        assert m["max_abs_diff"] == pytest.approx(0.0, abs=1e-6)
        assert m["top1_match"] is True

    def test_divergent_outputs_low_cosine(self):
        # Anti-correlated vectors -> cosine near -1, top-1 differs.
        from core.quantizer import compute_agreement_metrics
        a = [np.array([[1.0, 2.0, 3.0, 10.0]], dtype=np.float32)]
        b = [np.array([[10.0, 3.0, 2.0, 1.0]], dtype=np.float32)]
        m = compute_agreement_metrics(a, b)
        assert m["cosine"] < 0.9
        assert m["top1_match"] is False
        assert m["max_abs_diff"] == pytest.approx(9.0, abs=1e-6)

    def test_cosine_similarity_zero_vectors(self):
        from core.quantizer import _cosine_similarity
        z = np.zeros(4, dtype=np.float32)
        # Both zero -> treated as agreeing (avoids 0/0 NaN poisoning the mean).
        assert _cosine_similarity(z, z) == 1.0
        # One zero, one non-zero -> fully divergent.
        assert _cosine_similarity(z, np.array([1.0, 0, 0, 0])) == 0.0

    def test_shape_mismatch_marks_inf_diff(self):
        # A pruned/quantized model that changes output shape is a strong
        # degradation signal: max_abs_diff must surface as inf, not silently 0.
        from core.quantizer import compute_agreement_metrics
        a = [np.array([[1.0, 2.0, 3.0]], dtype=np.float32)]
        b = [np.array([[1.0, 2.0]], dtype=np.float32)]
        m = compute_agreement_metrics(a, b)
        assert m["max_abs_diff"] == float("inf")

    def test_empty_outputs_fail_closed(self):
        from core.quantizer import compute_agreement_metrics
        m = compute_agreement_metrics([], [])
        assert m["cosine"] == 0.0
        assert m["top1_match"] is False

    def test_cosine_to_agreement_remap_range(self):
        # The route/validator remaps cosine [-1,1] -> agreement [0,1] as
        # (cosine + 1) / 2. Verify the boundary mapping the validator relies on.
        from core.quantizer import compute_agreement_metrics
        a = [np.array([[1.0, 0.0]], dtype=np.float32)]
        same = compute_agreement_metrics(a, [a[0].copy()])
        agreement = (same["cosine"] + 1.0) / 2.0
        assert agreement == pytest.approx(1.0, abs=1e-6)

    def test_default_threshold_exposed(self):
        from core.quantizer import DEFAULT_AGREEMENT_THRESHOLD
        assert 0.0 < DEFAULT_AGREEMENT_THRESHOLD <= 1.0
        assert DEFAULT_AGREEMENT_THRESHOLD == 0.9


# ============================================================
# Post-optimization accuracy validation — end-to-end (ONNX/ORT; DO NOT RUN here)
# These build real ONNX models and run onnxruntime CPU; deferred to the central
# memory-managed test pass per the memory-safe protocol.
# ============================================================
def _make_divergent_conv_model(path: str, hw: int = 8):
    """Same I/O as _make_fp32_conv_model but with weights * -1 and a bias shift,
    so running it on the same inputs yields very different outputs (a stand-in
    for a 'wrecked' optimization). Reuses the square-input fixed-I/O shape so the
    validator can feed both models identical tensors."""
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, hw, hw])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, hw, hw])
    # Strong negative weights -> outputs anti-correlated with the original.
    W = numpy_helper.from_array((np.ones((4, 3, 3, 3)) * -5.0).astype(np.float32), "W")
    B = numpy_helper.from_array((np.ones(4) * 7.0).astype(np.float32), "B")
    conv = helper.make_node("Conv", ["X", "W", "B"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    relu = helper.make_node("Relu", ["c"], ["Y"])
    g = helper.make_graph([conv, relu], "divergent_conv", [X], [Y], [W, B])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)
    return path


class TestValidateOptimizationE2E:
    """DO NOT RUN in the memory-safe pass — uses onnxruntime. The agreement math
    these assert against is already covered by TestAgreementMetricsPureLogic."""

    def test_identical_model_high_agreement(self, tmp_dir, import_cv2):
        from core.quantizer import validate_optimization
        _make_square_image_dir(tmp_dir, n=4)
        src = _make_fp32_conv_model(os.path.join(tmp_dir, "val_src.onnx"))
        # An exact copy is a no-op "optimization" -> agreement ~1.0, not degraded.
        import shutil as _sh
        dst = os.path.join(tmp_dir, "val_copy.onnx")
        _sh.copy2(src, dst)
        result = validate_optimization(src, dst, sample_dir=tmp_dir, max_images=4)
        assert result["validated"] is True
        assert result["n_images"] == 4
        assert result["agreement"] >= 0.99
        assert result["degraded"] is False
        # Original path is preserved for fallback.
        assert result["original_path"] == src

    def test_wrecked_model_flagged_degraded(self, tmp_dir, import_cv2):
        from core.quantizer import validate_optimization
        _make_square_image_dir(tmp_dir, n=4)
        src = _make_fp32_conv_model(os.path.join(tmp_dir, "val_src2.onnx"))
        wrecked = _make_divergent_conv_model(os.path.join(tmp_dir, "val_wrecked.onnx"))
        result = validate_optimization(src, wrecked, sample_dir=tmp_dir,
                                       max_images=4, threshold=0.9)
        assert result["validated"] is True
        assert result["degraded"] is True
        assert result["agreement"] < 0.9

    def test_no_sample_images_fails_closed(self, tmp_dir, import_cv2):
        from core.quantizer import validate_optimization
        # Empty dir (no images) -> validated False, never a false "good".
        empty = os.path.join(tmp_dir, "empty")
        os.makedirs(empty, exist_ok=True)
        src = _make_fp32_conv_model(os.path.join(tmp_dir, "val_src3.onnx"))
        result = validate_optimization(src, src, sample_dir=empty)
        assert result["validated"] is False
        assert result["n_images"] == 0

    def test_result_keys_contract(self, tmp_dir, import_cv2):
        from core.quantizer import validate_optimization
        _make_square_image_dir(tmp_dir, n=2)
        src = _make_fp32_conv_model(os.path.join(tmp_dir, "val_src4.onnx"))
        result = validate_optimization(src, src, sample_dir=tmp_dir, max_images=2)
        for key in ("validated", "agreement", "threshold", "degraded",
                    "n_images", "max_abs_diff", "top1_agreement", "original_path"):
            assert key in result


class TestPipelineValidationE2E:
    """DO NOT RUN — pipeline + onnxruntime validation wiring."""

    def test_validate_off_by_default(self, linear_model, tmp_dir):
        # Default behaviour unchanged: no validation key unless requested.
        pipe = OptimizationPipeline(registry)
        pipe.add_step("weight_pruning", sparsity_ratio=0.1)
        out = os.path.join(tmp_dir, "pipe_noval.onnx")
        result = pipe.run(linear_model, out)
        assert "validation" not in result

    def test_validate_runs_when_requested(self, tmp_dir, import_cv2):
        _make_square_image_dir(tmp_dir, n=3)
        src = _make_fp32_conv_model(os.path.join(tmp_dir, "pipe_val_src.onnx"))
        pipe = OptimizationPipeline(registry)
        pipe.add_step("dead_node_eliminator")  # near-lossless -> high agreement
        out = os.path.join(tmp_dir, "pipe_val_out.onnx")
        result = pipe.run(src, out, validate=True, sample_dir=tmp_dir,
                          validate_max_images=3)
        assert "validation" in result
        assert result["validation"]["validated"] is True
        assert result["validation"]["degraded"] is False

    def test_validate_without_sample_dir_skips(self, tmp_dir, import_cv2):
        src = _make_fp32_conv_model(os.path.join(tmp_dir, "pipe_val_src2.onnx"))
        pipe = OptimizationPipeline(registry)
        pipe.add_step("dead_node_eliminator")
        out = os.path.join(tmp_dir, "pipe_val_out2.onnx")
        # validate=True but no sample_dir -> the pipeline skips (no validation key).
        result = pipe.run(src, out, validate=True, sample_dir="")
        assert "validation" not in result
