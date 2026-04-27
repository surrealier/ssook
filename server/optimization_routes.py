"""/api/optimize/*, /api/diagnose/* — Model optimization & diagnosis API routes."""
import os
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from server.state import all_states, executor

router = APIRouter()

# ── State ───────────────────────────────────────────────
opt_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": {}}
diag_state = {"running": False, "progress": 0, "total": 0, "msg": "", "results": {}}
all_states["optimize"] = opt_state
all_states["diagnose"] = diag_state


# ── Optimization endpoints ──────────────────────────────

class OptimizeRequest(BaseModel):
    model_path: str
    output_path: str = ""
    # Single method mode
    method: str = ""
    params: dict = {}
    # Pipeline mode
    pipeline: list[dict] = []  # [{"optimizer": "name", "params": {...}}, ...]


@router.get("/api/optimize/methods")
async def list_methods():
    from core.optimizer_registry import registry
    result = {}
    for opt in registry.list_all():
        cat = opt.category
        if cat not in result:
            result[cat] = []
        result[cat].append(opt.to_dict())
    return result


@router.post("/api/optimize/run")
async def run_optimize(req: OptimizeRequest):
    if opt_state["running"]:
        return {"error": "Optimization already running"}
    if not req.model_path or not os.path.isfile(req.model_path):
        return {"error": "Model file not found"}

    out = req.output_path
    if not out:
        base, ext = os.path.splitext(req.model_path)
        suffix = req.method or "optimized"
        out = f"{base}_{suffix}{ext}"

    opt_state.update(running=True, progress=0, total=0, msg="Starting...", results={})

    def _run():
        try:
            from core.optimizer_registry import registry
            from core.optimization_pipeline import OptimizationPipeline

            if req.pipeline:
                # Pipeline mode
                pipe = OptimizationPipeline(registry)
                for step in req.pipeline:
                    pipe.add_step(step["optimizer"], **step.get("params", {}))
                opt_state["total"] = len(req.pipeline)
                opt_state["msg"] = f"Running pipeline ({len(req.pipeline)} steps)..."

                def _prog(step_idx, total, name, result):
                    opt_state.update(progress=step_idx, total=total,
                                     msg=f"Step {step_idx}/{total}: {name}")

                result = pipe.run(req.model_path, out, on_progress=_prog)
            elif req.method:
                # Single method mode
                opt = registry.get(req.method)
                if not opt:
                    opt_state.update(running=False, msg=f"Error: unknown method '{req.method}'")
                    return
                opt_state.update(total=1, msg=f"Running {req.method}...")
                result = opt.apply(req.model_path, out, **req.params)
                opt_state["progress"] = 1
            else:
                opt_state.update(running=False, msg="Error: specify method or pipeline")
                return

            opt_state.update(running=False, msg="Complete", results=result)
        except Exception as e:
            opt_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True, "output_path": out}


@router.get("/api/optimize/status")
async def optimize_status():
    return dict(opt_state)


# ── Diagnose endpoints ──────────────────────────────────

class DiagnoseRequest(BaseModel):
    model_path: str
    include_charts: bool = True


@router.post("/api/diagnose/run")
async def run_diagnose(req: DiagnoseRequest):
    if diag_state["running"]:
        return {"error": "Diagnosis already running"}
    if not req.model_path or not os.path.isfile(req.model_path):
        return {"error": "Model file not found"}

    diag_state.update(running=True, progress=0, total=4, msg="Starting diagnosis...", results={})

    def _run():
        try:
            from core.model_diagnosis import ModelDiagnosisEngine, RecommendationEngine

            diag_state["msg"] = "Analyzing model structure..."
            diag_state["progress"] = 1
            engine = ModelDiagnosisEngine()
            diagnosis = engine.diagnose(req.model_path)

            diag_state["msg"] = "Generating recommendations..."
            diag_state["progress"] = 2
            recs = RecommendationEngine().recommend(diagnosis)
            diagnosis["recommendations"] = recs

            if req.include_charts:
                diag_state["msg"] = "Generating charts..."
                diag_state["progress"] = 3
                try:
                    from core.diagnosis_charts import (
                        generate_weight_distribution_chart,
                        generate_op_time_chart,
                        generate_quantization_heatmap,
                        generate_channel_importance_chart,
                        generate_model_overview_chart,
                    )
                    diagnosis["charts"] = {
                        "weight_distribution": generate_weight_distribution_chart(diagnosis["weight_analysis"]),
                        "op_distribution": generate_op_time_chart(diagnosis["op_summary"]),
                        "quantization_sensitivity": generate_quantization_heatmap(diagnosis["quantization_analysis"]),
                        "channel_importance": generate_channel_importance_chart(diagnosis["pruning_analysis"]),
                        "model_overview": generate_model_overview_chart(diagnosis),
                    }
                except Exception as e:
                    diagnosis["charts"] = {"error": str(e)}

            diag_state["progress"] = 4
            diag_state.update(running=False, msg="Complete", results=diagnosis)
        except Exception as e:
            diag_state.update(running=False, msg=f"Error: {e}")

    executor.submit(_run)
    return {"ok": True}


@router.get("/api/diagnose/status")
async def diagnose_status():
    return dict(diag_state)


@router.post("/api/diagnose/apply-recommendation")
async def apply_recommendation(req: dict):
    """Apply a recommended optimization pipeline."""
    model_path = req.get("model_path", "")
    output_path = req.get("output_path", "")
    pipeline_config = req.get("pipeline_config")
    recommendation_index = req.get("recommendation_index")

    if not model_path or not os.path.isfile(model_path):
        return {"error": "Model file not found"}

    # Build pipeline from recommendation or direct config
    if pipeline_config:
        steps = [pipeline_config] if isinstance(pipeline_config, dict) else pipeline_config
    elif recommendation_index is not None and diag_state.get("results"):
        recs = diag_state["results"].get("recommendations", [])
        if 0 <= recommendation_index < len(recs):
            rec = recs[recommendation_index]
            if not rec.get("executable"):
                return {"error": f"Method '{rec['method']}' is not executable in ONNX-only mode"}
            steps = [rec["pipeline_config"]]
        else:
            return {"error": "Invalid recommendation index"}
    else:
        return {"error": "Provide pipeline_config or recommendation_index"}

    if not output_path:
        base, ext = os.path.splitext(model_path)
        output_path = f"{base}_optimized{ext}"

    # Delegate to optimize/run
    opt_req = OptimizeRequest(
        model_path=model_path,
        output_path=output_path,
        pipeline=[{"optimizer": s["optimizer"], "params": s.get("params", {})} for s in steps],
    )
    return await run_optimize(opt_req)
