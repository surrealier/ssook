"""Optimization Pipeline — chain multiple optimization steps sequentially."""
import os
import shutil
import tempfile
from core.optimizer_registry import OptimizerRegistry


class OptimizationPipeline:
    def __init__(self, registry: OptimizerRegistry):
        self._registry = registry
        self._steps: list[tuple[str, dict]] = []

    def add_step(self, optimizer_name: str, **kwargs):
        if self._registry.get(optimizer_name) is None:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        self._steps.append((optimizer_name, kwargs))

    def run(self, model_path: str, output_path: str, on_progress=None,
            validate: bool = False, sample_dir: str = "",
            validate_max_images: int = 16,
            validate_threshold: float = None) -> dict:
        """Run the chained steps, optionally validating the result's accuracy.

        `validate` is OFF by default to preserve existing behaviour. When True
        AND a `sample_dir` of images is given, after producing `output_path` the
        original and optimized models are run on the same images and an
        agreement score is attached under the result key `validation`
        (see core.quantizer.validate_optimization). Validation never blocks the
        optimization: any failure is caught and reported as
        `validation: {validated: False, error: ...}`.
        """
        orig_size = os.path.getsize(model_path)
        if not self._steps:
            shutil.copy2(model_path, output_path)
            return {
                "steps": [],
                "original_size_mb": round(orig_size / 1024 / 1024, 2),
                "final_size_mb": round(orig_size / 1024 / 1024, 2),
                "output_path": output_path,
            }

        tmp_dir = tempfile.mkdtemp(prefix="ssook_pipe_")
        step_results = []
        current_input = model_path

        try:
            for idx, (name, kwargs) in enumerate(self._steps):
                opt = self._registry.get(name)
                is_last = idx == len(self._steps) - 1
                step_output = output_path if is_last else os.path.join(
                    tmp_dir, f"_step_{idx}_{name}.onnx")

                result = opt.apply(current_input, step_output, **kwargs)
                result["step"] = idx + 1
                result["optimizer"] = name
                step_results.append(result)

                if on_progress:
                    on_progress(idx + 1, len(self._steps), name, result)

                current_input = step_output
        finally:
            # Clean up intermediate files
            shutil.rmtree(tmp_dir, ignore_errors=True)

        final_size = os.path.getsize(output_path) if os.path.isfile(output_path) else 0
        result = {
            "steps": step_results,
            "original_size_mb": round(orig_size / 1024 / 1024, 2),
            "final_size_mb": round(final_size / 1024 / 1024, 2),
            "compression_ratio": round(orig_size / max(final_size, 1), 2),
            "output_path": output_path,
        }

        if validate and sample_dir:
            # Import lazily: validation pulls onnxruntime/cv2, which the pipeline
            # itself does not otherwise require.
            from core.quantizer import (
                validate_optimization, DEFAULT_AGREEMENT_THRESHOLD,
            )
            threshold = (DEFAULT_AGREEMENT_THRESHOLD
                         if validate_threshold is None else validate_threshold)
            try:
                result["validation"] = validate_optimization(
                    original_path=model_path,
                    optimized_path=output_path,
                    sample_dir=sample_dir,
                    max_images=validate_max_images,
                    threshold=threshold,
                )
            except Exception as e:  # noqa: BLE001 — validation must never sink the run
                result["validation"] = {"validated": False, "error": str(e),
                                        "original_path": model_path}

        return result
