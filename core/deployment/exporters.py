"""Backend export helpers for the Deployment module.

Two source kinds are supported:

* A PyTorch checkpoint (``.pt``) — delegated to Ultralytics' own exporter,
  which natively emits onnx / engine / openvino / coreml / torchscript and
  honours opset / imgsz / batch / dynamic / half. This is the primary path and
  matches the project's training env (ultralytics + torch + tensorrt present).
* An existing ``.onnx`` — transformed directly: ONNX precision conversion
  reuses ``core.quantizer``; a TensorRT engine is built via the ``tensorrt``
  Python API; an OpenVINO IR via ``openvino.convert_model``.

Every backend import is lazy and guarded so importing this module is cheap and
safe in the ONNX-runtime-only base environment.
"""
from __future__ import annotations

import os
import shutil
from typing import Callable, Optional

# Source extensions that mean "a PyTorch checkpoint" (vs. an already-built ONNX).
_TORCH_EXTS = {".pt", ".pth"}


def is_torch_checkpoint(model_path: str) -> bool:
    return os.path.splitext(model_path)[1].lower() in _TORCH_EXTS


def file_size_mb(path: str) -> float:
    try:
        return round(os.path.getsize(path) / 1024 / 1024, 2)
    except OSError:
        return 0.0


def dir_size_mb(path: str) -> float:
    """Total size of a directory artifact (OpenVINO IR / CoreML package)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return round(total / 1024 / 1024, 2)


def resolve_imgsz(opts: dict):
    """Normalize an input-size option into what Ultralytics/exporters accept.

    Accepts ``input_size`` as an int (square), "HxW"/"H,W" string, or [H, W]
    list. Returns an int (square) or a [H, W] list. ``0``/empty -> None so the
    model's native size is kept.
    """
    raw = opts.get("input_size", opts.get("imgsz", 0))
    if raw in (0, "", None):
        return None
    if isinstance(raw, (list, tuple)):
        vals = [int(v) for v in raw][:2]
        return vals if len(vals) == 2 and vals[0] != vals[1] else int(vals[0])
    if isinstance(raw, str):
        parts = [p for p in raw.replace("x", ",").replace("X", ",").split(",") if p.strip()]
        nums = [int(float(p)) for p in parts]
        if len(nums) == 2:
            return nums if nums[0] != nums[1] else nums[0]
        return nums[0] if nums else None
    return int(raw)


# ── Ultralytics path (.pt source) ─────────────────────────────────────────
# Maps our target id -> the Ultralytics ``export(format=...)`` string.
ULTRALYTICS_FORMAT = {
    "onnx": "onnx",
    "onnx_fp16": "onnx",
    "onnx_int8": "onnx",
    "tensorrt": "engine",
    "openvino": "openvino",
    "coreml": "coreml",
    "torchscript": "torchscript",
}


def _calibration_data_yaml(calibration_dir: str) -> str:
    """Return a data.yaml path for ultralytics int8 calibration over a dir.

    Ultralytics' int8 export wants a dataset YAML (not a bare image folder) so
    it knows where to pull calibration images from. If the dir already ships a
    data.yaml/data.yml we hand that back; otherwise we synthesize a minimal one
    in a tmp dir pointing train/val at the folder itself. This is best-effort:
    the names map is a single placeholder class because calibration only needs
    representative inputs, not correct labels.

    Returns the dir unchanged on any failure so the export still proceeds.
    """
    try:
        for name in ("data.yaml", "data.yml"):
            existing = os.path.join(calibration_dir, name)
            if os.path.isfile(existing):
                return existing

        try:
            from core.paths import tmp_dir
            out_dir = str(tmp_dir("deploy"))
        except Exception:
            import tempfile
            out_dir = tempfile.mkdtemp(prefix="ssook_deploy_calib_")

        yaml_path = os.path.join(out_dir, "calibration_data.yaml")
        # Lazy import per the stream contract — dataset.py is owned elsewhere.
        from core.training.dataset import _write_yaml
        _write_yaml(yaml_path, {
            "path": os.path.abspath(calibration_dir),
            "train": ".",
            "val": ".",
            "names": {0: "object"},
        })
        return yaml_path
    except Exception:
        # Never block the export on calibration-yaml generation — fall back to
        # passing the dir; ultralytics may still cope or surface its own error.
        return calibration_dir


def ultralytics_export(model_path: str, fmt: str, opts: dict) -> str:
    """Run Ultralytics export and return the produced artifact path.

    Raises ImportError if ultralytics is absent (caller maps to a friendly
    "not installed" result).
    """
    from ultralytics import YOLO  # lazy: heavy, optional

    imgsz = resolve_imgsz(opts)
    kwargs: dict = {"format": fmt}
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    if opts.get("opset"):
        kwargs["opset"] = int(opts["opset"])
    if opts.get("batch"):
        kwargs["batch"] = int(opts["batch"])
    if opts.get("dynamic"):
        kwargs["dynamic"] = True
    if opts.get("half") or opts.get("precision") == "fp16":
        kwargs["half"] = True
    if opts.get("precision") == "int8":
        kwargs["int8"] = True
        if opts.get("calibration_dir"):
            # Ultralytics int8 wants a dataset YAML; synthesize one over a bare
            # image dir (or reuse an existing data.yaml in it). See helper.
            kwargs["data"] = _calibration_data_yaml(opts["calibration_dir"])
    if opts.get("simplify"):
        kwargs["simplify"] = True
    device = opts.get("device", "")
    if device != "":
        kwargs["device"] = device
    elif fmt == "engine":
        kwargs["device"] = 0  # TensorRT build requires a CUDA device

    model = YOLO(model_path)
    produced = model.export(**kwargs)
    return str(produced)


def finalize_artifact(produced: str, output_path: str) -> str:
    """Move an exporter's produced file/dir to the user's output_path.

    Directory artifacts (OpenVINO IR, CoreML .mlpackage) move as a directory.
    Returns the final path actually written.
    """
    produced = os.path.abspath(produced)
    output_path = os.path.abspath(output_path)
    if produced == output_path:
        return produced
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.isdir(produced):
        if os.path.exists(output_path):
            shutil.rmtree(output_path, ignore_errors=True)
        shutil.move(produced, output_path)
    else:
        if os.path.exists(output_path):
            os.remove(output_path)
        shutil.move(produced, output_path)
    return output_path


# ── ONNX -> TensorRT engine (.onnx source) ────────────────────────────────
def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    opts: dict,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Build a serialized TensorRT engine from an ONNX file via the TRT API.

    Raises ImportError if tensorrt is absent. fp16 is enabled via opts.half /
    precision. Dynamic batch uses an optimization profile spanning [1, batch].
    """
    import tensorrt as trt  # lazy: heavy, optional

    if on_progress:
        on_progress(0, 1, "Parsing ONNX into TensorRT network...")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT failed to parse ONNX: {errors}")

    config = builder.create_builder_config()
    workspace_gb = float(opts.get("workspace_gb", 4))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if opts.get("half") or opts.get("precision") == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

    # Optimization profile for dynamic shapes. Only the batch axis (0) is given
    # a range [1..batch]; a FIXED batch is left untouched, and non-batch dynamic
    # dims (e.g. dynamic H/W) are pinned to the requested input size (or 1) since
    # they can't be derived from the batch. Inputs with no dynamic dim are skipped.
    if opts.get("dynamic"):
        max_batch = int(opts.get("batch", 1)) or 1
        imgsz = resolve_imgsz(opts)
        if isinstance(imgsz, int):
            spatial = (imgsz, imgsz)
        elif isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
            spatial = (int(imgsz[0]), int(imgsz[1]))
        else:
            spatial = None
        profile = None
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            shape = list(inp.shape)
            if -1 not in shape:
                continue
            if profile is None:
                profile = builder.create_optimization_profile()
            min_s, opt_s, max_s = [], [], []
            for axis, dim in enumerate(shape):
                if dim != -1:
                    min_s.append(dim); opt_s.append(dim); max_s.append(dim)
                elif axis == 0:
                    min_s.append(1); opt_s.append(max(1, max_batch // 2)); max_s.append(max_batch)
                else:
                    pinned = 1
                    if spatial and axis in (2, 3):
                        pinned = spatial[0] if axis == 2 else spatial[1]
                    min_s.append(pinned); opt_s.append(pinned); max_s.append(pinned)
            profile.set_shape(inp.name, tuple(min_s), tuple(opt_s), tuple(max_s))
        if profile is not None:
            config.add_optimization_profile(profile)

    if on_progress:
        on_progress(0, 1, "Building TensorRT engine (this can take minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned None (check FP16/shape support)")
    parent = os.path.dirname(os.path.abspath(engine_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    if on_progress:
        on_progress(1, 1, "Engine written")
    return {"output_path": engine_path, "output_size_mb": file_size_mb(engine_path)}


# ── ONNX -> OpenVINO IR (.onnx source) ─────────────────────────────────────
def convert_openvino(onnx_path: str, out_xml: str, opts: dict) -> dict:
    """Convert an ONNX model to OpenVINO IR (.xml + .bin).

    Raises ImportError if openvino is absent. fp16 IR is requested via
    compress_to_fp16 when half/precision asks for it.
    """
    import openvino as ov  # lazy: optional

    model = ov.convert_model(onnx_path)
    parent = os.path.dirname(os.path.abspath(out_xml))
    if parent:
        os.makedirs(parent, exist_ok=True)
    compress = bool(opts.get("half") or opts.get("precision") == "fp16")
    ov.save_model(model, out_xml, compress_to_fp16=compress)
    bin_path = os.path.splitext(out_xml)[0] + ".bin"
    return {
        "output_path": out_xml,
        "bin_path": bin_path,
        "output_size_mb": round(file_size_mb(out_xml) + file_size_mb(bin_path), 2),
    }
