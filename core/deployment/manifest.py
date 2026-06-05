"""Deployment manifest writer.

Every export drops a ``<artifact>.deploy.json`` next to the artifact recording
exactly how to run it: the target execution provider, the recommended
onnxruntime session/provider options, the input shape / batch / opset / class
names, the source-model fingerprint, and an environment snapshot. This makes a
produced artifact self-describing for whoever deploys it.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("ssook.deployment")


def _session_recommendation(provider: Optional[str], provider_options: dict) -> dict:
    """Recommended onnxruntime session config for the target provider.

    Encodes the few provider quirks the app already knows about (DirectML
    must disable the memory-pattern optimizer — see model_loader._create_session).
    """
    if not provider:
        return {}
    providers = [[provider, provider_options]] if provider_options else [provider]
    # Always keep CPU as a fallback so a missing GPU/EP degrades instead of failing.
    providers.append("CPUExecutionProvider")
    rec: dict = {"providers": providers}
    if provider == "DmlExecutionProvider":
        rec["session_options"] = {"enable_mem_pattern": False}
        rec["note"] = "DirectML requires enable_mem_pattern=False."
    return rec


def build_manifest(
    model_path: str,
    target,
    opts: dict,
    export_result: dict,
    model_meta: Optional[dict] = None,
    actual_opset: Optional[int] = None,
) -> dict:
    """Assemble the manifest dict from the export inputs and result.

    ``actual_opset`` (when known, read back from the produced .onnx) overrides
    the requested opset in ``config.opset`` so the manifest reflects what the
    artifact really carries — opset re-spec can silently keep the original.
    """
    try:
        from core.run_record import env_snapshot, model_meta as fingerprint
        env = env_snapshot()
        source_fp = fingerprint(model_path) or {}
    except Exception as e:  # run_record is best-effort provenance, never fatal
        log.debug("manifest provenance unavailable: %s", e)
        env, source_fp = {}, {}

    artifact_path = export_result.get("output_path", "")
    provider = export_result.get("provider", getattr(target, "provider", None))
    provider_options = export_result.get("provider_options", {})

    manifest = {
        "schema": "ssook.deploy/1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_model": {"path": model_path, **source_fp},
        "target": {
            "name": getattr(target, "name", export_result.get("target", "")),
            "category": getattr(target, "category", ""),
            "provider": provider,
            "output_ext": getattr(target, "output_ext", ""),
        },
        "artifact": {
            "path": artifact_path,
            "size_mb": export_result.get("output_size_mb"),
        },
        "config": {
            "opset": actual_opset if actual_opset is not None else opts.get("opset"),
            "batch": opts.get("batch"),
            "input_size": opts.get("input_size"),
            "dynamic": bool(opts.get("dynamic")),
            "precision": opts.get("precision", "fp32"),
        },
        "runtime": _session_recommendation(provider, provider_options),
        "classes": opts.get("class_names") or model_meta and model_meta.get("names") or {},
        "environment": env,
    }
    return manifest


def write_manifest(manifest: dict, artifact_path: str) -> str:
    """Write the manifest as ``<artifact>.deploy.json``. Returns its path.

    For a directory artifact (OpenVINO IR / CoreML package) the manifest goes
    *inside* the directory as ``deploy.json``.
    """
    if os.path.isdir(artifact_path):
        manifest_path = os.path.join(artifact_path, "deploy.json")
    else:
        manifest_path = os.path.splitext(artifact_path)[0] + ".deploy.json"
    parent = os.path.dirname(os.path.abspath(manifest_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path
