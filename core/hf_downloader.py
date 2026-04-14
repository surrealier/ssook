"""HuggingFace Hub ONNX model search & download."""
import os
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent.parent / "models" / "hf_cache"

# HF task → ssook model_type mapping
TASK_MAP = {
    "object-detection": "yolo",
    "image-classification": "cls_vit",
    "image-segmentation": "seg_custom",
    "zero-shot-image-classification": "clip_custom",
    "visual-question-answering": "vlm_vqa",
    "image-to-text": "vlm_caption",
    "feature-extraction": "emb_custom",
}


def _ensure_hub():
    try:
        from huggingface_hub import HfApi, hf_hub_download  # noqa
        return True
    except ImportError:
        return False


def search_models(query: str, task: str = "", limit: int = 20) -> list[dict]:
    """Search HuggingFace Hub for ONNX models."""
    if not _ensure_hub():
        return []
    from huggingface_hub import HfApi
    api = HfApi()
    kwargs = {"search": query, "limit": limit, "sort": "downloads", "direction": -1}
    if task:
        kwargs["pipeline_tag"] = task
    results = []
    for m in api.list_models(**kwargs):
        # Check if repo likely has ONNX files
        tags = m.tags or []
        has_onnx = "onnx" in tags or "onnxruntime" in tags
        results.append({
            "repo_id": m.id,
            "task": m.pipeline_tag or "",
            "downloads": m.downloads or 0,
            "has_onnx": has_onnx,
            "tags": tags[:10],
            "ssook_type": TASK_MAP.get(m.pipeline_tag, ""),
        })
    return results


def list_onnx_files(repo_id: str) -> list[str]:
    """List .onnx files in a HuggingFace repo."""
    if not _ensure_hub():
        return []
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id)
        return [f for f in files if f.endswith(".onnx")]
    except Exception:
        return []


def download_model(repo_id: str, filename: str) -> str:
    """Download a single ONNX file from HuggingFace Hub. Returns local path."""
    if not _ensure_hub():
        raise RuntimeError("huggingface-hub not installed. Run: pip install huggingface-hub")
    from huggingface_hub import hf_hub_download
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
        repo_id=repo_id, filename=filename,
        cache_dir=str(_CACHE_DIR), local_dir=str(_CACHE_DIR / repo_id.replace("/", "_")),
    )
    return local


def list_cached() -> list[dict]:
    """List already downloaded models in cache."""
    if not _CACHE_DIR.is_dir():
        return []
    results = []
    for onnx in _CACHE_DIR.rglob("*.onnx"):
        results.append({
            "path": str(onnx),
            "name": onnx.name,
            "size_mb": round(onnx.stat().st_size / 1024 / 1024, 1),
        })
    return results
