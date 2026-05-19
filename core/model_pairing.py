"""Auto-pair CLIP image/text encoder ONNX files.

VLM v1 needs both an image encoder AND a text encoder ONNX. Asking the
user to pick both is friction — most CLIP exports keep predictable
filenames (`*image_encoder*.onnx` next to `*text_encoder*.onnx`). This
module formalises the heuristic:

1. **Name-based**: replace common image-encoder tokens with text-encoder
   ones and check if the resulting file exists.
2. **Signature-based**: scan the same directory and use the classifier
   to find the partner.

Public API:
    find_partner(path) -> dict[partner_path?, role, confidence, reason]
"""
from __future__ import annotations

import os
import re
from typing import Optional

from core.model_classifier import classify

# (substring in source name, replacement) — try left-to-right.
_NAME_PAIRS = [
    ("image_encoder", "text_encoder"),
    ("image-encoder", "text-encoder"),
    ("img_encoder", "text_encoder"),
    ("vision_encoder", "text_encoder"),
    ("vision_model", "text_model"),
    ("visual", "textual"),
    ("image", "text"),
    ("img", "txt"),
    ("vision", "text"),
]


def _flip_role(stem: str) -> Optional[str]:
    """Given a filename stem of one role, return the partner stem."""
    low = stem.lower()
    for a, b in _NAME_PAIRS:
        if a in low and b not in low:
            return re.sub(re.escape(a), b, low, count=1, flags=re.IGNORECASE)
        if b in low and a not in low:
            return re.sub(re.escape(b), a, low, count=1, flags=re.IGNORECASE)
    return None


def _my_role(stem: str) -> Optional[str]:
    """Crude name-based role detection — used as a tiebreaker."""
    low = stem.lower()
    text_markers = ("text_encoder", "text-encoder", "txt", "textual", "text_model")
    image_markers = ("image_encoder", "image-encoder", "img_encoder", "vision_encoder",
                     "vision_model", "visual", "image", "img", "vision")
    if any(m in low for m in text_markers):
        return "text"
    if any(m in low for m in image_markers):
        return "image"
    return None


def find_partner(path: str) -> dict:
    """Find a partner ONNX in the same directory.

    Returns dict with keys:
      role:        "image" | "text" | "unknown"
      partner_path: str | None
      confidence:  float
      reason:      str
    """
    if not path or not os.path.isfile(path):
        return {"role": "unknown", "partner_path": None,
                "confidence": 0.0, "reason": "Source file not found"}

    directory = os.path.dirname(os.path.abspath(path))
    stem, ext = os.path.splitext(os.path.basename(path))

    # 1. Try direct name substitution.
    flipped = _flip_role(stem)
    if flipped:
        for cand in (flipped + ext, flipped + ext.upper()):
            for actual in os.listdir(directory):
                if actual.lower() == cand.lower() and actual.lower() != os.path.basename(path).lower():
                    role = _my_role(stem) or "unknown"
                    return {
                        "role": role,
                        "partner_path": os.path.join(directory, actual),
                        "confidence": 0.9,
                        "reason": f"Filename substitution matched: {actual}",
                    }

    # 2. Classifier-driven search across the directory.
    my_info = classify(path)
    my_task = my_info.get("task_type")
    my_role = None
    want_task = None
    if my_task == "vlm_image_encoder":
        my_role, want_task = "image", "vlm_text_encoder"
    elif my_task == "vlm_text_encoder":
        my_role, want_task = "text", "vlm_image_encoder"
    elif _my_role(stem) == "image":
        my_role, want_task = "image", "vlm_text_encoder"
    elif _my_role(stem) == "text":
        my_role, want_task = "text", "vlm_image_encoder"

    if want_task:
        for entry in sorted(os.listdir(directory)):
            if not entry.lower().endswith(".onnx"):
                continue
            full = os.path.join(directory, entry)
            if os.path.samefile(full, path) if os.path.exists(full) else False:
                continue
            cand_info = classify(full)
            if cand_info.get("task_type") == want_task:
                return {
                    "role": my_role,
                    "partner_path": full,
                    "confidence": 0.7,
                    "reason": f"Signature match: {entry} classified as {want_task}",
                }

    return {
        "role": my_role or "unknown",
        "partner_path": None,
        "confidence": 0.0,
        "reason": "No partner ONNX found in same directory",
    }
