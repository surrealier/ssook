"""Class-label fallback catalogue.

Many ONNX exports drop the `class_names` metadata. ssook then shows
`cls0, cls1, …` which is useless for evaluation/visualisation. This
module supplies well-known label maps for the most common output
sizes so the user can auto-fill labels with one click.

Catalogue keyed by canonical name. Get list via `get(name)` or auto-
pick from class count via `suggest(num_classes)`.
"""
from __future__ import annotations

from typing import Optional


# COCO 2017 — 80 detection classes, contiguous ids 0..79.
COCO80: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Pascal VOC 2007/2012 — 20 detection classes.
VOC20: list[str] = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Pascal VOC segmentation — 21 (background + 20 classes).
VOC21_SEG: list[str] = ["background"] + VOC20

# COCO panoptic 91 (kept for completeness — some models use this older id space).
# Only the 80 thing-classes are populated; stuff classes mark with "_stuff_<id>".
_COCO91_THING = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus",
    7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant",
    13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat",
    18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
    24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag",
    32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard",
    37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
    41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl",
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
    77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink",
    82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors",
    88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}
COCO91: list[str] = [_COCO91_THING.get(i, f"_unused_{i}") for i in range(91)]


_CATALOG: dict[str, list[str]] = {
    "coco80": COCO80,
    "coco91": COCO91,
    "voc20": VOC20,
    "voc21_seg": VOC21_SEG,
}


def list_catalogs() -> list[dict]:
    """List available label catalogues for the Settings UI."""
    return [
        {"name": k, "num_classes": len(v),
         "description": _DESC.get(k, ""), "preview": v[:10]}
        for k, v in _CATALOG.items()
    ] + [
        {"name": "imagenet1k", "num_classes": 1000,
         "description": _DESC["imagenet1k"], "preview": _imagenet1k_preview()},
    ]


def get(name: str) -> Optional[list[str]]:
    """Return the labels for `name`, or None if unknown.

    ImageNet1k is loaded lazily because it's 1000 entries.
    """
    if name == "imagenet1k":
        return _load_imagenet1k()
    return _CATALOG.get(name)


def suggest(num_classes: int) -> Optional[str]:
    """Suggest a catalogue name for an unlabelled model with N classes."""
    if num_classes == 80: return "coco80"
    if num_classes == 91: return "coco91"
    if num_classes == 20: return "voc20"
    if num_classes == 21: return "voc21_seg"
    if num_classes == 1000: return "imagenet1k"
    return None


def as_class_names(name: str) -> Optional[dict[int, str]]:
    """Catalogue formatted for ssook's `class_names: dict[int,str]`."""
    labels = get(name)
    if labels is None:
        return None
    return {i: lbl for i, lbl in enumerate(labels)}


_DESC = {
    "coco80": "COCO 2017 — 80 object detection classes",
    "coco91": "COCO 91-id space (includes legacy ids)",
    "voc20": "Pascal VOC — 20 detection classes",
    "voc21_seg": "Pascal VOC segmentation — background + 20 classes",
    "imagenet1k": "ImageNet 1000-class classification (synset order)",
}


def _imagenet1k_preview() -> list[str]:
    return [
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
        "electric ray", "stingray", "cock", "hen", "ostrich",
    ]


_IMAGENET1K_CACHE: Optional[list[str]] = None


def _load_imagenet1k() -> list[str]:
    """ImageNet-1k synset readable names. Lazy-loaded from a packaged file
    if present; otherwise falls back to the embedded condensed list.
    """
    global _IMAGENET1K_CACHE
    if _IMAGENET1K_CACHE is not None:
        return _IMAGENET1K_CACHE

    # Try a packaged data file first (so users can swap their own list in).
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "data", "imagenet1k_labels.txt")
    if os.path.isfile(candidate):
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                labels = [line.strip() for line in f if line.strip()]
            if len(labels) == 1000:
                _IMAGENET1K_CACHE = labels
                return labels
        except OSError:
            pass

    # Fallback: generate placeholder synset ids so eval still works on the
    # index axis. Users who want real names should drop their list into
    # core/data/imagenet1k_labels.txt — and that's documented in the
    # backend-runbook.
    _IMAGENET1K_CACHE = [f"n{idx:08d}" for idx in range(1000)]
    return _IMAGENET1K_CACHE
