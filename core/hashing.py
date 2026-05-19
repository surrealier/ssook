"""Perceptual hashing utilities for ssook.

`compute_dhash` and `hamming` were previously copy-pasted across
`server/data_routes.py` (merger) and `server/quality_routes.py`
(duplicate detector / leaky split / similarity search). They drift over
time — one used 9×8 grayscale, the other 8×9 — so the same image got
different hashes depending on which route called it.

This module is the single source of truth.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def compute_dhash(image_or_path, size: int = 8) -> Optional[int]:
    """Difference hash for an image. Returns None when the image cannot
    be read. `size` controls the precision (8 → 64-bit hash).

    Accepts either a numpy ndarray (BGR or grayscale) or a filesystem
    path. The classic 9×size resize → row-diff trick.
    """
    if isinstance(image_or_path, str):
        img = cv2.imread(image_or_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
    else:
        img = image_or_path
        if img is None:
            return None
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    """Bit-count distance between two integer hashes."""
    return bin(a ^ b).count("1")


def find_near_duplicates(hashes: list[int], threshold: int = 10) -> list[tuple[int, int]]:
    """Return index pairs (i, j) with i<j whose Hamming distance is
    within `threshold`. O(n²) — fine for the local-app dataset sizes we
    actually see; consider VP-tree if datasets exceed ~50k images.
    """
    pairs: list[tuple[int, int]] = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hamming(hashes[i], hashes[j]) <= threshold:
                pairs.append((i, j))
    return pairs
