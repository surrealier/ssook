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

    Project-wide hash width is the default `size=8` → 64-bit. Every quality
    tool (duplicate / leaky / similarity) and the merger MUST hash at this
    width so the same image yields the same integer and `threshold` (Hamming
    distance) means the same thing everywhere — a 256-bit hash in a 64-bit
    threshold space is four times less sensitive (QUAL-02). Do not pass a
    different `size` unless you also rescale the threshold.

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
    actually see; use `BKTree` below if datasets exceed ~50k images.
    """
    pairs: list[tuple[int, int]] = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hamming(hashes[i], hashes[j]) <= threshold:
                pairs.append((i, j))
    return pairs


def cluster_near_duplicates(hashes: list[int], threshold: int = 10) -> list[int]:
    """Group hashes into connected components of near-duplicates.

    Returns a list of group ids, one per input index, so that two images
    within `threshold` Hamming distance — directly OR transitively through
    a chain — share the same id. This is the difference that makes the
    duplicate detector's "Group" column meaningful: A~B and B~C put A, B, C
    in ONE cluster even when A and C are not within `threshold` of each
    other (QUAL-05). Group ids are small contiguous ints assigned in order
    of first appearance.

    Union-find over `find_near_duplicates`. The pair search is O(n²); for
    large datasets feed candidate pairs from `BKTree` instead.
    """
    n = len(hashes)
    parent = list(range(n))

    def find(x: int) -> int:
        # Path compression keeps repeated lookups near-flat.
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for i, j in find_near_duplicates(hashes, threshold):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # Relabel roots to contiguous ids in first-seen order for stable output.
    relabel: dict[int, int] = {}
    groups: list[int] = []
    for idx in range(n):
        root = find(idx)
        if root not in relabel:
            relabel[root] = len(relabel)
        groups.append(relabel[root])
    return groups


class BKTree:
    """Burkhard-Keller tree over integer hashes under Hamming distance.

    Hamming distance is a metric, so a BK-tree lets us answer "all hashes
    within `max_dist` of h" in roughly O(log n) subtree visits instead of
    the O(n) full scan `find_near_duplicates` does per query. Built once,
    queried many times — the intended backing structure for similarity
    search and large-dataset duplicate clustering.

    Nodes store the index into the original `hashes` list so callers can
    map results back to filenames.
    """

    def __init__(self) -> None:
        self._root: Optional[int] = None  # index of the root hash
        self._hashes: list[int] = []
        # children[node_index] = {distance: child_node_index}
        self._children: dict[int, dict[int, int]] = {}

    def build(self, hashes: list[int]) -> "BKTree":
        """(Re)build the tree from `hashes`. Returns self for chaining."""
        self._hashes = list(hashes)
        self._children = {}
        self._root = None
        for index in range(len(self._hashes)):
            self._add(index)
        return self

    def _add(self, index: int) -> None:
        if self._root is None:
            self._root = index
            self._children[index] = {}
            return
        node = self._root
        while True:
            # Exact duplicates land at edge distance 0 and chain into the
            # 0-subtree, so every matching index is still reachable on query.
            dist = hamming(self._hashes[index], self._hashes[node])
            child = self._children[node].get(dist)
            if child is None:
                self._children[node][dist] = index
                self._children[index] = {}
                return
            node = child

    def query(self, h: int, max_dist: int) -> list[int]:
        """Return indices whose hash is within `max_dist` of `h`."""
        if self._root is None:
            return []
        found: list[int] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            dist = hamming(h, self._hashes[node])
            if dist <= max_dist:
                found.append(node)
            # Triangle inequality: only children at distance in
            # [dist-max_dist, dist+max_dist] from `node` can match.
            lo, hi = dist - max_dist, dist + max_dist
            for edge, child in self._children[node].items():
                if lo <= edge <= hi:
                    stack.append(child)
        return found
