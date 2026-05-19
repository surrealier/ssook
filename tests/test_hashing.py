"""Tests for core.hashing — dHash consistency between merger and quality."""
import numpy as np


def _gray_image(value: int) -> np.ndarray:
    return np.full((64, 64, 3), value, dtype=np.uint8)


def test_dhash_stable_for_identical_input():
    from core.hashing import compute_dhash
    img = _gray_image(128)
    h1 = compute_dhash(img)
    h2 = compute_dhash(img)
    assert h1 == h2
    assert isinstance(h1, int)


def test_dhash_differs_for_clearly_different():
    from core.hashing import compute_dhash, hamming
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = np.zeros((64, 64, 3), dtype=np.uint8)
    # Embed a horizontal stripe so the row diffs are non-zero.
    b[:, 32:] = 255
    ha = compute_dhash(a)
    hb = compute_dhash(b)
    # Flat image vs striped should differ noticeably.
    assert hamming(ha, hb) > 0


def test_dhash_accepts_path_and_array(tmp_path):
    import cv2
    from core.hashing import compute_dhash
    img = _gray_image(50)
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), img)
    h_array = compute_dhash(img)
    h_path = compute_dhash(str(p))
    # PNG round-trip should reproduce the same hash (or be ≤ 2 bits off).
    from core.hashing import hamming
    assert hamming(h_array, h_path) <= 2


def test_find_near_duplicates():
    from core.hashing import find_near_duplicates
    hashes = [0b1100, 0b1101, 0b1111_1111]
    pairs = find_near_duplicates(hashes, threshold=1)
    assert (0, 1) in pairs
    assert (0, 2) not in pairs
