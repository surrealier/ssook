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


def test_cluster_near_duplicates_transitive_chain():
    """A~B and B~C must share ONE group id even when A and C are not
    directly within threshold (transitive clustering — QUAL-05)."""
    from core.hashing import cluster_near_duplicates
    # 0b0000 ~ 0b0001 (d=1), 0b0001 ~ 0b0011 (d=1), but 0b0000 ~ 0b0011 (d=2).
    hashes = [0b0000, 0b0001, 0b0011]
    groups = cluster_near_duplicates(hashes, threshold=1)
    assert groups[0] == groups[1] == groups[2]


def test_cluster_near_duplicates_separate_components():
    from core.hashing import cluster_near_duplicates
    # Two far-apart clusters: {0,1} and {2,3}.
    hashes = [0b0000_0000, 0b0000_0001, 0b1111_1111, 0b1111_1110]
    groups = cluster_near_duplicates(hashes, threshold=1)
    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    assert groups[0] != groups[2]
    # Group ids are contiguous starting at 0.
    assert set(groups) == {0, 1}


def test_cluster_near_duplicates_no_matches_each_singleton():
    from core.hashing import cluster_near_duplicates
    hashes = [0b0000, 0b1111, 0b1010_1010]
    groups = cluster_near_duplicates(hashes, threshold=0)
    assert len(set(groups)) == 3


def test_cluster_near_duplicates_empty():
    from core.hashing import cluster_near_duplicates
    assert cluster_near_duplicates([], threshold=5) == []


def test_bktree_query_returns_within_distance():
    from core.hashing import BKTree, hamming
    hashes = [0b0000_0000, 0b0000_0001, 0b0000_0011, 0b1111_1111]
    tree = BKTree().build(hashes)
    # Query the first hash at radius 1 → indices 0 and 1 (distances 0 and 1).
    found = set(tree.query(0b0000_0000, max_dist=1))
    assert found == {0, 1}
    # Radius wide enough to catch everything.
    found_all = set(tree.query(0b0000_0000, max_dist=8))
    assert found_all == {0, 1, 2, 3}
    # Every returned index is genuinely within range (no false positives).
    for idx in tree.query(0b0000_0011, max_dist=2):
        assert hamming(0b0000_0011, hashes[idx]) <= 2


def test_bktree_matches_bruteforce():
    """BK-tree query must return exactly the brute-force in-range set."""
    import random
    from core.hashing import BKTree, hamming
    random.seed(1234)
    hashes = [random.getrandbits(64) for _ in range(200)]
    tree = BKTree().build(hashes)
    for _ in range(20):
        q = random.getrandbits(64)
        r = random.randint(0, 10)
        tree_hits = set(tree.query(q, max_dist=r))
        brute = {i for i, h in enumerate(hashes) if hamming(q, h) <= r}
        assert tree_hits == brute


def test_bktree_empty():
    from core.hashing import BKTree
    tree = BKTree().build([])
    assert tree.query(0, max_dist=5) == []
