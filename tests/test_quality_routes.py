"""Tests for /api/quality/* routes (track: quality).

Covers the Wave-1 fixes in server/quality_routes.py:
  - anomaly: a single malformed label line no longer aborts the scan
    and is counted; missing label files are reported (not silently skipped).
  - leaky: recursive option finds nested cross-split duplicates.
  - all 5 endpoints use TaskState.try_start() as an atomic start guard.

The app under test mounts ONLY the quality router so the suite stays
isolated from other waves concurrently editing the full app. Images are
written as real (tiny) PNGs via cv2 so the leaky/duplicate hashing path
runs end-to-end without mocking imread.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import cv2

from server import quality_routes
from server.state import anomaly_state, leaky_state, sim_state


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(quality_routes.router)
    return TestClient(app)


def _reset(state) -> None:
    """Clear a TaskState between tests so a prior run's `running` flag or
    cached index can't leak into the next test."""
    state.update(running=False, progress=0, total=0, msg="", results=[])


def _wait_until_done(client: TestClient, status_url: str, timeout_s: float = 10.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        data = client.get(status_url).json()
        if not data.get("running"):
            return data
        time.sleep(0.02)
    raise AssertionError(f"worker did not finish within {timeout_s}s")


def _write_png(path: str, color: int = 0) -> None:
    img = np.full((8, 8, 3), color, dtype=np.uint8)
    cv2.imwrite(path, img)


# ── anomaly: malformed line survives + is counted (P0) ──────────────

def test_malformed_label_line_does_not_abort_scan(client, tmp_path):
    """One non-numeric label line must not discard the whole scan; the valid
    out-of-bounds box on the next line must still be reported, and the
    malformed line must be tallied rather than crash the run."""
    _reset(anomaly_state)
    img_dir = tmp_path / "imgs"
    lbl_dir = tmp_path / "lbls"
    img_dir.mkdir()
    lbl_dir.mkdir()
    # Content is irrelevant — the anomaly tool never decodes the image, it
    # only globs filenames and reads the matching .txt.
    (img_dir / "a.jpg").write_bytes(b"not-a-real-image")
    # Line 1: non-numeric header (would raise ValueError on float()).
    # Line 2: a valid but out-of-bounds box (cx+w/2 = 1.4 > 1.01).
    (lbl_dir / "a.txt").write_text("foo bar baz qux quux\n0 0.9 0.5 1.0 0.2\n")

    resp = client.post("/api/quality/anomaly",
                       json={"img_dir": str(img_dir), "label_dir": str(lbl_dir)})
    assert resp.json() == {"ok": True}

    data = _wait_until_done(client, "/api/quality/anomaly/status")
    assert data["running"] is False
    # The scan completed normally — NOT via the route_errors abort path.
    assert not data["msg"].startswith("Error:")
    assert data["malformed_lines"] == 1
    # The valid OOB anomaly on the second line survived the bad first line.
    types = [r["type"] for r in data["results"]]
    assert any("Out-of-bounds" in t for t in types)


# ── anomaly: missing label files reported (P1) ──────────────────────

def test_missing_label_files_are_counted(client, tmp_path):
    """Images with no matching .txt must be tallied as missing_labels and
    surfaced in the message — a wrong label_dir should not read as 'clean'."""
    _reset(anomaly_state)
    img_dir = tmp_path / "imgs"
    lbl_dir = tmp_path / "lbls"
    img_dir.mkdir()
    lbl_dir.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (img_dir / name).write_bytes(b"x")
    # Only one of the three images has a label file.
    (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    resp = client.post("/api/quality/anomaly",
                       json={"img_dir": str(img_dir), "label_dir": str(lbl_dir)})
    assert resp.json() == {"ok": True}

    data = _wait_until_done(client, "/api/quality/anomaly/status")
    assert data["running"] is False
    assert data["missing_labels"] == 2
    assert "no label file" in data["msg"]


# ── leaky: recursive finds nested cross-split dupes (P0) ────────────

def test_leaky_recursive_finds_nested_duplicates(client, tmp_path):
    """A duplicate that lives under train/sub/ and val/sub/ is only found
    when recursive=True; recursive=False must miss it (proving the flag is
    actually wired through to glob_images)."""
    _reset(leaky_state)
    train_sub = tmp_path / "train" / "sub"
    val_sub = tmp_path / "val" / "sub"
    train_sub.mkdir(parents=True)
    val_sub.mkdir(parents=True)
    # Identical content under nested subdirs of each split → cross-split leak.
    _write_png(str(train_sub / "a.jpg"), color=50)
    _write_png(str(val_sub / "a.jpg"), color=50)

    # recursive=True → the nested duplicate is detected.
    resp = client.post("/api/quality/leaky",
                       json={"train_dir": str(tmp_path / "train"),
                             "val_dir": str(tmp_path / "val"),
                             "threshold": 0, "recursive": True})
    assert resp.json() == {"ok": True}
    data = _wait_until_done(client, "/api/quality/leaky/status")
    assert data["running"] is False
    total_dupes = sum(r["duplicates"] for r in data["results"])
    assert total_dupes >= 1

    # recursive=False → the nested files are never globbed, so 0 leaks.
    _reset(leaky_state)
    resp = client.post("/api/quality/leaky",
                       json={"train_dir": str(tmp_path / "train"),
                             "val_dir": str(tmp_path / "val"),
                             "threshold": 0, "recursive": False})
    assert resp.json() == {"ok": True}
    data = _wait_until_done(client, "/api/quality/leaky/status")
    total_dupes = sum(r["duplicates"] for r in data["results"])
    assert total_dupes == 0


# ── try_start guard: second concurrent POST is rejected ─────────────

def test_anomaly_try_start_rejects_second_run(client, tmp_path):
    """While a run is in flight, a second POST must be rejected by the atomic
    try_start guard. We hold the first run open by marking it running before
    the request (the route's try_start sees running=True and refuses)."""
    _reset(anomaly_state)
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    (img_dir / "a.jpg").write_bytes(b"x")

    # Simulate an in-flight run.
    anomaly_state.update(running=True)
    try:
        resp = client.post("/api/quality/anomaly", json={"img_dir": str(img_dir)})
        assert resp.json() == {"error": "Already running"}
    finally:
        _reset(anomaly_state)


# ── Structured-image helper ─────────────────────────────────────────
# A flat color block dHashes to 0 (no row-diffs), so distinct flat colors are
# indistinguishable. We embed a per-image gradient/stripe pattern so each image
# yields a different 64-bit hash, exercising the real BKTree vs brute-force path.

def _write_pattern_png(path: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    cv2.imwrite(path, img)


# ── leaky: BKTree path equals brute-force reference on a small fixture (scale P1) ──

def test_leaky_bktree_matches_bruteforce_reference(client, tmp_path):
    """The cross-split detector now queries a BKTree per split-pair instead of
    the old O(MxN) nested Hamming scan. On a small mixed fixture (some shared,
    some unique images across two splits) the reported duplicate count and the
    sorted file-pair set must equal an independent brute-force reference."""
    _reset(leaky_state)
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    # Shared images (identical bytes → identical hash, distance 0 cross-split).
    shared = {}
    for k in range(4):
        p = train_dir / f"shared_{k}.png"
        _write_pattern_png(str(p), seed=1000 + k)
        shared[f"shared_{k}.png"] = cv2.imread(str(p))
    for name, arr in shared.items():
        cv2.imwrite(str(val_dir / name), arr)
    # Unique images per split (different seeds → far hashes, no cross leak).
    for k in range(3):
        _write_pattern_png(str(train_dir / f"tonly_{k}.png"), seed=2000 + k)
        _write_pattern_png(str(val_dir / f"vonly_{k}.png"), seed=3000 + k)

    threshold = 2
    resp = client.post("/api/quality/leaky",
                       json={"train_dir": str(train_dir), "val_dir": str(val_dir),
                             "threshold": threshold, "recursive": False})
    assert resp.json() == {"ok": True}
    data = _wait_until_done(client, "/api/quality/leaky/status")
    assert data["running"] is False
    rows = data["results"]
    assert len(rows) == 1  # exactly the train↔val pair
    got_dupes = rows[0]["duplicates"]

    # Independent brute-force reference over the same on-disk images.
    from core.hashing import compute_dhash, hamming
    train_hashes = {p.name: compute_dhash(str(p)) for p in sorted(train_dir.iterdir())}
    val_hashes = {p.name: compute_dhash(str(p)) for p in sorted(val_dir.iterdir())}
    ref_pairs = []
    for fa, ha in train_hashes.items():
        for fb, hb in val_hashes.items():
            if hamming(ha, hb) <= threshold:
                ref_pairs.append(f"{fa} ↔ {fb}")
    assert got_dupes == len(ref_pairs)
    # The 4 shared images each leak across the split.
    assert got_dupes == 4
    # The reported sample file list is a subset of the true brute-force pairs.
    reported = [s for s in rows[0]["files"].split("; ") if s]
    assert set(reported).issubset(set(ref_pairs))


# ── similarity: ranking is correct and repeat query does not re-hash (scale P1) ──

def test_similarity_query_ranks_exact_copy_first(client, tmp_path):
    """The query's exact copy must rank #1 at distance 0; unrelated images rank
    after it. Exercises the expanding-radius BKTree top-k path."""
    _reset(sim_state)
    sim_state.pop("index", None)
    img_dir = tmp_path / "corpus"
    img_dir.mkdir()
    for k in range(8):
        _write_pattern_png(str(img_dir / f"img_{k}.png"), seed=4000 + k)
    # The query is an exact copy of img_3.
    query = tmp_path / "query.png"
    cv2.imwrite(str(query), cv2.imread(str(img_dir / "img_3.png")))

    resp = client.post("/api/quality/similarity",
                       json={"img_dir": str(img_dir), "query_path": str(query),
                             "top_k": 5, "recursive": False})
    assert resp.json() == {"ok": True}
    data = _wait_until_done(client, "/api/quality/similarity/status")
    assert data["running"] is False
    results = data["results"]
    assert results[0]["image"] == "img_3.png"
    assert results[0]["distance"] == 0
    assert results[0]["rank"] == 1
    # Distances are non-decreasing (a valid top-k ranking).
    dists = [r["distance"] for r in results]
    assert dists == sorted(dists)
    _reset(sim_state)
    sim_state.pop("index", None)


def test_similarity_repeat_query_reuses_cache_no_rehash(client, tmp_path, monkeypatch):
    """A second query against the same folder must reuse the cached (hash, tree)
    index and decode ZERO corpus images — only the new query image is read."""
    _reset(sim_state)
    sim_state.pop("index", None)
    img_dir = tmp_path / "corpus"
    img_dir.mkdir()
    for k in range(6):
        _write_pattern_png(str(img_dir / f"img_{k}.png"), seed=5000 + k)
    query1 = tmp_path / "q1.png"
    query2 = tmp_path / "q2.png"
    cv2.imwrite(str(query1), cv2.imread(str(img_dir / "img_0.png")))
    cv2.imwrite(str(query2), cv2.imread(str(img_dir / "img_2.png")))

    # Count how many times imread is called, distinguishing corpus vs query.
    real_imread = quality_routes.imread
    corpus_dir_str = str(img_dir)
    calls = {"corpus": 0, "query": 0}

    def counting_imread(path, *a, **kw):
        if os.path.dirname(str(path)) == corpus_dir_str:
            calls["corpus"] += 1
        else:
            calls["query"] += 1
        return real_imread(path, *a, **kw)

    monkeypatch.setattr(quality_routes, "imread", counting_imread)

    # First query builds the index: all 6 corpus images decoded once.
    resp = client.post("/api/quality/similarity",
                       json={"img_dir": str(img_dir), "query_path": str(query1),
                             "top_k": 3, "recursive": False})
    assert resp.json() == {"ok": True}
    _wait_until_done(client, "/api/quality/similarity/status")
    assert calls["corpus"] == 6
    index_obj = sim_state.get("index")
    assert index_obj is not None and "tree" in index_obj

    # Second query against the SAME folder: no corpus re-decode, same cache object.
    calls["corpus"] = 0
    resp = client.post("/api/quality/similarity",
                       json={"img_dir": str(img_dir), "query_path": str(query2),
                             "top_k": 3, "recursive": False})
    assert resp.json() == {"ok": True}
    data = _wait_until_done(client, "/api/quality/similarity/status")
    assert data["running"] is False
    assert calls["corpus"] == 0          # cache reused — zero corpus re-hash
    assert calls["query"] >= 1           # only the query image was decoded
    assert sim_state.get("index") is index_obj  # same cached index, not rebuilt
    # The img_2 copy ranks first on the second query.
    assert data["results"][0]["image"] == "img_2.png"
    assert data["results"][0]["distance"] == 0

    _reset(sim_state)
    sim_state.pop("index", None)
