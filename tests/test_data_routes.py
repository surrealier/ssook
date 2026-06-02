"""Tests for /api/data/* routes (track: data — explorer/splitter/converter/
remapper/merger/sampler).

Focus on the Wave-1 backend fixes:
- converter COCO->YOLO path traversal via untrusted file_name stem
- splitter includes .webp/.tif images (canonical extension set)
- sampler recursive basename collision preserves both images + labels
- TaskState.try_start rejects a second concurrent start

The background worker is normally dispatched via executor.submit; the tests
patch data_routes.executor with an inline runner so each POST completes
synchronously and we can assert on the final state without polling.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import server.data_routes as dr
from server.state import (
    converter_state,
    splitter_state,
    sampler_state,
    remapper_state,
)


class _InlineExecutor:
    """Runs submitted callables synchronously so the worker finishes before
    the POST coroutine returns — no status polling needed in tests."""

    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)
        return None


@pytest.fixture
def inline_executor(monkeypatch):
    monkeypatch.setattr(dr, "executor", _InlineExecutor())


def _run_coro(coro):
    """Drive an async route handler to completion from a sync test."""
    import asyncio
    return asyncio.run(coro)


def _png_bytes(width: int = 8, height: int = 8) -> bytes:
    """A real tiny PNG so cv2.imread (used by the converter to read dims) works."""
    import numpy as np
    import cv2
    img = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


# ── try_start guard (concurrency) ──────────────────────────────────────────
def test_try_start_rejects_second_concurrent_start():
    """Two near-simultaneous starts on a TaskState must yield exactly one winner.

    Mirrors the TOCTOU the data routes' old read-then-write guard had; the
    routes now delegate to this exact primitive.
    """
    from server.state import TaskState

    s = TaskState()
    results: list[bool] = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def starter():
        barrier.wait()
        ok = s.try_start(msg="go", results={})
        with lock:
            results.append(ok)

    threads = [threading.Thread(target=starter) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(results) == 1
    assert s["running"] is True


def test_converter_second_post_rejected_while_running(inline_executor, tmp_path, monkeypatch):
    """A converter POST while one is already running returns 'Already running'
    and does NOT clobber the in-flight run's state."""
    monkeypatch.setattr(converter_state, "running", False, raising=False)
    converter_state.update(running=True)  # simulate an in-flight run
    try:
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        req = dr.ConverterRequest(input_dir=str(in_dir), output_dir=str(out_dir),
                                  from_fmt="YOLO", to_fmt="COCO JSON")
        resp = _run_coro(dr.data_converter(req))
        assert resp == {"error": "Already running"}
    finally:
        converter_state.update(running=False)


# ── converter COCO->YOLO path traversal (P0 security) ───────────────────────
def test_converter_coco_to_yolo_blocks_path_traversal(inline_executor, tmp_path):
    """A COCO file_name of '../escape' must be reduced to its basename and land
    safely inside output_dir — never write a sibling file above it."""
    import json

    in_dir = tmp_path / "coco_in"
    out_dir = tmp_path / "coco_out"
    in_dir.mkdir()
    out_dir.mkdir()

    coco = {
        "images": [
            {"id": 1, "file_name": "../../escape.png", "width": 100, "height": 100},
            {"id": 2, "file_name": "safe.png", "width": 100, "height": 100},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20]},
            {"id": 2, "image_id": 2, "category_id": 0, "bbox": [10, 10, 20, 20]},
        ],
    }
    (in_dir / "instances.json").write_text(json.dumps(coco))

    req = dr.ConverterRequest(input_dir=str(in_dir), output_dir=str(out_dir),
                              from_fmt="COCO JSON", to_fmt="YOLO")
    resp = _run_coro(dr.data_converter(req))
    assert resp == {"ok": True}

    snap = converter_state.snapshot()
    assert snap["msg"] == "Complete", snap

    # The traversal segments were stripped: the file lands as escape.txt INSIDE
    # out_dir, and nothing was written above it.
    assert not (tmp_path / "escape.txt").exists()
    assert not (in_dir.parent / "escape.txt").exists()
    written = sorted(p.name for p in out_dir.rglob("*.txt"))
    assert written == ["escape.txt", "safe.txt"], written
    # All written files are genuinely under out_dir (no escape via symlink/abs).
    for p in out_dir.rglob("*.txt"):
        assert os.path.realpath(p).startswith(os.path.realpath(str(out_dir)))


def test_converter_coco_to_yolo_empty_stem_skipped(inline_executor, tmp_path):
    """A file_name that reduces to an empty stem is counted skipped_unsafe, not
    written."""
    import json

    in_dir = tmp_path / "in_e"
    out_dir = tmp_path / "out_e"
    in_dir.mkdir()
    out_dir.mkdir()
    coco = {
        "images": [
            {"id": 1, "file_name": "/", "width": 50, "height": 50},
            {"id": 2, "file_name": "ok.png", "width": 50, "height": 50},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 5, 5]},
            {"id": 2, "image_id": 2, "category_id": 0, "bbox": [1, 1, 5, 5]},
        ],
    }
    (in_dir / "a.json").write_text(json.dumps(coco))
    req = dr.ConverterRequest(input_dir=str(in_dir), output_dir=str(out_dir),
                              from_fmt="COCO JSON", to_fmt="YOLO")
    _run_coro(dr.data_converter(req))
    res = converter_state.snapshot()["results"]
    assert res["skipped_unsafe"] == 1, res
    assert res["images"] == 1, res
    assert sorted(p.name for p in out_dir.glob("*.txt")) == ["ok.txt"]


def test_converter_coco_to_yolo_absolute_path_blocked(inline_executor, tmp_path):
    """An absolute file_name must be reduced to basename, never honored as-is."""
    import json

    in_dir = tmp_path / "in2"
    out_dir = tmp_path / "out2"
    in_dir.mkdir()
    out_dir.mkdir()
    # basename of this absolute path is 'pwned' — it should land safely in out_dir,
    # NOT at the absolute location.
    abs_name = str(tmp_path / "elsewhere" / "pwned.png")
    coco = {
        "images": [{"id": 1, "file_name": abs_name, "width": 50, "height": 50}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 5, 10, 10]}],
    }
    (in_dir / "a.json").write_text(json.dumps(coco))

    req = dr.ConverterRequest(input_dir=str(in_dir), output_dir=str(out_dir),
                              from_fmt="COCO JSON", to_fmt="YOLO")
    _run_coro(dr.data_converter(req))
    # File written under out_dir with the basename only.
    assert (out_dir / "pwned.txt").is_file()
    assert not (tmp_path / "elsewhere").exists()


# ── splitter webp/tif inclusion (P0 correctness) ────────────────────────────
def test_glob_canonical_includes_webp_and_tiff(tmp_path):
    """The canonical globber must enumerate .webp/.tif, which the shared
    glob_images drops."""
    for name in ("a.jpg", "b.png", "c.webp", "d.tif", "e.tiff", "f.bmp"):
        (tmp_path / name).write_bytes(b"x")
    found = {os.path.basename(p) for p in dr._glob_images_canonical(str(tmp_path))}
    assert {"c.webp", "d.tif", "e.tiff"}.issubset(found), found
    assert found == {"a.jpg", "b.png", "c.webp", "d.tif", "e.tiff", "f.bmp"}


def test_splitter_includes_webp_and_tif(inline_executor, tmp_path):
    """A split with train=1.0 over a dir of .png + .webp must count BOTH images."""
    img_dir = tmp_path / "imgs"
    out_dir = tmp_path / "split_out"
    img_dir.mkdir()
    (img_dir / "one.png").write_bytes(b"x")
    (img_dir / "two.webp").write_bytes(b"x")

    req = dr.SplitterRequest(img_dir=str(img_dir), output_dir=str(out_dir),
                             train=1.0, val=0.0, test=0.0, strategy="random")
    resp = _run_coro(dr.data_splitter(req))
    assert resp == {"ok": True}
    snap = splitter_state.snapshot()
    assert snap["msg"] == "Complete", snap
    assert snap["results"]["train"] == 2, snap["results"]


def test_splitter_seed_is_reproducible(inline_executor, tmp_path):
    """Same seed -> identical split assignment across two runs."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(20):
        (img_dir / f"i{i:02d}.png").write_bytes(b"x")

    def run(out_name: str) -> dict:
        out_dir = tmp_path / out_name
        req = dr.SplitterRequest(img_dir=str(img_dir), output_dir=str(out_dir),
                                 train=0.6, val=0.2, test=0.2, strategy="random", seed=123)
        _run_coro(dr.data_splitter(req))
        # collect the basenames per split
        return {
            split: sorted(p.name for p in (out_dir / split / "images").glob("*"))
            for split in ("train", "val", "test")
        }

    a = run("o1")
    b = run("o2")
    assert a == b, (a, b)


def test_splitter_overwrite_guard(inline_executor, tmp_path):
    """Re-running into a non-empty output dir without overwrite errors out;
    with overwrite=True it clears stale files so splits stay disjoint."""
    img_dir = tmp_path / "imgs"
    out_dir = tmp_path / "out"
    img_dir.mkdir()
    for i in range(6):
        (img_dir / f"i{i}.png").write_bytes(b"x")

    req = dr.SplitterRequest(img_dir=str(img_dir), output_dir=str(out_dir),
                             train=1.0, val=0.0, test=0.0, strategy="random", seed=1)
    _run_coro(dr.data_splitter(req))
    assert splitter_state.snapshot()["msg"] == "Complete"

    # Second run without overwrite must refuse.
    _run_coro(dr.data_splitter(req))
    snap = splitter_state.snapshot()
    assert snap["results"].get("error") == "OUTPUT_NOT_EMPTY", snap

    # With overwrite it should clear and re-split cleanly.
    req_ow = dr.SplitterRequest(img_dir=str(img_dir), output_dir=str(out_dir),
                                train=1.0, val=0.0, test=0.0, strategy="random",
                                seed=1, overwrite=True)
    _run_coro(dr.data_splitter(req_ow))
    snap = splitter_state.snapshot()
    assert snap["msg"] == "Complete", snap
    assert snap["results"]["train"] == 6, snap["results"]


# ── sampler recursive basename collision (P0 correctness) ───────────────────
def test_sampler_recursive_basename_collision_preserved(inline_executor, tmp_path):
    """a/1.jpg and b/1.jpg (different labels) must both survive sampling with
    their own paired labels — no clobber."""
    img_dir = tmp_path / "imgs"
    out_dir = tmp_path / "out"
    sub_a = img_dir / "a"
    sub_b = img_dir / "b"
    sub_a.mkdir(parents=True)
    sub_b.mkdir(parents=True)
    # Same basename, different content + different label class.
    (sub_a / "1.png").write_bytes(b"AAA")
    (sub_b / "1.png").write_bytes(b"BBB")
    (sub_a / "1.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (sub_b / "1.txt").write_text("1 0.5 0.5 0.2 0.2\n")

    req = dr.SamplerRequest(img_dir=str(img_dir), output_dir=str(out_dir),
                            strategy="Random", target_count=2, seed=7,
                            include_labels=True, recursive=True)
    resp = _run_coro(dr.data_sampler(req))
    assert resp == {"ok": True}
    snap = sampler_state.snapshot()
    assert snap["msg"] == "Complete", snap

    out_imgs = sorted((out_dir / "images").glob("*.png"))
    out_lbls = sorted((out_dir / "labels").glob("*.txt"))
    assert len(out_imgs) == 2, [p.name for p in out_imgs]
    assert len(out_lbls) == 2, [p.name for p in out_lbls]

    # Both source contents present (no clobber).
    contents = {p.read_bytes() for p in out_imgs}
    assert contents == {b"AAA", b"BBB"}, contents

    # Each image's paired label keeps the source class; one label has class 0,
    # the other class 1.
    label_classes = sorted(p.read_text().split()[0] for p in out_lbls)
    assert label_classes == ["0", "1"], label_classes


def test_sampler_label_resolver_recursive_sibling(tmp_path):
    """Recursive label resolution prefers the sibling .txt next to the image."""
    sub = tmp_path / "x"
    sub.mkdir()
    (sub / "img.png").write_bytes(b"x")
    (sub / "img.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    resolved = dr._sampler_label_for(str(sub / "img.png"), str(tmp_path), recursive=True)
    assert resolved == str(sub / "img.txt")


# ── sampler proportional trim (P1 correctness) ──────────────────────────────
def test_proportional_trim_preserves_class_ratio():
    """Trimming a stratified selection should keep the dominant class dominant,
    not collapse toward a uniform random sample."""
    import random as _rnd

    # 100 images of class 0, 10 of class 1 — all selected, trim to 22.
    class0 = [f"c0_{i}" for i in range(100)]
    class1 = [f"c1_{i}" for i in range(10)]
    class_images = {0: set(class0), 1: set(class1)}
    selected = class0 + class1
    trimmed = dr._proportional_trim(selected, 22, class_images, _rnd.Random(0))
    assert len(trimmed) == 22
    n0 = sum(1 for f in trimmed if f.startswith("c0"))
    n1 = sum(1 for f in trimmed if f.startswith("c1"))
    # Class 0 (10x larger) must remain the clear majority — a flat random trim
    # would land near 20:2 by chance but the proportional trim guarantees the
    # dominant class is not decimated below its share.
    assert n0 > n1
    assert n0 >= 18, (n0, n1)


# ── remapper recursive subtree preservation (P0 correctness) ────────────────
def test_remapper_recursive_preserves_subtree(inline_executor, tmp_path):
    """sub1/x.txt and sub2/x.txt must NOT both flatten to out/x.txt; the subtree
    is mirrored so neither overwrites the other."""
    lbl_dir = tmp_path / "labels"
    out_dir = tmp_path / "out"
    (lbl_dir / "sub1").mkdir(parents=True)
    (lbl_dir / "sub2").mkdir(parents=True)
    (lbl_dir / "sub1" / "x.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (lbl_dir / "sub2" / "x.txt").write_text("1 0.5 0.5 0.2 0.2\n")

    req = dr.RemapperRequest(label_dir=str(lbl_dir), output_dir=str(out_dir),
                             mapping={}, auto_reindex=False, recursive=True)
    resp = _run_coro(dr.data_remapper(req))
    assert resp == {"ok": True}
    snap = remapper_state.snapshot()
    assert snap["msg"] == "Complete", snap

    # Both outputs exist under their own subdir with their own content.
    out1 = out_dir / "sub1" / "x.txt"
    out2 = out_dir / "sub2" / "x.txt"
    assert out1.is_file() and out2.is_file()
    assert out1.read_text().split()[0] == "0"
    assert out2.read_text().split()[0] == "1"
    # No flattened collision.
    assert snap["results"]["collisions"] == 0, snap["results"]
