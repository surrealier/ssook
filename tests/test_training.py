"""Tests for the Training module (core.training + routes).

Registry / dataset-prep / route tests run anywhere. Actual model training is
NOT exercised here (needs a framework + GPU + time); availability is reported
via the registry and gated by the trainer's is_available().
"""
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="ssook_train_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_flat_yolo_dataset(root: str, n=6):
    """Create n dummy images + sibling YOLO .txt labels (no image decoding needed
    by prepare_yolo_dataset — it only globs + links files)."""
    os.makedirs(root, exist_ok=True)
    for i in range(n):
        img = os.path.join(root, f"img_{i}.png")
        with open(img, "wb") as f:
            f.write(b"\x89PNG\r\n")  # not a real PNG; prep never decodes it
        cid = i % 3
        with open(os.path.join(root, f"img_{i}.txt"), "w") as f:
            f.write(f"{cid} 0.5 0.5 0.2 0.2\n")
    return root


def _make_flat_imagefolder(root: str, class_counts: dict) -> str:
    """Create a flat ImageFolder: <root>/<class>/img_*.png dummy byte files.

    prepare_classification_dataset only globs + links files (never decodes), so
    these can be tiny non-image bytes.
    """
    os.makedirs(root, exist_ok=True)
    for cls, n in class_counts.items():
        cls_dir = os.path.join(root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n):
            with open(os.path.join(cls_dir, f"img_{i}.png"), "wb") as f:
                f.write(b"\x89PNG\r\n")  # not a real PNG; prep never decodes it
    return root


# ── Registry ───────────────────────────────────────────
from core.training.registry import BaseTrainer, TrainerRegistry, registry


class TestTrainerRegistry:
    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseTrainer()

    def test_builtins_present(self):
        names = {t.name for t in registry.list_all()}
        for expected in ("yolo_detect", "yolo_segment", "yolo_pose", "yolo_classify", "timm_classify"):
            assert expected in names

    def test_to_dict_shape(self):
        for t in registry.list_all():
            d = t.to_dict()
            for key in ("name", "label", "framework", "task", "dataset_kind", "available"):
                assert key in d

    def test_list_by_task(self):
        assert len(registry.list_by_task("classify")) >= 1


# ── Dataset preparation ────────────────────────────────
from core.training.dataset import (
    prepare_yolo_dataset, discover_classes, resolve_names, find_existing_data_yaml,
    prepare_classification_dataset,
)


class TestDatasetPrep:
    def test_discover_classes(self, tmp_dir):
        root = _make_flat_yolo_dataset(tmp_dir, n=6)
        labels = [os.path.join(root, f"img_{i}.txt") for i in range(6)]
        assert discover_classes(labels) == 3  # ids 0,1,2 -> nc=3

    def test_resolve_names_variants(self):
        assert resolve_names(2, ["a", "b"]) == {0: "a", 1: "b"}
        assert resolve_names(2, "a, b") == {0: "a", 1: "b"}
        assert resolve_names(2, {0: "x", 1: "y"}) == {0: "x", 1: "y"}
        fallback = resolve_names(3, None)
        assert fallback[0].startswith("class_") or isinstance(fallback[0], str)

    def test_prepare_generates_data_yaml(self, tmp_dir):
        src = _make_flat_yolo_dataset(os.path.join(tmp_dir, "src"), n=6)
        out = os.path.join(tmp_dir, "prepared")
        info = prepare_yolo_dataset(src, out_root=out, val_ratio=0.34, seed=1)
        assert info["prepared"] is True
        assert os.path.isfile(info["data_yaml"])
        assert info["nc"] == 3
        assert info["n_train"] + info["n_val"] == 6
        assert info["n_val"] >= 1
        # data.yaml is parseable and has the expected keys.
        import yaml
        with open(info["data_yaml"]) as f:
            data = yaml.safe_load(f)
        assert data["nc"] == 3 and "names" in data and "train" in data and "val" in data
        # train/val image dirs are populated.
        assert os.path.isdir(os.path.join(out, "train", "images"))
        assert len(os.listdir(os.path.join(out, "train", "images"))) >= 1

    def test_existing_data_yaml_passthrough(self, tmp_dir):
        src = os.path.join(tmp_dir, "ready")
        os.makedirs(src)
        yml = os.path.join(src, "data.yaml")
        open(yml, "w").close()
        assert find_existing_data_yaml(src) == yml
        info = prepare_yolo_dataset(src)
        assert info["prepared"] is False
        assert info["data_yaml"] == yml

    def test_names_shorter_than_nc_is_reconciled(self, tmp_dir):
        # Labels reference class ids 0,1,2 (nc=3) but caller gives only 2 names —
        # data.yaml must stay consistent (len(names) == nc), gaps filled.
        src = _make_flat_yolo_dataset(os.path.join(tmp_dir, "src"), n=6)
        out = os.path.join(tmp_dir, "prep")
        info = prepare_yolo_dataset(src, names=["cat", "dog"], out_root=out, seed=1)
        assert info["nc"] == 3
        assert len(info["names"]) == info["nc"]
        assert info["names"][0] == "cat" and info["names"][1] == "dog"
        assert info["names"][2].startswith("class_")

    def test_single_image_val_falls_back_to_train(self, tmp_dir):
        src = _make_flat_yolo_dataset(os.path.join(tmp_dir, "one"), n=1)
        out = os.path.join(tmp_dir, "prep1")
        info = prepare_yolo_dataset(src, out_root=out)
        import yaml
        with open(info["data_yaml"]) as f:
            data = yaml.safe_load(f)
        # No val split for a single image -> val must not point at an empty dir.
        assert data["val"] == "train/images"

    def test_no_labels_raises(self, tmp_dir):
        src = os.path.join(tmp_dir, "imgsonly")
        os.makedirs(src)
        with open(os.path.join(src, "a.png"), "wb") as f:
            f.write(b"x")
        with pytest.raises(ValueError):
            prepare_yolo_dataset(src, out_root=os.path.join(tmp_dir, "o"))


# ── Classification dataset preparation ─────────────────
class TestClassificationDatasetPrep:
    def test_flat_split_same_classes_and_counts(self, tmp_dir):
        src = _make_flat_imagefolder(os.path.join(tmp_dir, "src"),
                                     {"cat": 5, "dog": 5, "bird": 4})
        out = os.path.join(tmp_dir, "out")
        info = prepare_classification_dataset(src, val_ratio=0.4, seed=1, out_root=out)
        assert info["prepared"] is True
        assert info["data_dir"] == out
        assert info["classes"] == ["bird", "cat", "dog"]  # sorted
        assert info["n_train"] + info["n_val"] == 14
        assert info["n_val"] >= 1
        # train/ and val/ exist with the same class set.
        train_classes = sorted(os.listdir(os.path.join(out, "train")))
        val_classes = sorted(os.listdir(os.path.join(out, "val")))
        assert train_classes == ["bird", "cat", "dog"]
        assert val_classes == ["bird", "cat", "dog"]

    def test_deterministic_same_seed(self, tmp_dir):
        src = _make_flat_imagefolder(os.path.join(tmp_dir, "src"), {"a": 8, "b": 6})
        out1 = os.path.join(tmp_dir, "o1")
        out2 = os.path.join(tmp_dir, "o2")
        info1 = prepare_classification_dataset(src, val_ratio=0.25, seed=7, out_root=out1)
        info2 = prepare_classification_dataset(src, val_ratio=0.25, seed=7, out_root=out2)
        assert (info1["n_train"], info1["n_val"]) == (info2["n_train"], info2["n_val"])
        # Exact same files land in val for both runs.
        for cls in info1["classes"]:
            v1 = sorted(os.listdir(os.path.join(out1, "val", cls)))
            v2 = sorted(os.listdir(os.path.join(out2, "val", cls)))
            assert v1 == v2

    def test_presplit_passthrough(self, tmp_dir):
        # A dir already containing train/ (and val/) is returned untouched.
        root = os.path.join(tmp_dir, "ready")
        _make_flat_imagefolder(os.path.join(root, "train"), {"a": 3, "b": 2})
        _make_flat_imagefolder(os.path.join(root, "val"), {"a": 1, "b": 1})
        info = prepare_classification_dataset(root)
        assert info["prepared"] is False
        assert info["data_dir"] == root
        assert info["classes"] == ["a", "b"]
        assert info["n_train"] == 5
        assert info["n_val"] == 2

    def test_single_image_class_all_to_train(self, tmp_dir):
        # A class with one image can't spare a val sample -> all to train; val
        # simply omits that class. Train is never empty.
        src = _make_flat_imagefolder(os.path.join(tmp_dir, "src"), {"solo": 1, "many": 6})
        out = os.path.join(tmp_dir, "out")
        info = prepare_classification_dataset(src, val_ratio=0.5, seed=0, out_root=out)
        assert info["prepared"] is True
        assert os.path.isdir(os.path.join(out, "train", "solo"))
        assert len(os.listdir(os.path.join(out, "train", "solo"))) == 1
        # 'solo' must NOT appear under val (only one image, kept in train).
        val_dir = os.path.join(out, "val")
        if os.path.isdir(val_dir):
            assert "solo" not in os.listdir(val_dir)

    def test_no_class_subdirs_raises(self, tmp_dir):
        empty = os.path.join(tmp_dir, "empty")
        os.makedirs(empty)
        with pytest.raises(ValueError):
            prepare_classification_dataset(empty, out_root=os.path.join(tmp_dir, "o"))

    def test_class_dirs_without_images_raises(self, tmp_dir):
        # Class subdirs exist but hold no images -> clear error, no output dir.
        src = os.path.join(tmp_dir, "src")
        os.makedirs(os.path.join(src, "a"))
        os.makedirs(os.path.join(src, "b"))
        with pytest.raises(ValueError):
            prepare_classification_dataset(src, out_root=os.path.join(tmp_dir, "o"))


# ── Runs management ────────────────────────────────────
from core.training.runs import list_runs, delete_run, default_runs_dir


def _make_yolo_run(runs_root: str, name: str = "yolo_detect") -> str:
    """Create a fake YOLO run dir: weights/best.pt + args.yaml + results.csv."""
    run_dir = os.path.join(runs_root, name)
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
    with open(os.path.join(run_dir, "weights", "best.pt"), "wb") as f:
        f.write(b"\x00" * 16)
    with open(os.path.join(run_dir, "args.yaml"), "w", encoding="utf-8") as f:
        f.write("task: detect\nmodel: yolo11n.pt\nepochs: 100\n")
    with open(os.path.join(run_dir, "results.csv"), "w", encoding="utf-8") as f:
        # Headers padded with whitespace like Ultralytics does.
        f.write("epoch,   train/box_loss,   metrics/mAP50(B)\n")
        f.write("1, 0.9, 0.10\n")
        f.write("2, 0.5, 0.42\n")
    return run_dir


class TestRunsManagement:
    def test_list_runs_parses_yolo_run(self, tmp_dir):
        runs_root = os.path.join(tmp_dir, "runs", "train")
        _make_yolo_run(runs_root, "run_a")
        runs = list_runs(runs_dir=runs_root)
        assert len(runs) == 1
        run = runs[0]
        assert run["name"] == "run_a"
        assert run["task"] == "detect"
        assert run["framework"] == "ultralytics"
        assert run["has_weights"] is True
        assert run["best_model"].endswith(os.path.join("weights", "best.pt"))
        # Last (numeric) row of results.csv, headers stripped.
        assert run["last_metrics"]["metrics/mAP50(B)"] == 0.42
        assert "epoch" in run["last_metrics"]  # numeric column kept
        assert isinstance(run["mtime"], float)
        assert isinstance(run["size_mb"], float)

    def test_list_runs_timm_best_at_root(self, tmp_dir):
        runs_root = os.path.join(tmp_dir, "runs", "train")
        run_dir = os.path.join(runs_root, "timm_run")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "best.pt"), "wb") as f:
            f.write(b"\x00" * 8)
        runs = list_runs(runs_dir=runs_root)
        run = next(r for r in runs if r["name"] == "timm_run")
        assert run["has_weights"] is True
        assert run["framework"] == "torchvision"
        # No torch load here -> task/metrics stay best-effort empty.
        assert run["task"] is None
        assert run["last_metrics"] == {}

    def test_list_runs_sorted_newest_first(self, tmp_dir):
        runs_root = os.path.join(tmp_dir, "runs", "train")
        _make_yolo_run(runs_root, "older")
        newer = _make_yolo_run(runs_root, "newer")
        # Force a strictly later mtime on the second run.
        future = os.path.getmtime(newer) + 1000
        os.utime(newer, (future, future))
        runs = list_runs(runs_dir=runs_root)
        assert runs[0]["name"] == "newer"

    def test_list_runs_missing_root_returns_empty(self, tmp_dir):
        assert list_runs(runs_dir=os.path.join(tmp_dir, "does_not_exist")) == []

    def test_delete_run_removes_dir(self, tmp_dir):
        runs_root = os.path.join(tmp_dir, "runs", "train")
        run_dir = _make_yolo_run(runs_root, "deleteme")
        assert os.path.isdir(run_dir)
        assert delete_run("deleteme", runs_dir=runs_root) is True
        assert not os.path.exists(run_dir)

    def test_delete_run_traversal_raises(self, tmp_dir):
        runs_root = os.path.join(tmp_dir, "runs", "train")
        os.makedirs(runs_root, exist_ok=True)
        with pytest.raises(ValueError):
            delete_run("../escape", runs_dir=runs_root)

    def test_delete_run_absolute_raises(self, tmp_dir):
        runs_root = os.path.join(tmp_dir, "runs", "train")
        os.makedirs(runs_root, exist_ok=True)
        # An absolute path joined onto the root resolves outside it.
        absolute = os.path.abspath(os.path.join(tmp_dir, "outside"))
        os.makedirs(absolute, exist_ok=True)
        with pytest.raises(ValueError):
            delete_run(absolute, runs_dir=runs_root)

    def test_delete_run_missing_raises(self, tmp_dir):
        runs_root = os.path.join(tmp_dir, "runs", "train")
        os.makedirs(runs_root, exist_ok=True)
        with pytest.raises(ValueError):
            delete_run("nope", runs_dir=runs_root)

    def test_default_runs_dir_location(self):
        d = default_runs_dir()
        assert d.name == "train"
        assert d.parent.name == "runs"


# ── Runner ─────────────────────────────────────────────
from core.training.runner import run_training


class TestRunTraining:
    def test_unknown_trainer(self):
        result = run_training({"trainer": "nope", "images_dir": "/tmp"})
        assert "error" in result

    def test_unavailable_trainer_reports_reason(self, monkeypatch):
        trainer = registry.get("yolo_detect")
        monkeypatch.setattr(trainer, "is_available", lambda: (False, "torch missing"))
        result = run_training({"trainer": "yolo_detect", "images_dir": "/tmp"})
        assert "error" in result and "torch missing" in result["error"]


# ── Route smoke ────────────────────────────────────────
class TestTrainingRoutes:
    def _client(self):
        from fastapi.testclient import TestClient
        from server import app
        return TestClient(app)

    def test_trainers_endpoint(self):
        r = self._client().get("/api/training/trainers")
        assert r.status_code == 200
        names = {t["name"] for t in r.json()["trainers"]}
        assert "yolo_detect" in names
        assert "cuda_available" in r.json()

    def test_run_requires_images_dir(self):
        r = self._client().post("/api/training/run", json={"trainer": "yolo_detect"})
        assert "error" in r.json()

    def test_runs_endpoint(self):
        r = self._client().get("/api/training/runs")
        assert r.status_code == 200
        body = r.json()
        assert "runs" in body
        assert isinstance(body["runs"], list)
