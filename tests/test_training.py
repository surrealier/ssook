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
