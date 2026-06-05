"""Built-in trainers. Importing this module registers them into ``registry``.

* ``UltralyticsTrainer`` — one class, registered four times (detect / segment /
  pose / classify). Streams per-epoch progress + metrics via Ultralytics
  callbacks and stops cooperatively by setting ``trainer.stop``.
* ``TimmClassifierTrainer`` — a compact torchvision/timm image-classification
  loop over an ImageFolder dataset, for classification backbones beyond YOLO.

All heavy imports are lazy (inside ``train``/``is_available``).
"""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Optional

from core.training.registry import BaseTrainer, registry
from core.training.dataset import prepare_yolo_dataset, prepare_classification_dataset

log = logging.getLogger("ssook.training")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _default_runs_dir() -> str:
    # Outside settings/tmp so trained checkpoints are NOT auto-cleaned.
    return str(_REPO_ROOT / "runs" / "train")


# ── Ultralytics YOLO (detect / segment / pose / classify) ─────────────────
class UltralyticsTrainer(BaseTrainer):
    framework = "ultralytics"

    def __init__(self, task: str, label: str, default_model: str, dataset_kind: str = "yolo"):
        self.task = task
        self.name = f"yolo_{task}"
        self.label = label
        self.default_model = default_model
        self.dataset_kind = dataset_kind
        self.description = f"Ultralytics YOLO {task} training (configurable epochs/imgsz/batch/lr)."

    def is_available(self) -> tuple[bool, str]:
        if not _have("torch"):
            return False, "needs torch (pip install torch)"
        if not _have("ultralytics"):
            return False, "needs ultralytics (pip install ultralytics)"
        return True, ""

    def train(self, config, on_progress=None, on_metric=None, is_stopped=None) -> dict:
        from ultralytics import YOLO  # lazy: heavy, optional

        model_name = config.get("model") or self.default_model
        epochs = int(config.get("epochs", 100))
        imgsz = int(config.get("imgsz", 640))
        # batch -1 lets ultralytics auto-pick by VRAM.
        batch = config.get("batch", 16)
        try:
            batch = int(batch)
        except (TypeError, ValueError):
            batch = 16
        device = config.get("device", "")
        project = config.get("project") or _default_runs_dir()
        run_name = config.get("name") or self.name

        # ── Dataset resolution ──
        dataset_info: dict = {}
        if self.dataset_kind == "imagefolder":
            images_dir = config.get("images_dir") or config.get("data", "")
            if not images_dir:
                raise ValueError("classification training needs images_dir (an ImageFolder of class subfolders).")
            # Auto-split a flat ImageFolder into train/val (no-op if already split).
            dataset_info = prepare_classification_dataset(
                images_dir,
                val_ratio=float(config.get("val_ratio", 0.2)),
                seed=int(config.get("seed", 0)),
            )
            data = dataset_info["data_dir"]
        else:
            dataset_info = prepare_yolo_dataset(
                images_dir=config["images_dir"],
                labels_dir=config.get("labels_dir") or None,
                names=config.get("names"),
                val_ratio=float(config.get("val_ratio", 0.2)),
                seed=int(config.get("seed", 0)),
                keep=config.get("keep"),
            )
            data = dataset_info["data_yaml"]

        model = YOLO(model_name)

        # Per-epoch progress/metrics + cooperative stop. on_fit_epoch_end fires
        # after validation each epoch, so trainer.metrics is populated.
        def _on_epoch(trainer):
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            total = int(getattr(trainer, "epochs", epochs)) or epochs
            metrics = {
                k: float(v) for k, v in (getattr(trainer, "metrics", {}) or {}).items()
                if isinstance(v, (int, float))
            }
            if on_progress:
                on_progress(epoch, total, f"Epoch {epoch}/{total}")
            if on_metric:
                on_metric(epoch, metrics)
            if is_stopped and is_stopped():
                # Ultralytics checks trainer.stop at the epoch boundary and breaks.
                trainer.stop = True

        model.add_callback("on_fit_epoch_end", _on_epoch)

        kwargs = dict(data=data, epochs=epochs, imgsz=imgsz, batch=batch,
                      project=project, name=run_name, exist_ok=True, verbose=False)
        if device != "":
            kwargs["device"] = device
        if config.get("lr0") not in (None, ""):
            kwargs["lr0"] = float(config["lr0"])
        if config.get("patience") not in (None, ""):
            kwargs["patience"] = int(config["patience"])

        log.info("ultralytics train: model=%s task=%s epochs=%d imgsz=%d batch=%s",
                 model_name, self.task, epochs, imgsz, batch)
        results = model.train(**kwargs)

        trainer = getattr(model, "trainer", None)
        best = str(getattr(trainer, "best", "")) if trainer else ""
        last = str(getattr(trainer, "last", "")) if trainer else ""
        save_dir = str(getattr(trainer, "save_dir", "")) if trainer else ""
        epochs_done = int(getattr(trainer, "epoch", epochs - 1)) + 1 if trainer else epochs

        final_metrics: dict = {}
        try:
            rd = getattr(results, "results_dict", None) or {}
            final_metrics = {k: float(v) for k, v in rd.items() if isinstance(v, (int, float))}
        except Exception:
            pass

        return {
            "trainer": self.name,
            "framework": self.framework,
            "task": self.task,
            "base_model": model_name,
            "best_model": best,
            "last_model": last,
            "save_dir": save_dir,
            "epochs_completed": epochs_done,
            "metrics": final_metrics,
            "dataset": dataset_info,
        }


# ── timm / torchvision image classification ───────────────────────────────
class TimmClassifierTrainer(BaseTrainer):
    name = "timm_classify"
    label = "Image Classifier (timm / torchvision)"
    framework = "torchvision"
    task = "classify"
    dataset_kind = "imagefolder"
    default_model = "resnet18"
    description = "Train a classification backbone (timm if installed, else torchvision) on an ImageFolder."

    def is_available(self) -> tuple[bool, str]:
        if not _have("torch"):
            return False, "needs torch (pip install torch)"
        if not _have("torchvision"):
            return False, "needs torchvision (pip install torchvision)"
        return True, ""

    def _build_model(self, arch: str, num_classes: int):
        # Prefer timm (huge model zoo); fall back to torchvision.
        if _have("timm"):
            import timm
            return timm.create_model(arch, pretrained=True, num_classes=num_classes)
        import torchvision
        model = torchvision.models.get_model(arch, weights="DEFAULT")
        # Swap the classifier head to num_classes for common architectures.
        import torch.nn as nn
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, "classifier"):
            clf = model.classifier
            if isinstance(clf, nn.Linear):
                model.classifier = nn.Linear(clf.in_features, num_classes)
            elif isinstance(clf, nn.Sequential) and isinstance(clf[-1], nn.Linear):
                clf[-1] = nn.Linear(clf[-1].in_features, num_classes)
        return model

    def train(self, config, on_progress=None, on_metric=None, is_stopped=None) -> dict:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

        images_dir = config.get("images_dir") or config.get("data", "")
        if not images_dir:
            raise ValueError("classification needs images_dir (an ImageFolder of class subfolders).")
        # Auto-split a flat ImageFolder into train/val (no-op if already split).
        dataset_info = prepare_classification_dataset(
            images_dir,
            val_ratio=float(config.get("val_ratio", 0.2)),
            seed=int(config.get("seed", 0)),
        )
        data_dir = dataset_info["data_dir"]

        arch = config.get("model") or self.default_model
        epochs = int(config.get("epochs", 20))
        imgsz = int(config.get("imgsz", 224))
        batch = int(config.get("batch", 32))
        lr = float(config.get("lr0", 1e-3))
        # patience: None / "" disables early stopping (the prior behavior).
        patience_raw = config.get("patience")
        patience = int(patience_raw) if patience_raw not in (None, "") else None
        device_str = config.get("device", "") or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)

        train_tf = transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        eval_tf = transforms.Compose([transforms.Resize((imgsz, imgsz)), transforms.ToTensor()])

        train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
        val_path = os.path.join(data_dir, "val")
        val_ds = datasets.ImageFolder(val_path, transform=eval_tf) if os.path.isdir(val_path) else None
        classes = train_ds.classes
        num_classes = len(classes)

        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0) if val_ds else None

        model = self._build_model(arch, num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        save_dir = config.get("project") or os.path.join(_default_runs_dir(), config.get("name") or "timm_classify")
        os.makedirs(save_dir, exist_ok=True)
        best_acc, best_path = -1.0, os.path.join(save_dir, "best.pt")
        epochs_done = 0
        epochs_since_best = 0  # for cooperative early stopping when patience is set

        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), targets)
                loss.backward()
                optimizer.step()
                running += float(loss.item())
            train_loss = running / max(1, len(train_loader))

            val_acc = None
            if val_loader:
                model.eval()
                correct = total = 0
                with torch.no_grad():
                    for images, targets in val_loader:
                        images, targets = images.to(device), targets.to(device)
                        preds = model(images).argmax(1)
                        correct += int((preds == targets).sum().item())
                        total += int(targets.numel())
                val_acc = round(correct / total, 4) if total else None

            epochs_done = epoch
            metric_row = {"train_loss": round(train_loss, 4)}
            if val_acc is not None:
                metric_row["val_acc"] = val_acc
            if on_progress:
                on_progress(epoch, epochs, f"Epoch {epoch}/{epochs}")
            if on_metric:
                on_metric(epoch, metric_row)

            score = val_acc if val_acc is not None else -train_loss
            if score is not None and score > best_acc:
                best_acc = score
                epochs_since_best = 0
                torch.save({"model_state": model.state_dict(), "arch": arch,
                            "classes": classes, "num_classes": num_classes}, best_path)
            else:
                epochs_since_best += 1

            if is_stopped and is_stopped():
                log.info("timm train: cooperative stop at epoch %d", epoch)
                break

            # Early stopping: bail once `patience` epochs pass with no improvement.
            if patience is not None and epochs_since_best >= patience:
                log.info("timm train: early stop at epoch %d (no improvement in %d epochs)",
                         epoch, patience)
                break

        return {
            "trainer": self.name,
            "framework": self.framework,
            "task": self.task,
            "base_model": arch,
            "best_model": best_path if os.path.exists(best_path) else "",
            "save_dir": save_dir,
            "epochs_completed": epochs_done,
            "metrics": {"best_val_acc": best_acc if best_acc >= 0 else None},
            "classes": classes,
            "dataset": dataset_info,
        }


def _register_builtins() -> None:
    builtins = [
        UltralyticsTrainer("detect", "YOLO Detection", "yolo11n.pt", "yolo"),
        UltralyticsTrainer("segment", "YOLO Segmentation", "yolo11n-seg.pt", "yolo"),
        UltralyticsTrainer("pose", "YOLO Pose", "yolo11n-pose.pt", "yolo"),
        UltralyticsTrainer("classify", "YOLO Classification", "yolo11n-cls.pt", "imagefolder"),
        TimmClassifierTrainer(),
    ]
    for trainer in builtins:
        try:
            registry.register(trainer)
        except ValueError:
            pass


_register_builtins()
