# Training

Train and fine-tune vision models on your datasets, then hand the result
straight to the [Deployment](deployment.md) tab for export. Training is a
long-running background job with live per-epoch progress and metrics.

> The core app is ONNX-runtime-only; training frameworks are **optional**.
> Install them with `pip install -r requirements-train.txt`. A trainer whose
> backend is missing is shown greyed-out with the reason — startup never breaks.

## Trainers

| Trainer | Framework | Task | Dataset |
|---------|-----------|------|---------|
| `yolo_detect` | Ultralytics | Detection | YOLO images + `.txt` labels |
| `yolo_segment` | Ultralytics | Segmentation | YOLO seg labels |
| `yolo_pose` | Ultralytics | Pose | YOLO pose labels |
| `yolo_classify` | Ultralytics | Classification | ImageFolder — flat `class/img.*` (auto-split) or pre-split `train/`,`val/` |
| `timm_classify` | timm / torchvision | Classification | ImageFolder — flat `class/img.*` (auto-split) or pre-split `train/`,`val/` |

The base model can be a name Ultralytics auto-fetches (`yolo11n.pt`), an
architecture-only config (`yolo11n.yaml`, no download), or a path to your own
checkpoint. For `timm_classify` the base model is a backbone name
(`resnet18`, `efficientnet_b0`, `vit_base_patch16_224`, …).

## Dataset preparation (YOLO tasks)

Point the **Dataset images dir** at a directory of images. Labels are found as
sibling `.txt` files, or in a separate **Labels dir**. ssook then:

1. discovers class ids from the labels (`nc = max id + 1`),
2. fills class names from your comma-separated list, else a known catalog
   (COCO/VOC/ImageNet) by count, else `class_{i}`,
3. splits into `train`/`val` by the validation ratio (deterministic by seed),
4. hard-links files into a standard layout and writes a `data.yaml`.

A directory that already contains a `data.yaml` is used as-is. The repo's flat
`labeling/` directory (images + sibling `.txt`) trains directly.

## Dataset preparation (classification tasks)

Classification mirrors detection — point the images dir at a folder and ssook
splits it for you (no manual `train/`,`val/` needed):

1. a **flat** ImageFolder (`class_name/img.*`) is split per-class into
   `train/`,`val/` by the validation ratio (deterministic by seed, hard-linked);
2. a folder that **already** has a `train/` subdir is used as-is (pass-through).

A class with a single image keeps it in `train` (val just omits that class), so
small folders still train.

## Parameters

`epochs`, image size, batch (`-1` = auto by VRAM), device (`0`/`cpu`), learning
rate (blank = framework default) and validation split. **Patience** (Advanced)
enables early stopping — training bails after that many epochs with no metric
improvement (YOLO uses it natively; `timm_classify` honours it too; blank =
disabled). Training also stops cooperatively at the next epoch boundary when you
press **Stop**.

## Trained runs

Finished runs are written under `runs/train/<name>/` and never auto-cleaned. The
**Trained runs** card lists them with their task, last metrics and size, and
one-click **Deploy** / **Benchmark** / **Delete** — so the train → export →
benchmark pipeline survives a page reload.

## API

- `GET /api/training/trainers` — trainers + availability + CUDA hint
- `POST /api/training/run` — start (returns immediately)
- `GET /api/training/status` — `{running, progress, total, msg, results}`
- `GET /api/training/stop` — request a cooperative stop
- `GET /api/training/runs` — list finished runs (newest first) with task,
  last metrics, best-checkpoint path and size
- `DELETE /api/training/runs/{name}` — delete one run dir (name validated to
  stay inside `runs/train`)

`results.epochs` is a per-epoch metric list; `results.summary` carries the best
checkpoint path on completion.

## Subway fork (keep / multilabel / dominant)

The detection trainer passes a `keep` block through to `data.yaml` verbatim. A
vendored Ultralytics fork (the `anydetection` env) consumes `keep` /
`multilabel` / `dominant_classes` for per-folder class gating; stock Ultralytics
ignores the unknown keys. See `task.md` for the mechanism.

---

## 한국어 요약

데이터셋으로 비전 모델을 학습/파인튜닝하고, 결과 체크포인트를 [배포](deployment.md)
탭으로 바로 내보냅니다. 학습은 에폭별 진행·메트릭을 실시간 표시하는 백그라운드
작업입니다. 학습 프레임워크(torch/ultralytics/timm)는 **선택 설치**이며
(`pip install -r requirements-train.txt`), 미설치 시 해당 트레이너는 사유와 함께
비활성으로 표시됩니다. YOLO 작업은 이미지 폴더(+sibling 라벨 또는 라벨 폴더)에서
클래스 자동 발견·train/val 분할·`data.yaml` 생성을 거쳐 학습하며, 저장소의 평면
`labeling/` 데이터셋을 바로 학습할 수 있습니다. 분류(classify)도 평면
`class/img.*` 폴더를 자동으로 train/val 분할하므로 수동 분할이 필요 없습니다(이미
`train/`이 있으면 그대로 사용). **patience**(고급)로 조기 종료가 가능하고,
**학습 결과** 카드에서 완료된 run을 목록·삭제하거나 바로 배포/벤치마크로 넘길 수
있어 새로고침 후에도 파이프라인이 유지됩니다.
