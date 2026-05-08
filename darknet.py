"""ARGOS DarkCenterNet Video Inference.

Model: ARGOS_JIHACHUL_v4.8.onnx
Input: (1, 3, 736, 1280) float32 [0,1]
Output: (1, 100, 13) — 100 detections, 13 attributes per detection

Usage:
    python darknet.py --video Videos/stroller_01.mp4
    python darknet.py --video Videos/stroller_01.mp4 --output result.mp4 --conf 0.3
    python darknet.py --video Videos/wheelchair_01.mp4 --no-show
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ── Model Config ────────────────────────────────────────
MODEL_PATH = "Models/ARGOS_JIHACHUL_v4.8.onnx"
INPUT_SIZE = (736, 1280)  # (H, W)

CLASS_NAMES = {0: "person", 1: "face", 2: "red-sign", 3: "wheelchair", 4: "cane"}

# Multi-label attributes (col 6~12) — 각각 독립적인 sigmoid 확률
# 상호 배타적이지 않음. 각 속성별 threshold로 판단.
ATTR_NAMES = ["fall", "crawl", "jump", "front", "back", "side", "no-mask"]
ATTR_THRESHOLD = 0.3  # 속성 활성화 임계값

COLORS = [(67, 160, 71), (255, 167, 38), (229, 57, 53), (30, 136, 229), (156, 39, 176)]


# ── Preprocessing ───────────────────────────────────────
def preprocess(frame: np.ndarray) -> np.ndarray:
    """DarkCenterNet 전처리.

    1. 단순 resize (letterbox 없음)
    2. BGR → RGB
    3. HWC → CHW
    4. float32 / 255.0

    Returns: (1, 3, 736, 1280) float32 [0, 1]
    """
    h, w = INPUT_SIZE
    img = cv2.resize(frame, (w, h))
    img = img[..., ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(img[np.newaxis], dtype=np.float32) / 255.0


# ── Postprocessing ──────────────────────────────────────
def postprocess(output: np.ndarray, conf: float, orig_shape: tuple):
    """DarkCenterNet 후처리.

    Output tensor layout (1, 100, 13):
        col[0]  : top-left X (정규화 0~1)
        col[1]  : top-left Y (정규화 0~1)
        col[2]  : width (정규화 0~1)
        col[3]  : height (정규화 0~1)
        col[4]  : confidence (0~1)
        col[5]  : class_id (0=person, 1=face, 2=red-sign, 3=wheelchair, 4=cane)
        col[6]  : fall     (multi-label, sigmoid)
        col[7]  : crawl    (multi-label, sigmoid)
        col[8]  : jump     (multi-label, sigmoid)
        col[9]  : front    (multi-label, sigmoid) — 정면 방향
        col[10] : back     (multi-label, sigmoid) — 후면 방향
        col[11] : side     (multi-label, sigmoid) — 측면 방향
        col[12] : no-mask  (multi-label, sigmoid) — 마스크 미착용

    Multi-label: 각 속성은 독립적. argmax가 아닌 threshold 기반 판단.
    NMS 불필요: CenterNet은 heatmap peak 기반 NMS-free 구조.

    Returns: (boxes, scores, class_ids, attrs)
        boxes: list of (x1, y1, x2, y2) in pixel coords
        scores: np.ndarray (N,)
        class_ids: np.ndarray (N,)
        attrs: np.ndarray (N, 7) or None — multi-label 확률
    """
    bboxes = output[0]  # (100, 13)
    mask = bboxes[:, 4] > conf
    if not mask.any():
        return [], np.array([]), np.array([]), None

    bboxes = bboxes[mask]
    scores = bboxes[:, 4].astype(np.float32)
    class_ids = bboxes[:, 5].astype(np.int32)

    oh, ow = orig_shape[:2]
    x1 = np.clip(bboxes[:, 0] * ow, 0, ow).astype(int)
    y1 = np.clip(bboxes[:, 1] * oh, 0, oh).astype(int)
    x2 = np.clip((bboxes[:, 0] + bboxes[:, 2]) * ow, 0, ow).astype(int)
    y2 = np.clip((bboxes[:, 1] + bboxes[:, 3]) * oh, 0, oh).astype(int)
    boxes = list(zip(x1, y1, x2, y2))

    attrs = bboxes[:, 6:13].astype(np.float32)
    return boxes, scores, class_ids, attrs


# ── Visualization ───────────────────────────────────────
def draw(frame, boxes, scores, class_ids, attrs=None):
    """검출 결과 시각화. Multi-label 속성은 threshold 초과 시 모두 표시."""
    for i, (box, sc, cid) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = box
        c = COLORS[cid % len(COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

        lbl = f"{CLASS_NAMES.get(int(cid), str(int(cid)))} {sc:.2f}"

        # Multi-label: threshold 초과하는 모든 속성 표시
        if attrs is not None and i < len(attrs):
            active = []
            for j, val in enumerate(attrs[i]):
                if val > ATTR_THRESHOLD:
                    active.append(f"{ATTR_NAMES[j]}")
            if active:
                lbl += " [" + ",".join(active) + "]"

        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), c, -1)
        cv2.putText(frame, lbl, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


# ── Main ────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="ARGOS DarkCenterNet Video Inference")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--video", required=True)
    ap.add_argument("--output", default=None, help="Output video path")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--attr-thresh", type=float, default=0.3, help="Multi-label attribute threshold")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    global ATTR_THRESHOLD
    ATTR_THRESHOLD = args.attr_thresh

    # Load model
    sess = ort.InferenceSession(args.model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print(f"Model: {Path(args.model).name}")
    print(f"Input: {inp.name} shape={inp.shape}")
    print(f"Conf threshold: {args.conf} | Attr threshold: {ATTR_THRESHOLD}")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw, vh = int(cap.get(3)), int(cap.get(4))
    print(f"Video: {Path(args.video).name} ({vw}x{vh} @ {fps:.1f}fps, {total} frames)")

    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))

    idx, t_total = 0, 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tensor = preprocess(frame)
        t0 = time.perf_counter()
        out = sess.run(None, {inp.name: tensor})
        ms = (time.perf_counter() - t0) * 1000
        t_total += ms

        boxes, scores, cids, attrs = postprocess(out[0], args.conf, frame.shape)
        vis = draw(frame.copy(), boxes, scores, cids, attrs)
        cv2.putText(vis, f"{idx}/{total} | {ms:.1f}ms | {len(boxes)} det",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if writer:
            writer.write(vis)
        if not args.no_show:
            cv2.imshow("ARGOS", vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    avg = t_total / max(idx, 1)
    print(f"\nDone: {idx} frames | avg {avg:.1f}ms/frame ({1000/avg:.1f} FPS)")


if __name__ == "__main__":
    main()
