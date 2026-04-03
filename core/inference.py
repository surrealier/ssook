"""YOLO 추론 파이프라인: letterbox → preprocess → run → postprocess"""
import time
from dataclasses import dataclass

import cv2
import numpy as np

from core.model_loader import ModelInfo

_NMS_IOU = 0.45

# 통합 14-class 체계
UNIFIED_NAMES = {
    0: "person", 1: "falldown", 2: "crawl", 3: "jump",
    4: "front", 5: "back", 6: "side",
    7: "nomask", 8: "cosk", 9: "mask",
    10: "gatesign", 11: "wheelchair", 12: "cane", 13: "stroller",
}


@dataclass
class DetectionResult:
    boxes: np.ndarray      # (N, 4) xyxy, 원본 이미지 좌표계
    scores: np.ndarray     # (N,) float32
    class_ids: np.ndarray  # (N,) int32
    infer_ms: float
    extra_attrs: np.ndarray | None = None  # (N, 7) darknet: fall,crawl,jump,front,back,side,mask

    @classmethod
    def empty(cls) -> "DetectionResult":
        return cls(
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros(0, dtype=np.float32),
            class_ids=np.zeros(0, dtype=np.int32),
            infer_ms=0.0,
        )


@dataclass
class ClassificationResult:
    class_id: int              # top-1 클래스 ID
    confidence: float          # top-1 confidence
    top_k: list                # [(class_id, score), ...] top-k 결과
    probabilities: np.ndarray  # (N,) 전체 클래스 확률
    infer_ms: float


def letterbox(img: np.ndarray, new_shape: tuple) -> tuple:
    """비율 유지 리사이즈 + 패딩. (padded_img, ratio, (pad_w, pad_h)) 반환"""
    h, w = img.shape[:2]
    nh, nw = new_shape
    ratio = min(nw / w, nh / h)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    pad_w = (nw - new_w) / 2
    pad_h = (nh - new_h) / 2
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img_padded, ratio, (pad_w, pad_h)


def preprocess(frame: np.ndarray, input_size: tuple) -> np.ndarray:
    """BGR frame → NCHW float32 [0,1]"""
    padded, _, _ = letterbox(frame, input_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis], dtype=np.float32) / 255.0
    return tensor


def preprocess_darknet(frame: np.ndarray, input_size: tuple) -> np.ndarray:
    """DarkCenterNet 전처리: 단순 resize (letterbox 없음), BGR→RGB, NCHW, float32/255"""
    h, w = input_size
    img = cv2.resize(frame, (w, h))
    img = img[..., ::-1]            # BGR→RGB
    img = img.transpose(2, 0, 1)   # HWC→CHW
    return np.ascontiguousarray(img[np.newaxis], dtype=np.float32) / 255.0


def _nms(boxes_xyxy, scores, class_ids, iou_thres):
    """클래스별 NMS → 최종 인덱스 반환"""
    keep = []
    for cid in np.unique(class_ids):
        mask = class_ids == cid
        idxs = np.where(mask)[0]
        b = boxes_xyxy[mask]
        # cv2.dnn.NMSBoxes는 [x, y, w, h] 포맷 필요 → xyxy를 xywh로 변환
        b_xywh = np.empty_like(b)
        b_xywh[:, 0] = b[:, 0]
        b_xywh[:, 1] = b[:, 1]
        b_xywh[:, 2] = b[:, 2] - b[:, 0]
        b_xywh[:, 3] = b[:, 3] - b[:, 1]
        s = scores[mask].tolist()
        if len(s) == 0:
            continue
        nms_idx = cv2.dnn.NMSBoxes(b_xywh.tolist(), s, score_threshold=0.0, nms_threshold=iou_thres)
        for i in nms_idx:
            keep.append(idxs[int(i)])
    return keep


def postprocess_v8(output: np.ndarray, conf: float,
                   ratio: float, pad: tuple, orig_shape: tuple) -> DetectionResult:
    """YOLOv8/v9/v11 출력: (1, 4+N, 8400)"""
    logits = output[0]                    # (4+N, 8400)
    logits = logits.T                     # (8400, 4+N)
    boxes_xywh = logits[:, :4]
    class_scores = logits[:, 4:]

    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1).astype(np.int32)
    mask = max_scores > conf
    if not mask.any():
        return DetectionResult.empty()

    boxes_xywh = boxes_xywh[mask]
    scores = max_scores[mask]
    class_ids = class_ids[mask]

    # xywh(letterbox 공간) → xyxy(원본 공간)
    boxes_xyxy = _xywh_to_xyxy_unscale(boxes_xywh, ratio, pad, orig_shape)
    keep = _nms(boxes_xyxy, scores, class_ids, _NMS_IOU)
    if not keep:
        return DetectionResult.empty()

    return DetectionResult(
        boxes=boxes_xyxy[keep].astype(np.float32),
        scores=scores[keep].astype(np.float32),
        class_ids=class_ids[keep],
        infer_ms=0.0,
    )


def postprocess_v5(output: np.ndarray, conf: float,
                   ratio: float, pad: tuple, orig_shape: tuple) -> DetectionResult:
    """YOLOv5/v7 출력: (1, 25200, 5+N) — objectness × class_prob"""
    logits = output[0]                    # (25200, 5+N)
    obj = logits[:, 4:5]
    class_scores = logits[:, 5:] * obj
    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1).astype(np.int32)
    mask = max_scores > conf
    if not mask.any():
        return DetectionResult.empty()

    boxes_xywh = logits[mask, :4]
    scores = max_scores[mask]
    class_ids = class_ids[mask]

    boxes_xyxy = _xywh_to_xyxy_unscale(boxes_xywh, ratio, pad, orig_shape)
    keep = _nms(boxes_xyxy, scores, class_ids, _NMS_IOU)
    if not keep:
        return DetectionResult.empty()

    return DetectionResult(
        boxes=boxes_xyxy[keep].astype(np.float32),
        scores=scores[keep].astype(np.float32),
        class_ids=class_ids[keep],
        infer_ms=0.0,
    )


def _xywh_to_xyxy_unscale(boxes_xywh: np.ndarray, ratio: float,
                            pad: tuple, orig_shape: tuple) -> np.ndarray:
    """letterbox 좌표 → 원본 이미지 좌표 (clip 포함)"""
    pad_w, pad_h = pad
    oh, ow = orig_shape[:2]
    cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = (cx - bw / 2 - pad_w) / ratio
    y1 = (cy - bh / 2 - pad_h) / ratio
    x2 = (cx + bw / 2 - pad_w) / ratio
    y2 = (cy + bh / 2 - pad_h) / ratio
    x1 = np.clip(x1, 0, ow)
    y1 = np.clip(y1, 0, oh)
    x2 = np.clip(x2, 0, ow)
    y2 = np.clip(y2, 0, oh)
    return np.stack([x1, y1, x2, y2], axis=1)


def postprocess_darknet(output: np.ndarray, conf: float,
                        orig_shape: tuple) -> DetectionResult:
    """DarkCenterNet 출력: (1, N, M)
    bbox[:, 0:4] = 정규화된 tlxtlywh (top-left x, top-left y, w, h)
    bbox[:, 4]   = confidence
    bbox[:, 5]   = class_id
    NMS 없음 (CenterNet은 NMS-free 구조)
    """
    bboxes = output[0]              # (N, M)
    scores = bboxes[:, 4]
    mask = scores > conf
    if not mask.any():
        return DetectionResult.empty()

    bboxes = bboxes[mask]
    scores = bboxes[:, 4].astype(np.float32)
    class_ids = bboxes[:, 5].astype(np.int32)

    oh, ow = orig_shape[:2]
    x1 = np.clip(bboxes[:, 0] * ow, 0, ow)
    y1 = np.clip(bboxes[:, 1] * oh, 0, oh)
    x2 = np.clip((bboxes[:, 0] + bboxes[:, 2]) * ow, 0, ow)
    y2 = np.clip((bboxes[:, 1] + bboxes[:, 3]) * oh, 0, oh)
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    # extra_attrs: bbox[6:13] = fall,crawl,jump,front,back,side,mask
    extra = bboxes[:, 6:13].astype(np.float32) if bboxes.shape[1] > 6 else None

    return DetectionResult(
        boxes=boxes_xyxy,
        scores=scores,
        class_ids=class_ids,
        infer_ms=0.0,
        extra_attrs=extra,
    )


def convert_darknet_to_unified(
    result: DetectionResult,
    fall_thresh: float = 0.5,
    crawl_thresh: float = 0.5,
    jump_thresh: float = 0.5,
) -> DetectionResult:
    """darknet 5-class + extra_attrs → 통합 14-class 변환.

    person(0) 하나당 이벤트(1-3) + 방향(4-6) 각각 별도 detection 생성.
    face(1) → nomask(7)/mask(9).
    red-sign(2)→10, wheelchair(3)→11, cane(4)→12.
    """
    if len(result.boxes) == 0:
        return DetectionResult.empty()

    boxes_out, scores_out, cids_out = [], [], []

    for i in range(len(result.boxes)):
        box = result.boxes[i]
        score = result.scores[i]
        cid = int(result.class_ids[i])
        extra = result.extra_attrs[i] if result.extra_attrs is not None else None

        if cid == 0 and extra is not None:
            fall, crawl, jump = extra[0], extra[1], extra[2]
            # person은 항상 생성 + 이벤트는 임계값 초과 시만 추가
            boxes_out.append(box); scores_out.append(score); cids_out.append(0)
            if fall > fall_thresh:
                boxes_out.append(box); scores_out.append(float(fall)); cids_out.append(1)
            if crawl > crawl_thresh:
                boxes_out.append(box); scores_out.append(float(crawl)); cids_out.append(2)
            if jump > jump_thresh:
                boxes_out.append(box); scores_out.append(float(jump)); cids_out.append(3)

        elif cid == 1 and extra is not None:
            mask_prob = extra[6]
            if mask_prob > 0.5:
                boxes_out.append(box); scores_out.append(score); cids_out.append(9)   # mask
            else:
                boxes_out.append(box); scores_out.append(score); cids_out.append(7)   # nomask

        elif cid == 2:
            boxes_out.append(box); scores_out.append(score); cids_out.append(10)
        elif cid == 3:
            boxes_out.append(box); scores_out.append(score); cids_out.append(11)
        elif cid == 4:
            boxes_out.append(box); scores_out.append(score); cids_out.append(12)
        else:
            boxes_out.append(box); scores_out.append(score); cids_out.append(cid)

    return DetectionResult(
        boxes=np.array(boxes_out, dtype=np.float32).reshape(-1, 4),
        scores=np.array(scores_out, dtype=np.float32),
        class_ids=np.array(cids_out, dtype=np.int32),
        infer_ms=result.infer_ms,
    )


def run_inference(model_info: ModelInfo, frame: np.ndarray,
                  conf: float) -> DetectionResult:
    if model_info.session is None:
        return DetectionResult.empty()

    orig_shape = frame.shape
    bs = model_info.batch_size
    t0 = time.perf_counter()

    if model_info.model_type == "darknet":
        tensor = preprocess_darknet(frame, model_info.input_size)
        if bs > 1:
            tensor = np.repeat(tensor, bs, axis=0)
        output = model_info.session.run(None, {model_info.input_name: tensor})
        infer_ms = (time.perf_counter() - t0) * 1000.0
        result = postprocess_darknet(output[0][:1], conf, orig_shape)
    else:
        _, ratio, pad = letterbox(frame, model_info.input_size)
        tensor = preprocess(frame, model_info.input_size)
        if bs > 1:
            tensor = np.repeat(tensor, bs, axis=0)
        output = model_info.session.run(None, {model_info.input_name: tensor})
        infer_ms = (time.perf_counter() - t0) * 1000.0
        single_out = output[0][:1]  # 첫 번째 결과만
        if model_info.output_layout == "v8":
            result = postprocess_v8(single_out, conf, ratio, pad, orig_shape)
        else:
            result = postprocess_v5(single_out, conf, ratio, pad, orig_shape)

    result.infer_ms = infer_ms
    return result


def run_inference_batch(model_info: ModelInfo, frames: list,
                        conf: float) -> "list[DetectionResult]":
    """여러 프레임을 배치로 묶어 한번에 추론. 고정/동적 배치 모두 지원."""
    if model_info.session is None or not frames:
        return [DetectionResult.empty() for _ in frames]

    n = len(frames)
    bs = model_info.batch_size
    orig_shapes = [f.shape for f in frames]
    t0 = time.perf_counter()

    if model_info.model_type == "darknet":
        tensors = [preprocess_darknet(f, model_info.input_size) for f in frames]
        batch_tensor = np.concatenate(tensors, axis=0)
        if bs > n:
            batch_tensor = np.concatenate([batch_tensor, np.repeat(tensors[-1], bs - n, axis=0)], axis=0)
        output = model_info.session.run(None, {model_info.input_name: batch_tensor})
        infer_ms = (time.perf_counter() - t0) * 1000.0
        results = []
        for i in range(n):
            r = postprocess_darknet(output[0][i:i+1], conf, orig_shapes[i])
            r.infer_ms = infer_ms / n
            results.append(r)
    else:
        preprocessed = [letterbox(f, model_info.input_size) for f in frames]
        tensors = []
        for padded, _, _ in preprocessed:
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            tensors.append(np.ascontiguousarray(
                rgb.transpose(2, 0, 1)[np.newaxis], dtype=np.float32) / 255.0)
        batch_tensor = np.concatenate(tensors, axis=0)
        if bs > n:
            batch_tensor = np.concatenate([batch_tensor, np.repeat(tensors[-1], bs - n, axis=0)], axis=0)
        output = model_info.session.run(None, {model_info.input_name: batch_tensor})
        infer_ms = (time.perf_counter() - t0) * 1000.0
        results = []
        for i in range(n):
            _, ratio, pad = preprocessed[i]
            single = output[0][i:i+1]
            if model_info.output_layout == "v8":
                r = postprocess_v8(single, conf, ratio, pad, orig_shapes[i])
            else:
                r = postprocess_v5(single, conf, ratio, pad, orig_shapes[i])
            r.infer_ms = infer_ms / n
            results.append(r)

    return results


# ------------------------------------------------------------------ #
# Classification 추론
# ------------------------------------------------------------------ #

def preprocess_classification(frame: np.ndarray, input_size: tuple) -> np.ndarray:
    """BGR frame → NCHW float32 [0,1] (center crop + resize)"""
    h, w = frame.shape[:2]
    # center crop to square
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = frame[y0:y0+s, x0:x0+s]
    resized = cv2.resize(crop, (input_size[1], input_size[0]))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis], dtype=np.float32) / 255.0
    return tensor


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def run_classification(model_info: ModelInfo, frame: np.ndarray,
                       top_k: int = 5) -> ClassificationResult:
    """Classification 모델 추론"""
    if model_info.session is None:
        return ClassificationResult(0, 0.0, [], np.zeros(0), 0.0)

    bs = model_info.batch_size
    t0 = time.perf_counter()
    tensor = preprocess_classification(frame, model_info.input_size)
    if bs > 1:
        tensor = np.repeat(tensor, bs, axis=0)
    output = model_info.session.run(None, {model_info.input_name: tensor})
    infer_ms = (time.perf_counter() - t0) * 1000.0

    logits = output[0][0].flatten()  # 첫 번째 결과만
    probs = _softmax(logits)
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_results = [(int(i), float(probs[i])) for i in top_indices]

    return ClassificationResult(
        class_id=int(top_indices[0]),
        confidence=float(probs[top_indices[0]]),
        top_k=top_results,
        probabilities=probs,
        infer_ms=infer_ms,
    )
