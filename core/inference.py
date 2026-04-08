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


class PreprocessBuffer:
    """전처리 버퍼 재사용 — 동일 input_size에 대해 배열 재할당 방지"""
    __slots__ = ('_padded', '_rgb', '_tensor', '_size')

    def __init__(self):
        self._padded = None
        self._rgb = None
        self._tensor = None
        self._size = None

    def get_buffers(self, input_size: tuple):
        nh, nw = input_size
        if self._size == input_size and self._tensor is not None:
            return self._padded, self._rgb, self._tensor
        self._padded = np.empty((nh, nw, 3), dtype=np.uint8)
        self._rgb = np.empty((nh, nw, 3), dtype=np.uint8)
        self._tensor = np.empty((1, 3, nh, nw), dtype=np.float32)
        self._size = input_size
        return self._padded, self._rgb, self._tensor


# 모듈 레벨 싱글톤 버퍼 (단일 스레드 추론용)
_preproc_buf = PreprocessBuffer()


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
    """BGR frame → NCHW float32 [0,1] (버퍼 재사용)"""
    padded, ratio, pad = letterbox(frame, input_size)
    return _padded_to_tensor(padded, input_size)


def _padded_to_tensor(padded: np.ndarray, input_size: tuple) -> np.ndarray:
    """letterbox 결과 → NCHW float32 [0,1] (버퍼 재사용, 내부용)"""
    _, rgb_buf, tensor_buf = _preproc_buf.get_buffers(input_size)
    cv2.cvtColor(padded, cv2.COLOR_BGR2RGB, dst=rgb_buf)
    np.divide(rgb_buf.transpose(2, 0, 1), 255.0, out=tensor_buf[0])
    return tensor_buf


def preprocess_darknet(frame: np.ndarray, input_size: tuple) -> np.ndarray:
    """DarkCenterNet 전처리: 단순 resize (letterbox 없음), BGR→RGB, NCHW, float32/255"""
    h, w = input_size
    img = cv2.resize(frame, (w, h))
    img = img[..., ::-1]            # BGR→RGB
    img = img.transpose(2, 0, 1)   # HWC→CHW
    return np.ascontiguousarray(img[np.newaxis], dtype=np.float32) / 255.0


def _nms(boxes_xyxy, scores, class_ids, iou_thres):
    """클래스별 NMS → 최종 인덱스 반환 (offset 트릭으로 단일 호출)"""
    if len(scores) == 0:
        return []
    # 클래스별 offset을 적용하여 class-agnostic NMS 한 번으로 처리
    max_coord = boxes_xyxy.max()
    offsets = class_ids.astype(np.float32) * (max_coord + 1.0)
    shifted = boxes_xyxy.copy()
    shifted[:, 0] += offsets
    shifted[:, 1] += offsets
    shifted[:, 2] += offsets
    shifted[:, 3] += offsets
    # xywh 변환
    b_xywh = np.empty_like(shifted)
    b_xywh[:, 0] = shifted[:, 0]
    b_xywh[:, 1] = shifted[:, 1]
    b_xywh[:, 2] = shifted[:, 2] - shifted[:, 0]
    b_xywh[:, 3] = shifted[:, 3] - shifted[:, 1]
    nms_idx = cv2.dnn.NMSBoxes(b_xywh.tolist(), scores.tolist(),
                                score_threshold=0.0, nms_threshold=iou_thres)
    return [int(i) for i in nms_idx] if len(nms_idx) > 0 else []


def postprocess_v8(output: np.ndarray, conf: float,
                   ratio: float, pad: tuple, orig_shape: tuple) -> DetectionResult:
    """YOLOv8/v9/v11 출력: (1, 4+N, 8400)"""
    logits = output[0]                    # (4+N, 8400)
    # transpose 전에 class score max로 조기 필터링
    class_part = logits[4:]               # (N, 8400)
    max_scores = class_part.max(axis=0)   # (8400,)
    mask = max_scores > conf
    if not mask.any():
        return DetectionResult.empty()

    # 필터링된 것만 처리
    filtered = logits[:, mask].T          # (K, 4+N) where K << 8400
    boxes_xywh = filtered[:, :4]
    class_scores = filtered[:, 4:]
    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1).astype(np.int32)

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



def postprocess_detr(outputs: list, conf: float,
                     ratio: float, pad: tuple, orig_shape: tuple) -> DetectionResult:
    """DETR/RT-DETR/RF-DETR 출력 처리.
    지원 형태:
      - 단일 텐서 (1, N, 4+C): boxes + class scores
      - 두 텐서 (1, N, 4) + (1, N, C): boxes, scores 분리
    좌표: cxcywh 정규화(0~1) 또는 xyxy 절대좌표 자동 감지.
    """
    if len(outputs) >= 2 and outputs[0].shape[-1] == 4:
        boxes_raw = outputs[0][0]  # (N, 4)
        scores_raw = outputs[1][0]  # (N, C)
    else:
        data = outputs[0][0]  # (N, 4+C)
        boxes_raw = data[:, :4]
        scores_raw = data[:, 4:]

    max_scores = scores_raw.max(axis=1)
    class_ids = scores_raw.argmax(axis=1).astype(np.int32)
    mask = max_scores > conf
    if not mask.any():
        return DetectionResult.empty()

    boxes_raw = boxes_raw[mask]
    scores = max_scores[mask].astype(np.float32)
    class_ids = class_ids[mask]

    oh, ow = orig_shape[:2]
    # 좌표 형식 자동 감지: 값이 0~1 범위면 정규화된 cxcywh, 아니면 xyxy 절대좌표
    if boxes_raw.max() <= 1.5:
        cx, cy, bw, bh = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
        x1 = np.clip((cx - bw / 2) * ow, 0, ow)
        y1 = np.clip((cy - bh / 2) * oh, 0, oh)
        x2 = np.clip((cx + bw / 2) * ow, 0, ow)
        y2 = np.clip((cy + bh / 2) * oh, 0, oh)
    else:
        # letterbox 좌표 → 원본 좌표
        pad_w, pad_h = pad
        x1 = np.clip((boxes_raw[:, 0] - pad_w) / ratio, 0, ow)
        y1 = np.clip((boxes_raw[:, 1] - pad_h) / ratio, 0, oh)
        x2 = np.clip((boxes_raw[:, 2] - pad_w) / ratio, 0, ow)
        y2 = np.clip((boxes_raw[:, 3] - pad_h) / ratio, 0, oh)

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    return DetectionResult(boxes=boxes_xyxy, scores=scores, class_ids=class_ids, infer_ms=0.0)


def postprocess_yolo_nas(outputs: list, conf: float,
                         ratio: float, pad: tuple, orig_shape: tuple) -> DetectionResult:
    """YOLO-NAS 출력: 두 텐서 (1, N, 4) + (1, N, C) 또는 단일 (1, N, 4+C)."""
    return postprocess_detr(outputs, conf, ratio, pad, orig_shape)


def postprocess_custom(outputs: list, cmt, conf: float,
                       ratio: float, pad: tuple, orig_shape: tuple) -> DetectionResult:
    """사용자 정의 모델 타입에 따른 후처리."""
    oi = cmt.output_index if cmt.output_index < len(outputs) else 0
    data = outputs[oi][0]  # 배치 첫 번째

    attr_roles = cmt.attr_roles or []
    if not attr_roles:
        return DetectionResult.empty()

    # dim_roles로 전치 여부 결정: attrs 차원이 마지막이 아니면 전치
    if len(data.shape) == 2:
        n_items, n_attrs = data.shape
        if n_attrs == len(attr_roles):
            pass  # (N, attrs) 정상
        elif n_items == len(attr_roles):
            data = data.T  # (attrs, N) → (N, attrs)

    # attr_roles에서 좌표/confidence 인덱스 추출
    coord_keys = {"x1", "y1", "x2", "y2", "x_center", "y_center", "width", "height"}
    coord_idx = {}
    conf_indices = []
    single_conf_idx = -1
    class_id_idx = -1
    for i, role in enumerate(attr_roles):
        if role in coord_keys:
            coord_idx[role] = i
        elif role.startswith("conf_class"):
            conf_indices.append(i)
        elif role == "objectness":
            coord_idx["objectness"] = i
        elif role == "confidence":
            single_conf_idx = i
        elif role == "class_id":
            class_id_idx = i

    # confidence + class_id 직접 지정 방식 (CenterNet 등)
    if single_conf_idx >= 0 and class_id_idx >= 0:
        raw_scores = data[:, single_conf_idx]
        if "objectness" in coord_idx:
            raw_scores = raw_scores * data[:, coord_idx["objectness"]]
        mask = raw_scores > conf
        if not mask.any():
            return DetectionResult.empty()
        data = data[mask]
        scores = raw_scores[mask].astype(np.float32)
        class_ids = data[:, class_id_idx].astype(np.int32)
    elif conf_indices:
        class_scores = data[:, conf_indices]
        if "objectness" in coord_idx:
            class_scores = class_scores * data[:, coord_idx["objectness"]:coord_idx["objectness"]+1]
        max_scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1).astype(np.int32)
        mask = max_scores > conf
        if not mask.any():
            return DetectionResult.empty()
        data = data[mask]
        scores = max_scores[mask].astype(np.float32)
        class_ids = class_ids[mask]
    else:
        return DetectionResult.empty()

    oh, ow = orig_shape[:2]
    pad_w, pad_h = pad

    if "x1" in coord_idx and "y1" in coord_idx and "x2" in coord_idx and "y2" in coord_idx:
        raw_x1 = data[:, coord_idx["x1"]]
        raw_y1 = data[:, coord_idx["y1"]]
        raw_x2 = data[:, coord_idx["x2"]]
        raw_y2 = data[:, coord_idx["y2"]]
        if raw_x1.max() <= 1.5:
            x1 = np.clip(raw_x1 * ow, 0, ow)
            y1 = np.clip(raw_y1 * oh, 0, oh)
            x2 = np.clip(raw_x2 * ow, 0, ow)
            y2 = np.clip(raw_y2 * oh, 0, oh)
        else:
            x1 = np.clip((raw_x1 - pad_w) / ratio, 0, ow)
            y1 = np.clip((raw_y1 - pad_h) / ratio, 0, oh)
            x2 = np.clip((raw_x2 - pad_w) / ratio, 0, ow)
            y2 = np.clip((raw_y2 - pad_h) / ratio, 0, oh)
    elif "x1" in coord_idx and "y1" in coord_idx and "width" in coord_idx and "height" in coord_idx:
        # x1/y1/w/h → xyxy 변환
        raw_x1 = data[:, coord_idx["x1"]]
        raw_y1 = data[:, coord_idx["y1"]]
        raw_w = data[:, coord_idx["width"]]
        raw_h = data[:, coord_idx["height"]]
        if raw_x1.max() <= 1.5:
            x1 = np.clip(raw_x1 * ow, 0, ow)
            y1 = np.clip(raw_y1 * oh, 0, oh)
            x2 = np.clip((raw_x1 + raw_w) * ow, 0, ow)
            y2 = np.clip((raw_y1 + raw_h) * oh, 0, oh)
        else:
            x1 = np.clip((raw_x1 - pad_w) / ratio, 0, ow)
            y1 = np.clip((raw_y1 - pad_h) / ratio, 0, oh)
            x2 = np.clip((raw_x1 + raw_w - pad_w) / ratio, 0, ow)
            y2 = np.clip((raw_y1 + raw_h - pad_h) / ratio, 0, oh)
    elif "x_center" in coord_idx and "y_center" in coord_idx and "width" in coord_idx and "height" in coord_idx:
        cx = data[:, coord_idx["x_center"]]
        cy = data[:, coord_idx["y_center"]]
        bw = data[:, coord_idx["width"]]
        bh = data[:, coord_idx["height"]]
        if cx.max() <= 1.5:
            x1 = np.clip((cx - bw / 2) * ow, 0, ow)
            y1 = np.clip((cy - bh / 2) * oh, 0, oh)
            x2 = np.clip((cx + bw / 2) * ow, 0, ow)
            y2 = np.clip((cy + bh / 2) * oh, 0, oh)
        else:
            x1 = np.clip((cx - bw / 2 - pad_w) / ratio, 0, ow)
            y1 = np.clip((cy - bh / 2 - pad_h) / ratio, 0, oh)
            x2 = np.clip((cx + bw / 2 - pad_w) / ratio, 0, ow)
            y2 = np.clip((cy + bh / 2 - pad_h) / ratio, 0, oh)
    else:
        return DetectionResult.empty()

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    if cmt.nms:
        keep = _nms(boxes_xyxy, scores, class_ids, _NMS_IOU)
        if not keep:
            return DetectionResult.empty()
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

    return DetectionResult(boxes=boxes_xyxy, scores=scores, class_ids=class_ids, infer_ms=0.0)

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


def _make_batch_tensor(tensor: np.ndarray, bs: int, model_info: ModelInfo) -> np.ndarray:
    """고정 배치 모델용: 단일 이미지 텐서를 배치 크기로 확장. 캐싱된 버퍼 재사용."""
    if bs <= 1:
        return tensor
    cache = getattr(model_info, '_batch_buf', None)
    if cache is not None and cache.shape[0] == bs and cache.shape[1:] == tensor.shape[1:]:
        cache[0] = tensor[0]
        return cache
    buf = np.repeat(tensor, bs, axis=0)
    model_info._batch_buf = buf
    return buf


def run_inference(model_info: ModelInfo, frame: np.ndarray,
                  conf: float) -> DetectionResult:
    if model_info.session is None:
        return DetectionResult.empty()

    orig_shape = frame.shape
    bs = model_info.batch_size
    t0 = time.perf_counter()

    if model_info.model_type == "darknet":
        tensor = preprocess_darknet(frame, model_info.input_size)
        tensor = _make_batch_tensor(tensor, bs, model_info)
        output = model_info.session.run(None, {model_info.input_name: tensor})
        infer_ms = (time.perf_counter() - t0) * 1000.0
        result = postprocess_darknet(output[0][:1], conf, orig_shape)
    elif model_info.model_type == "custom":
        padded, ratio, pad = letterbox(frame, model_info.input_size)
        tensor = _padded_to_tensor(padded, model_info.input_size)
        tensor = _make_batch_tensor(tensor, bs, model_info)
        output = model_info.session.run(None, {model_info.input_name: tensor})
        infer_ms = (time.perf_counter() - t0) * 1000.0
        # custom_model_type 정보는 ModelInfo에 캐싱된 것을 우선 사용
        cmt = getattr(model_info, '_cached_cmt', None)
        if cmt is None:
            from core.app_config import AppConfig
            cmt = AppConfig().custom_model_types.get(model_info.custom_type_name)
            if cmt:
                model_info._cached_cmt = cmt
        if cmt:
            result = postprocess_custom(output, cmt, conf, ratio, pad, orig_shape)
        else:
            result = DetectionResult.empty()
    elif model_info.model_type == "detr":
        padded, ratio, pad = letterbox(frame, model_info.input_size)
        tensor = _padded_to_tensor(padded, model_info.input_size)
        tensor = _make_batch_tensor(tensor, bs, model_info)
        output = model_info.session.run(None, {model_info.input_name: tensor})
        infer_ms = (time.perf_counter() - t0) * 1000.0
        result = postprocess_detr(output, conf, ratio, pad, orig_shape)
    elif model_info.model_type == "yolo_nas":
        padded, ratio, pad = letterbox(frame, model_info.input_size)
        tensor = _padded_to_tensor(padded, model_info.input_size)
        tensor = _make_batch_tensor(tensor, bs, model_info)
        output = model_info.session.run(None, {model_info.input_name: tensor})
        infer_ms = (time.perf_counter() - t0) * 1000.0
        result = postprocess_yolo_nas(output, conf, ratio, pad, orig_shape)
    else:
        padded, ratio, pad = letterbox(frame, model_info.input_size)
        tensor = _padded_to_tensor(padded, model_info.input_size)
        tensor = _make_batch_tensor(tensor, bs, model_info)
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


def _build_feed(model_info: ModelInfo, tensor: np.ndarray) -> dict:
    """모델의 모든 입력에 대한 feed dict 구성. 이미지 텐서는 input_name에 매핑."""
    session = model_info.session
    inputs = session.get_inputs()
    if len(inputs) == 1:
        return {model_info.input_name: tensor}
    feed = {}
    for inp in inputs:
        if inp.name == model_info.input_name:
            feed[inp.name] = tensor
        elif "mask" in inp.name.lower():
            # attention_mask: ones
            shape = [s if isinstance(s, int) and s > 0 else 1 for s in inp.shape]
            feed[inp.name] = np.ones(shape, dtype=np.int64)
        elif "id" in inp.name.lower() or "token" in inp.name.lower():
            # input_ids / token_type_ids: zeros
            shape = [s if isinstance(s, int) and s > 0 else 1 for s in inp.shape]
            feed[inp.name] = np.zeros(shape, dtype=np.int64)
        elif "position" in inp.name.lower():
            shape = [s if isinstance(s, int) and s > 0 else 1 for s in inp.shape]
            seq_len = shape[-1] if len(shape) >= 2 else 1
            feed[inp.name] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        else:
            shape = [s if isinstance(s, int) and s > 0 else 1 for s in inp.shape]
            dtype = np.float32 if "float" in (inp.type or "tensor(float)") else np.int64
            feed[inp.name] = np.zeros(shape, dtype=dtype)
    return feed


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
    output = model_info.session.run(None, _build_feed(model_info, tensor))
    infer_ms = (time.perf_counter() - t0) * 1000.0

    logits = output[0][0].flatten()
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


# ── Segmentation Inference ──────────────────────────────
@dataclass
class SegmentationResult:
    mask: np.ndarray        # (H, W) class index map
    num_classes: int
    infer_ms: float


def run_segmentation(model_info: ModelInfo, frame: np.ndarray) -> SegmentationResult:
    """Segmentation 모델 추론 → class mask"""
    if model_info.session is None:
        return SegmentationResult(np.zeros((1, 1), dtype=np.uint8), 0, 0.0)
    t0 = time.perf_counter()
    h_orig, w_orig = frame.shape[:2]

    if model_info.model_type.startswith("seg_yolo"):
        padded, ratio, pad = letterbox(frame, model_info.input_size)
        tensor = _padded_to_tensor(padded, model_info.input_size)
    else:
        tensor = preprocess_classification(frame, model_info.input_size)
    bs = model_info.batch_size
    if bs > 1:
        tensor = np.repeat(tensor, bs, axis=0)
    output = model_info.session.run(None, _build_feed(model_info, tensor))
    infer_ms = (time.perf_counter() - t0) * 1000.0

    # YOLO-seg: output0=(1,116,8400) det+mask_coeff, output1=(1,32,160,160) protos
    if (model_info.model_type.startswith("seg_yolo") and len(output) >= 2
            and output[0].ndim == 3 and output[1].ndim == 4):
        det = output[0][0]   # (116, 8400)
        protos = output[1][0]  # (32, mh, mw)
        nm = protos.shape[0]  # 32
        nc = det.shape[0] - 4 - nm  # num_classes
        # Confidence filter
        scores_all = det[4:4+nc, :]  # (nc, 8400)
        max_scores = scores_all.max(axis=0)  # (8400,)
        conf = 0.25
        keep = max_scores > conf
        if keep.sum() == 0:
            return SegmentationResult(np.zeros((h_orig, w_orig), dtype=np.uint8), nc, infer_ms)
        det_keep = det[:, keep]  # (116, N)
        scores_keep = scores_all[:, keep]  # (nc, N)
        class_ids = scores_keep.argmax(axis=0)  # (N,)
        # Boxes: cx,cy,w,h → x1,y1,x2,y2
        cx, cy, bw, bh = det_keep[0], det_keep[1], det_keep[2], det_keep[3]
        x1 = cx - bw / 2; y1 = cy - bh / 2; x2 = cx + bw / 2; y2 = cy + bh / 2
        # NMS
        boxes_nms = np.stack([x1, y1, bw, bh], axis=1).tolist()
        confs_nms = max_scores[keep].tolist()
        indices = cv2.dnn.NMSBoxes(boxes_nms, confs_nms, conf, 0.45)
        if len(indices) == 0:
            return SegmentationResult(np.zeros((h_orig, w_orig), dtype=np.uint8), nc, infer_ms)
        indices = np.array(indices).flatten()
        # Mask coefficients
        mask_coeffs = det_keep[4+nc:, indices].T  # (N_nms, 32)
        # Matmul with protos: (N_nms, 32) @ (32, mh*mw) → (N_nms, mh*mw)
        mh, mw = protos.shape[1], protos.shape[2]
        masks = (mask_coeffs @ protos.reshape(nm, -1)).reshape(-1, mh, mw)
        # Sigmoid
        masks = 1.0 / (1.0 + np.exp(-masks))
        # Composite mask: assign class with highest mask score per pixel
        ih, iw = model_info.input_size
        composite = np.zeros((ih, iw), dtype=np.uint8)
        for i, idx in enumerate(indices):
            m = cv2.resize(masks[i], (iw, ih), interpolation=cv2.INTER_LINEAR)
            cid = int(class_ids[idx]) + 1  # +1 so background=0
            composite[m > 0.5] = cid
        # Remove padding, scale to original
        pad_h, pad_w = int(round(pad[1])), int(round(pad[0]))
        unpadded = composite[pad_h:ih-pad_h if pad_h else ih, pad_w:iw-pad_w if pad_w else iw]
        mask = cv2.resize(unpadded, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        return SegmentationResult(mask=mask, num_classes=nc, infer_ms=infer_ms)

    # Semantic segmentation: single output (B,C,H,W)
    logits = None
    for o in output:
        if o.ndim == 4:
            _, c, oh, ow = o.shape
            min_spatial = min(model_info.input_size) // 4
            if c <= 256 and oh >= min_spatial and ow >= min_spatial:
                logits = o[0]
                break
    if logits is None:
        first = output[0][0] if output[0].ndim >= 3 else output[0]
        if first.ndim == 3 and first.shape[0] <= 256:
            logits = first
        else:
            return SegmentationResult(np.zeros((h_orig, w_orig), dtype=np.uint8), 0, infer_ms)

    if logits.ndim == 3:
        mask = np.argmax(logits, axis=0).astype(np.uint8)
        mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        num_classes = logits.shape[0]
    elif logits.ndim == 2:
        mask = (logits > 0.5).astype(np.uint8)
        mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        num_classes = 1
    else:
        mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        num_classes = 0
    return SegmentationResult(mask=mask, num_classes=num_classes, infer_ms=infer_ms)


# ── Embedding Inference ─────────────────────────────────
@dataclass
class EmbeddingResult:
    embedding: np.ndarray   # normalized feature vector
    dim: int
    infer_ms: float


def run_embedding(model_info: ModelInfo, frame: np.ndarray) -> EmbeddingResult:
    """Embedding 모델 추론 → normalized feature vector"""
    if model_info.session is None:
        return EmbeddingResult(np.zeros(1), 0, 0.0)
    t0 = time.perf_counter()
    tensor = preprocess_classification(frame, model_info.input_size)
    bs = model_info.batch_size
    if bs > 1:
        tensor = np.repeat(tensor, bs, axis=0)
    output = model_info.session.run(None, _build_feed(model_info, tensor))
    infer_ms = (time.perf_counter() - t0) * 1000.0
    emb = output[0][0].flatten().astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return EmbeddingResult(embedding=emb, dim=len(emb), infer_ms=infer_ms)
