"""CLIP/VLM 추론: 이미지-텍스트 유사도, zero-shot classification"""
import os
import numpy as np
import cv2
import onnxruntime as ort


class CLIPModel:
    """ONNX CLIP 모델 래퍼. image_encoder + text_encoder 또는 통합 모델."""

    def __init__(self, image_encoder_path: str, text_encoder_path: str = None):
        self.img_session = ort.InferenceSession(image_encoder_path)
        self.txt_session = ort.InferenceSession(text_encoder_path) if text_encoder_path else None
        self._img_input = self.img_session.get_inputs()[0]
        # 입력 shape 자동 감지: [B,C,H,W] 또는 [B,H,W,C] 또는 동적
        shape = self._img_input.shape
        try:
            if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                self._img_size = (shape[2], shape[3])  # NCHW
            elif len(shape) == 4 and isinstance(shape[1], int) and isinstance(shape[2], int):
                self._img_size = (shape[1], shape[2])  # NHWC
            else:
                self._img_size = (224, 224)  # 기본값
        except (IndexError, TypeError):
            self._img_size = (224, 224)
        self._img_bs = shape[0] if isinstance(shape[0], int) and shape[0] > 0 else 1

    def encode_image(self, frame: np.ndarray) -> np.ndarray:
        """BGR frame → normalized embedding vector"""
        tensor = self._preprocess_image(frame)
        if self._img_bs > 1:
            tensor = np.repeat(tensor, self._img_bs, axis=0)
        # 모든 입력 피드 구성
        feed = {}
        for inp in self.img_session.get_inputs():
            if inp.name == self._img_input.name:
                feed[inp.name] = tensor
            else:
                shape = [s if isinstance(s, int) and s > 0 else tensor.shape[0] if i == 0 else 1
                         for i, s in enumerate(inp.shape)]
                feed[inp.name] = np.zeros(shape, dtype=np.float32 if "float" in (inp.type or "") else np.int64)
        out = self.img_session.run(None, feed)
        emb = out[0][0].flatten().astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)

    def encode_text(self, tokens: np.ndarray) -> np.ndarray:
        """토큰 배열 → normalized embedding vector"""
        if self.txt_session is None:
            raise RuntimeError("텍스트 인코더가 없습니다.")
        inputs = self.txt_session.get_inputs()
        # 배치 크기 맞추기
        bs = inputs[0].shape[0]
        bs = int(bs) if isinstance(bs, int) and bs > 0 else 1
        if bs > 1 and tokens.shape[0] < bs:
            tokens = np.repeat(tokens, bs, axis=0)[:bs]
        # 모든 입력 피드 구성
        feed = {}
        for inp in inputs:
            name = inp.name
            if name == "input_ids":
                feed[name] = tokens
            elif name == "attention_mask":
                feed[name] = np.ones_like(tokens, dtype=np.int64)
            elif name == "position_ids":
                feed[name] = np.arange(tokens.shape[1], dtype=np.int64)[np.newaxis].repeat(tokens.shape[0], axis=0)
            else:
                # 알 수 없는 입력은 shape에 맞는 zeros
                shape = [s if isinstance(s, int) and s > 0 else tokens.shape[0] if i == 0 else 1
                         for i, s in enumerate(inp.shape)]
                feed[name] = np.zeros(shape, dtype=np.float32 if "float" in (inp.type or "") else np.int64)
        out = self.txt_session.run(None, feed)
        emb = out[0][0].flatten().astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)

    def similarity(self, img_emb: np.ndarray, txt_emb: np.ndarray) -> float:
        """cosine similarity"""
        return float(np.dot(img_emb, txt_emb))

    def zero_shot_classify(self, frame: np.ndarray, text_embeddings: list,
                           labels: list) -> list:
        """이미지를 여러 텍스트 임베딩과 비교 → [(label, score), ...] 내림차순"""
        img_emb = self.encode_image(frame)
        scores = [float(np.dot(img_emb, te)) for te in text_embeddings]
        # softmax
        exp = np.exp(np.array(scores) * 100)  # temperature scaling
        probs = exp / exp.sum()
        ranked = sorted(zip(labels, probs.tolist()), key=lambda x: -x[1])
        return ranked

    def _preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """CLIP 이미지 전처리: resize, center crop, normalize"""
        h, w = frame.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        crop = frame[y0:y0+s, x0:x0+s]
        resized = cv2.resize(crop, (self._img_size[1], self._img_size[0]))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        rgb = (rgb - mean) / std
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)


def simple_tokenize(text: str, context_length: int = 77) -> np.ndarray:
    """간단한 CLIP 토크나이저 대체: 문자 기반 인코딩 (실제 사용 시 clip tokenizer 권장)"""
    # SOT=49406, EOT=49407
    sot, eot = 49406, 49407
    tokens = [sot]
    for ch in text.lower()[:context_length - 2]:
        tokens.append(ord(ch) % 49405 + 1)
    tokens.append(eot)
    tokens += [0] * (context_length - len(tokens))
    return np.array([tokens], dtype=np.int64)
