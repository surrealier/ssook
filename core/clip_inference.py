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
        self._img_size = (self._img_input.shape[2], self._img_input.shape[3])  # (H, W)
        self._img_bs = self._img_input.shape[0]
        self._img_bs = int(self._img_bs) if isinstance(self._img_bs, int) and self._img_bs > 0 else 1

    def encode_image(self, frame: np.ndarray) -> np.ndarray:
        """BGR frame → normalized embedding vector"""
        tensor = self._preprocess_image(frame)
        if self._img_bs > 1:
            tensor = np.repeat(tensor, self._img_bs, axis=0)
        out = self.img_session.run(None, {self._img_input.name: tensor})
        emb = out[0][0].flatten().astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)

    def encode_text(self, tokens: np.ndarray) -> np.ndarray:
        """토큰 배열 → normalized embedding vector"""
        if self.txt_session is None:
            raise RuntimeError("텍스트 인코더가 없습니다.")
        inp = self.txt_session.get_inputs()[0]
        bs = inp.shape[0]
        bs = int(bs) if isinstance(bs, int) and bs > 0 else 1
        if bs > 1 and tokens.shape[0] < bs:
            tokens = np.repeat(tokens, bs, axis=0)[:bs]
        out = self.txt_session.run(None, {inp.name: tokens})
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
