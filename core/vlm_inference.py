"""VLM(Vision-Language Model) inference — CLIP-based v1.

Backends:
- CLIPCaptioner: best-of-prompts captioning by zero-shot scoring a prompt bank.
- CLIPVQA: zero-shot answer selection from a candidate list (or yes/no default).

BLIP encoder-decoder generative backends are out of scope for v1 — they need
a tokenizer asset and beam decode wired up. Tracking issue: VLM v1.1.

Usage:
    backend = get_backend(model_path, text_encoder=text_encoder_path)
    text = backend.caption(frame, hint=None)            # → str
    text = backend.vqa(frame, question, candidates=...) # → str
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional, Protocol

import numpy as np

from core.clip_inference import CLIPModel, simple_tokenize


# ── Prompt banks ────────────────────────────────────────────────────────
# Curated short list — chosen for breadth over depth. CLIP zero-shot is not
# generative captioning; with a ~60-item vocab it can return a useful
# one-line description without paying a 100ms penalty per image.
_OBJECT_VOCAB: tuple[str, ...] = (
    "a person", "a group of people", "a child", "a baby",
    "a dog", "a cat", "a horse", "a cow", "a sheep", "a bird",
    "a car", "a truck", "a bus", "a motorcycle", "a bicycle", "an airplane",
    "a boat", "a train",
    "a building", "a house", "a skyscraper", "a bridge", "a street",
    "a tree", "a forest", "a mountain", "a beach", "a river", "a lake",
    "a sky", "a sunset", "a cloud", "snow", "rain",
    "food", "a meal on a plate", "a drink", "a cup of coffee", "a fruit",
    "a book", "a laptop", "a phone", "a television", "a chair", "a table",
    "a bed", "a window", "a door",
    "a flower", "a plant",
    "a sports field", "a stadium", "a sports player",
    "an indoor scene", "an outdoor scene", "a nighttime scene", "a daytime scene",
    "text or signage", "a chart or diagram", "a logo",
    "an empty scene", "a crowd",
)

_TEMPLATES: tuple[str, ...] = (
    "a photo of {}",
    "a picture of {}",
    "an image showing {}",
)

_YES_NO: tuple[str, ...] = ("yes", "no")


# ── Backend protocol ────────────────────────────────────────────────────
class VLMBackend(Protocol):
    def caption(self, frame: np.ndarray, hint: Optional[str] = None) -> str: ...
    def vqa(self, frame: np.ndarray, question: str,
            candidates: Optional[Iterable[str]] = None) -> str: ...


# ── CLIP backend ────────────────────────────────────────────────────────
class CLIPCaptioner:
    """Best-of-prompts captioner. Encodes a prompt bank once, then ranks any
    incoming frame against it via cosine similarity. Hint adds extra prompts
    derived from the user-supplied string.
    """

    def __init__(self, image_encoder: str, text_encoder: str):
        if not text_encoder:
            raise ValueError("CLIPCaptioner requires a text_encoder path")
        self._clip = CLIPModel(image_encoder, text_encoder)
        self._cache: dict[str, np.ndarray] = {}

    def _encode_prompts(self, prompts: list[str]) -> list[np.ndarray]:
        embs = []
        for p in prompts:
            cached = self._cache.get(p)
            if cached is None:
                cached = self._clip.encode_text(simple_tokenize(p))
                self._cache[p] = cached
            embs.append(cached)
        return embs

    def caption(self, frame: np.ndarray, hint: Optional[str] = None) -> str:
        prompts: list[str] = []
        labels: list[str] = []
        for obj in _OBJECT_VOCAB:
            for tmpl in _TEMPLATES:
                prompts.append(tmpl.format(obj))
                labels.append(obj.lstrip("a ").lstrip("an "))
        # Inject hint as its own candidate so user intent biases the result.
        hint_clean = (hint or "").strip()
        if hint_clean and hint_clean.lower() not in {"describe this image.", "describe this image"}:
            prompts.append(hint_clean)
            labels.append(hint_clean)
        text_embs = self._encode_prompts(prompts)
        ranked = self._clip.zero_shot_classify(frame, text_embs, prompts)
        # ranked = [(prompt, score), ...] sorted desc. Map back to short label.
        top_prompt, top_score = ranked[0]
        # Strip the template prefix for a cleaner one-liner.
        for tmpl in _TEMPLATES:
            prefix = tmpl.format("").strip()
            if top_prompt.startswith(prefix):
                top_prompt = top_prompt[len(prefix):].strip()
                break
        return f"{top_prompt} (confidence {top_score:.2f})"

    def vqa(self, frame: np.ndarray, question: str,
            candidates: Optional[Iterable[str]] = None) -> str:
        cands = [c.strip() for c in (candidates or _YES_NO) if c and c.strip()]
        if not cands:
            cands = list(_YES_NO)
        # Frame the question + answer into a single sentence that CLIP can score.
        q = (question or "").strip().rstrip("?").rstrip(".")
        prompts = [f"Question: {q}? Answer: {c}." for c in cands]
        text_embs = self._encode_prompts(prompts)
        ranked = self._clip.zero_shot_classify(frame, text_embs, cands)
        answer, score = ranked[0]
        return f"{answer} (confidence {score:.2f})"


# ── Factory ─────────────────────────────────────────────────────────────
def get_backend(model_path: str, *, text_encoder: Optional[str] = None) -> VLMBackend:
    """Pick a backend by heuristics over the ONNX inputs.

    Today the only supported backend is CLIP (encoder pair). A future BLIP
    backend would branch here on detecting decoder-style inputs.
    """
    if not text_encoder:
        raise NotImplementedError(
            "VLM v1 only supports CLIP-style image+text encoder pairs. "
            "Provide a text_encoder ONNX path alongside the image encoder. "
            "(BLIP-style single-file VLMs are tracked for v1.1.)"
        )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Image encoder not found: {model_path}")
    if not os.path.isfile(text_encoder):
        raise FileNotFoundError(f"Text encoder not found: {text_encoder}")
    return CLIPCaptioner(model_path, text_encoder)
