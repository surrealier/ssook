"""VLM (Vision-Language Model) inference — pluggable backends.

Three backends share one interface (VLMBackend), so routes and the batch
runner don't care which engine actually produces text:

- CLIPBackend: zero-shot best-of-prompts captioning + candidate-list VQA.
  Dependency-free (uses bundled onnxruntime). This is the default backend
  and keeps working with the hand-provided CLIP image+text encoder pair.
- TransformersBackend: real generative VLMs (Qwen2.5-VL / Qwen2-VL / LLaVA
  …) via a lazy torch+transformers import. Weights auto-download from the
  HF hub on first use; CUDA is used when available.
- OpenAICompatBackend: any OpenAI-/chat-completions-compatible HTTP server
  (vLLM, llama.cpp, Ollama, OpenAI itself) via base64 image_url payloads.

The two generative backends are optional dependencies — `list_backends()`
reports availability so the UI can grey out what isn't installed, and
`make_backend(spec)` validates per backend with actionable errors.

Usage:
    backend = make_backend({"backend": "clip", "model_path": img_enc,
                            "text_encoder": txt_enc})
    text = backend.describe(frame, "Describe this image.")
    text = backend.answer(frame, "Is it daytime?", candidates=["yes", "no"])
"""
from __future__ import annotations

import abc
import base64
import importlib.util
import os
from typing import Iterable, Optional

import numpy as np

from core.clip_inference import CLIPModel, simple_tokenize


# ── Prompt banks (CLIP backend) ─────────────────────────────────────────
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


# ── Backend interface ───────────────────────────────────────────────────
class VLMBackend(abc.ABC):
    """Common interface for every VLM engine.

    `describe`/`answer` take a BGR `np.ndarray` frame (OpenCV convention) so
    callers never have to know the engine's native image format.
    """

    @abc.abstractmethod
    def describe(self, frame: np.ndarray, prompt: str, *,
                 max_new_tokens: int = 128, temperature: float = 0.0) -> str:
        """Free-form description / caption of the frame given a prompt."""

    @abc.abstractmethod
    def answer(self, frame: np.ndarray, question: str, *,
               candidates: Optional[Iterable[str]] = None,
               max_new_tokens: int = 128, temperature: float = 0.0) -> str:
        """Answer a question about the frame, optionally constrained to candidates."""

    @classmethod
    @abc.abstractmethod
    def capabilities(cls) -> dict:
        """Static descriptor: name, tasks, generative, requires_text_encoder, deps."""

    @staticmethod
    @abc.abstractmethod
    def is_available() -> bool:
        """True if this backend's optional dependencies are importable."""


# ── CLIP backend (default, dependency-free) ─────────────────────────────
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


class CLIPBackend(VLMBackend):
    """VLMBackend adapter over CLIPCaptioner. Always available — onnxruntime
    is a core dependency. `temperature`/`max_new_tokens` are ignored because
    CLIP is a discriminative scorer, not a generator.
    """

    def __init__(self, image_encoder: str, text_encoder: str):
        self._captioner = CLIPCaptioner(image_encoder, text_encoder)

    def describe(self, frame: np.ndarray, prompt: str, *,
                 max_new_tokens: int = 128, temperature: float = 0.0) -> str:
        return self._captioner.caption(frame, hint=prompt or None)

    def answer(self, frame: np.ndarray, question: str, *,
               candidates: Optional[Iterable[str]] = None,
               max_new_tokens: int = 128, temperature: float = 0.0) -> str:
        return self._captioner.vqa(frame, question, candidates=candidates)

    @classmethod
    def capabilities(cls) -> dict:
        return {
            "name": "clip",
            "tasks": ["caption", "vqa"],
            "generative": False,
            "requires_text_encoder": True,
            "deps": [],
        }

    @staticmethod
    def is_available() -> bool:
        # onnxruntime is a hard dependency of ssook, so CLIP is always usable.
        return True


# ── Transformers backend (generative, optional) ─────────────────────────
class TransformersBackend(VLMBackend):
    """Generative VLM via HuggingFace transformers (Qwen2.5-VL, LLaVA, …).

    torch + transformers are imported lazily inside __init__ so the module
    keeps importing on a bare CLIP-only install. Weights auto-download from
    the HF hub on first construction (cached by transformers thereafter).
    """

    _DEPS = ("transformers", "torch")

    def __init__(self, model_id: str, *, device: Optional[str] = None):
        if not model_id:
            raise ValueError("TransformersBackend requires a model_id (HF repo or local path)")
        if not self.is_available():
            raise RuntimeError(
                "transformers backend needs torch + transformers. "
                "Install with: pip install -r requirements-vlm.txt"
            )
        import torch  # lazy: optional dependency
        from transformers import AutoProcessor

        self._torch = torch
        self._model_id = model_id
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = self._load_model(model_id, device, torch)

    @staticmethod
    def _load_model(model_id: str, device: str, torch):
        """Load the model with the generic image-text-to-text head, falling
        back to the Qwen2.5-VL class for older transformers that lack the
        AutoModel alias for that architecture.
        """
        dtype = torch.float16 if device == "cuda" else torch.float32
        device_map = "cuda" if device == "cuda" else "cpu"
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype=dtype, device_map=device_map,
                trust_remote_code=True,
            )
        except (ImportError, ValueError, KeyError):
            # Older/edge architectures: try the Qwen2.5-VL concrete class.
            from transformers import Qwen2_5_VLForConditionalGeneration
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, device_map=device_map,
                trust_remote_code=True,
            )

    def _to_pil(self, frame: np.ndarray):
        import cv2  # core dependency
        from PIL import Image
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def _generate(self, frame: np.ndarray, prompt: str,
                  max_new_tokens: int, temperature: float) -> str:
        image = self._to_pil(frame)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=[image], return_tensors="pt")
        inputs = inputs.to(self._model.device)
        do_sample = temperature > 0.0
        gen_kwargs = {"max_new_tokens": int(max_new_tokens), "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
        with self._torch.inference_mode():
            generated = self._model.generate(**inputs, **gen_kwargs)
        # Strip the prompt tokens so we decode only the freshly generated span.
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
        decoded = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return (decoded[0] if decoded else "").strip()

    def describe(self, frame: np.ndarray, prompt: str, *,
                 max_new_tokens: int = 128, temperature: float = 0.0) -> str:
        return self._generate(frame, prompt or "Describe this image.",
                              max_new_tokens, temperature)

    def answer(self, frame: np.ndarray, question: str, *,
               candidates: Optional[Iterable[str]] = None,
               max_new_tokens: int = 128, temperature: float = 0.0) -> str:
        prompt = question or ""
        cands = [c.strip() for c in (candidates or []) if c and c.strip()]
        if cands:
            # Constrain a generative model by naming the options in-prompt.
            prompt = f"{prompt}\nAnswer with one of: {', '.join(cands)}."
        return self._generate(frame, prompt, max_new_tokens, temperature)

    @classmethod
    def capabilities(cls) -> dict:
        return {
            "name": "transformers",
            "tasks": ["caption", "vqa"],
            "generative": True,
            "requires_text_encoder": False,
            "deps": list(cls._DEPS),
        }

    @staticmethod
    def is_available() -> bool:
        return all(importlib.util.find_spec(dep) is not None
                   for dep in TransformersBackend._DEPS)


# ── OpenAI-compatible HTTP backend (generative, optional) ───────────────
class OpenAICompatBackend(VLMBackend):
    """Talk to any OpenAI chat-completions-compatible endpoint.

    Works with vLLM, llama.cpp server, Ollama (/v1), LM Studio, and OpenAI
    itself. The image is sent inline as a base64 data URL so no file upload
    step is needed. httpx is preferred; requests is a fallback.
    """

    def __init__(self, endpoint_url: str, model_id: str, *, api_key: Optional[str] = None):
        if not endpoint_url:
            raise ValueError("OpenAICompatBackend requires an endpoint_url")
        if not model_id:
            raise ValueError("OpenAICompatBackend requires a model_id")
        if not self.is_available():
            raise RuntimeError(
                "openai backend needs httpx or requests. "
                "Install with: pip install httpx"
            )
        self._endpoint = endpoint_url.rstrip("/")
        self._model_id = model_id
        self._api_key = api_key or None

    def _frame_to_data_url(self, frame: np.ndarray) -> str:
        import cv2
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Failed to JPEG-encode frame for VLM request")
        b64 = base64.b64encode(buf).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _chat(self, frame: np.ndarray, prompt: str,
              max_new_tokens: int, temperature: float) -> str:
        data_url = self._frame_to_data_url(frame)
        payload = {
            "model": self._model_id,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        url = f"{self._endpoint}/chat/completions"
        data = self._post_json(url, payload, headers)
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected response shape from {url}: {e}") from e

    @staticmethod
    def _post_json(url: str, payload: dict, headers: dict) -> dict:
        # WHY lazy + dual-client: httpx is the modern default but requests is
        # far more commonly already installed; accept either.
        if importlib.util.find_spec("httpx") is not None:
            import httpx
            try:
                resp = httpx.post(url, json=payload, headers=headers, timeout=120.0)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPError as e:
                raise RuntimeError(f"VLM HTTP request failed: {e}") from e
        import requests
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120.0)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"VLM HTTP request failed: {e}") from e

    def describe(self, frame: np.ndarray, prompt: str, *,
                 max_new_tokens: int = 128, temperature: float = 0.0) -> str:
        return self._chat(frame, prompt or "Describe this image.",
                          max_new_tokens, temperature)

    def answer(self, frame: np.ndarray, question: str, *,
               candidates: Optional[Iterable[str]] = None,
               max_new_tokens: int = 128, temperature: float = 0.0) -> str:
        prompt = question or ""
        cands = [c.strip() for c in (candidates or []) if c and c.strip()]
        if cands:
            prompt = f"{prompt}\nAnswer with one of: {', '.join(cands)}."
        return self._chat(frame, prompt, max_new_tokens, temperature)

    @classmethod
    def capabilities(cls) -> dict:
        return {
            "name": "openai",
            "tasks": ["caption", "vqa"],
            "generative": True,
            "requires_text_encoder": False,
            "deps": ["httpx"],
        }

    @staticmethod
    def is_available() -> bool:
        return (importlib.util.find_spec("httpx") is not None
                or importlib.util.find_spec("requests") is not None)


# ── Registry ────────────────────────────────────────────────────────────
BACKENDS: dict[str, type[VLMBackend]] = {
    "clip": CLIPBackend,
    "transformers": TransformersBackend,
    "openai": OpenAICompatBackend,
}


def _missing_deps(cls: type[VLMBackend]) -> list[str]:
    """Which of a backend's declared deps are not importable.

    For backends with alternative deps (openai: httpx OR requests), report
    nothing missing when any alternative is present.
    """
    if cls is OpenAICompatBackend:
        return [] if cls.is_available() else ["httpx"]
    return [d for d in cls.capabilities().get("deps", [])
            if importlib.util.find_spec(d) is None]


def list_backends() -> list[dict]:
    """Describe every registered backend for the UI/availability checks."""
    out: list[dict] = []
    for name, cls in BACKENDS.items():
        caps = cls.capabilities()
        out.append({
            "name": name,
            "available": cls.is_available(),
            "generative": caps["generative"],
            "requires_text_encoder": caps["requires_text_encoder"],
            "tasks": caps["tasks"],
            "missing_deps": _missing_deps(cls),
        })
    return out


def make_backend(spec: dict) -> VLMBackend:
    """Construct a backend from a request spec, validating per backend.

    spec keys (all optional except `backend`):
        backend:       'clip' | 'transformers' | 'openai'
        model_path / image_encoder:  CLIP image encoder ONNX (clip)
        text_encoder:  CLIP text encoder ONNX (clip)
        model_id:      HF repo id / local path (transformers) or served
                       model name (openai)
        endpoint_url:  base URL of an OpenAI-compatible server (openai)
        api_key:       bearer token (openai, optional)
    """
    name = (spec.get("backend") or "clip").lower()
    cls = BACKENDS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown VLM backend '{name}'. Available: {sorted(BACKENDS)}"
        )

    if name == "clip":
        image_encoder = spec.get("model_path") or spec.get("image_encoder")
        text_encoder = spec.get("text_encoder")
        if not image_encoder:
            raise ValueError("clip backend requires model_path (image encoder ONNX)")
        if not text_encoder:
            raise ValueError("clip backend requires text_encoder (CLIP text encoder ONNX)")
        if not os.path.isfile(image_encoder):
            raise FileNotFoundError(f"Image encoder not found: {image_encoder}")
        if not os.path.isfile(text_encoder):
            raise FileNotFoundError(f"Text encoder not found: {text_encoder}")
        return CLIPBackend(image_encoder, text_encoder)

    if name == "transformers":
        model_id = spec.get("model_id") or spec.get("model_path")
        if not model_id:
            raise ValueError(
                "transformers backend requires model_id (HF repo id or local path)"
            )
        return TransformersBackend(model_id)

    if name == "openai":
        endpoint_url = spec.get("endpoint_url")
        model_id = spec.get("model_id")
        if not endpoint_url:
            raise ValueError("openai backend requires endpoint_url")
        if not model_id:
            raise ValueError("openai backend requires model_id (served model name)")
        return OpenAICompatBackend(endpoint_url, model_id, api_key=spec.get("api_key"))

    # Unreachable — registry membership checked above.
    raise ValueError(f"Unhandled backend '{name}'")


# ── Back-compat factory (CLIP only) ─────────────────────────────────────
def get_backend(model_path: str, *, text_encoder: Optional[str] = None) -> VLMBackend:
    """Legacy CLIP-only factory. Prefer make_backend() for new code.

    Kept because existing callers (and tests) construct a CLIP backend with
    just an image+text encoder pair and rely on the NotImplementedError when
    the text encoder is missing.
    """
    if not text_encoder:
        raise NotImplementedError(
            "VLM v1 only supports CLIP-style image+text encoder pairs. "
            "Provide a text_encoder ONNX path alongside the image encoder. "
            "(Generative backends: use make_backend with backend='transformers'.)"
        )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Image encoder not found: {model_path}")
    if not os.path.isfile(text_encoder):
        raise FileNotFoundError(f"Text encoder not found: {text_encoder}")
    return CLIPBackend(model_path, text_encoder)
