"""Embedder loadable interface + the three on-disk format loaders."""

from __future__ import annotations

import base64
import io
import logging
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from embedder import vendors
from embedder.torch_utils import pick_device, pick_dtype
from registry.schema import ModelSpec

logger = logging.getLogger("dam-ai.embedder")

_DATA_URL_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,(.*)$", re.DOTALL)


def decode_image_url(url: str) -> tuple[bytes | None, str | None]:
    """Split a data URL into bytes. http(s) URLs are passed through untouched
    (network fetch is delegated to the qwen wrapper); other schemes are
    rejected. Returns (payload, scheme)."""
    if url.startswith(("http://", "https://")):
        return None, "http"
    m = _DATA_URL_RE.match(url)
    if m:
        return base64.b64decode(m.group(1)), "data"
    return None, None


class LoadableModel(ABC):
    """One loaded model: lazy init, embed one batch, report readiness."""

    def __init__(self, spec: ModelSpec, models_root: str | None):
        self.spec = spec
        self.models_root = models_root
        self.backend: Any = None
        self.error: str | None = None

    @property
    def path(self) -> str:
        from embedder import model_root

        return model_root(self.spec, self.models_root)

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def embed(self, inputs: list[dict[str, Any]], dtype: str) -> list[list[float]]: ...

    @property
    def ready(self) -> bool:
        return self.backend is not None


class Qwen3VLEmbeddingModel(LoadableModel):
    """sentence_transformers-layout Qwen3-VL embedding model driven by its
    vendored wrapper class (scripts/qwen3_vl_embedding.py inside the model dir).
    Handles text, image (PIL/bytes), and http URLs natively via qwen_vl_utils."""

    def load(self) -> None:
        torch = vendors.torch_mod()
        cfg = self.spec.loader_config
        wrapper_ref = cfg.get("wrapper", "qwen3_vl_embedding.Qwen3VLEmbedder")
        module_name, _, class_name = wrapper_ref.rpartition(".")
        wrapper_dir = Path(self.path) / "scripts"
        if not wrapper_dir.is_dir():
            raise FileNotFoundError(f"wrapper scripts dir missing: {wrapper_dir}")
        if str(wrapper_dir) not in sys.path:
            sys.path.insert(0, str(wrapper_dir))
        module = __import__(module_name, fromlist=[class_name])
        embedder_cls = getattr(module, class_name)

        dtype = torch.float16 if cfg.get("dtype", "float16") == "float16" else torch.bfloat16
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if cfg.get("max_pixels"):
            kwargs["max_pixels"] = int(cfg["max_pixels"])
        # wrapper auto-selects cuda when torch.cuda.is_available()
        self.backend = embedder_cls(self.path, **kwargs)
        logger.info("qwen3-vl-embedding loaded on %s", self.backend.model.device)

    def embed(self, inputs: list[dict[str, Any]], dtype: str) -> list[list[float]]:
        Image = vendors.Image_cls()
        # wrapper format: {"text": str} / {"image": PIL | http(s)-url}
        qin: list[dict[str, Any]] = []
        for i in inputs:
            if "image_bytes" in i:
                qin.append({"image": Image.open(io.BytesIO(i["image_bytes"])).convert("RGB")})
            elif "image_url" in i:
                qin.append({"image": i["image_url"]})  # wrapper passes http(s) through
            else:
                qin.append({"text": i["text"]})
        emb = self.backend.process(qin, normalize=self.spec.normalized)
        vec = emb.float().cpu().numpy()
        return _maybe_half(vec, dtype)


class TransformersAutoModel(LoadableModel):
    """Plain HF transformers layout (AutoModel + AutoProcessor/AutoTokenizer):
    siglip-so400m (dual encoder) and dinov2-base (image-only CLS)."""

    def load(self) -> None:
        tf = vendors.transformers_mod()
        device = pick_device()
        dtype = pick_dtype(device)
        spec_has_text = "text" in self.spec.modalities
        self.backend = {
            "model": tf.AutoModel.from_pretrained(self.path, torch_dtype=dtype).to(device).eval(),
            "processor": tf.AutoProcessor.from_pretrained(self.path),
            # image-only checkpoints (dinov2) ship no tokenizer files
            "tokenizer": (tf.AutoTokenizer.from_pretrained(self.path)
                          if spec_has_text else None),
            "device": device,
        }
        logger.info("%s loaded on %s (%s)", self.spec.name, device, dtype)

    @staticmethod
    def _to_pil(payload: bytes):
        Image = vendors.Image_cls()
        return Image.open(io.BytesIO(payload)).convert("RGB")

    def embed(self, inputs: list[dict[str, Any]], dtype: str) -> list[list[float]]:
        np = vendors.np_mod()
        torch = vendors.torch_mod()
        model = self.backend["model"]
        device = self.backend["device"]
        texts = [i["text"] for i in inputs if "text" in i]
        images = [self._to_pil(i["image_bytes"]) for i in inputs if "image_bytes" in i]
        out: list[list[float]] = []

        def _collect(feats_or_out, pick: str) -> None:
            # transformers>=5 returns ModelOutput — pull the features tensor off it
            feats = feats_or_out
            if not hasattr(feats, "norm"):  # ModelOutput, not a tensor
                feats = getattr(feats, pick, None)
                if feats is None:
                    feats = feats_or_out.pooler_output
            out.extend(_norm_and_collect(feats, self.spec.normalized, np, torch))

        with torch.no_grad():
            if texts:
                tk = self.backend["tokenizer"](texts, padding=True, truncation=True,
                                               return_tensors="pt").to(device)
                if hasattr(model, "get_text_features"):
                    _collect(model.get_text_features(**tk), "pooler_output")
                else:
                    _collect(model(**tk), "pooler_output")
            if images:
                pp = self.backend["processor"](images=images, return_tensors="pt").to(device)
                if hasattr(model, "get_image_features"):
                    _collect(model.get_image_features(**pp), "pooler_output")
                else:
                    _collect(model(**pp).last_hidden_state[:, 0], "pooler_output")
        return _maybe_half(np.asarray(out), dtype)


class OpenClipModel(LoadableModel):
    """open_clip checkpoint (weights filename inside path): clip-vit-l14."""

    def load(self) -> None:
        oc = vendors.open_clip_mod()
        device = pick_device()
        weights = self.spec.weights
        if not weights:
            raise ValueError(f"{self.spec.name}: open_clip loader requires 'weights'")
        weight_path = Path(self.path) / weights
        # open_clip >=0.9 forwards weights_only to torch.load, whose PyTorch>=2.6
        # default (weights_only=True) rejects TorchScript .pt archives like
        # ViT-L-14.pt. Our weights are a fixed, admin-provisioned file — trusted.
        model, _, preprocess = oc.create_model_and_transforms(
            "ViT-L-14", pretrained=str(weight_path), device=device, weights_only=False)
        self.backend = {"model": model.eval(), "preprocess": preprocess, "tokenizer":
                        oc.get_tokenizer("ViT-L-14"), "device": device}
        logger.info("clip-vit-l14 (open_clip) loaded on %s", device)

    def embed(self, inputs: list[dict[str, Any]], dtype: str) -> list[list[float]]:
        np = vendors.np_mod()
        torch = vendors.torch_mod()
        Image = vendors.Image_cls()
        model = self.backend["model"]
        device = self.backend["device"]
        texts = [i["text"] for i in inputs if "text" in i]
        images = [self.backend["preprocess"](
            Image.open(io.BytesIO(i["image_bytes"])).convert("RGB"))
            for i in inputs if "image_bytes" in i]
        out: list[list[float]] = []
        with torch.no_grad():
            if texts:
                tk = self.backend["tokenizer"](texts).to(device)
                out.extend(
                    _norm_and_collect(model.encode_text(tk), self.spec.normalized, np, torch))
            if images:
                px = torch.stack(images).to(device)
                out.extend(
                    _norm_and_collect(model.encode_image(px), self.spec.normalized, np, torch))
        return _maybe_half(np.asarray(out), dtype)


def _norm_and_collect(feats, normalized: bool, np, torch) -> list[list[float]]:
    if normalized:
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    return feats.float().cpu().numpy().tolist()


def _maybe_half(vec: "Any", dtype: str) -> list[list[float]]:
    if dtype == "float16":
        np = vendors.np_mod()
        return np.asarray(vec, dtype=np.float16).astype(np.float32).tolist()
    np = vendors.np_mod()
    return np.asarray(vec, dtype=np.float32).tolist()


LOADERS = {
    "sentence_transformers": Qwen3VLEmbeddingModel,
    "transformers": TransformersAutoModel,
    "open_clip": OpenClipModel,
}
