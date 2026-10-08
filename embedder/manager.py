"""Model manager: background loading, readiness state, embed dispatch.

Models load one-at-a-time in a background thread at startup; a failed model
records its error and is skipped (server stays up, reports degraded, other
models still serve). Embed requests while loading return 503 with retry hint.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from embedder.engine import LOADERS, LoadableModel, decode_image_url
from registry.schema import ModelSpec, load_registry

logger = logging.getLogger("dam-ai.embedder.manager")


class ModelNotReadyError(RuntimeError):
    def __init__(self, name: str, state: str, error: str | None):
        self.name, self.state, self.error = name, state, error
        super().__init__(f"model {name} not ready (state={state})")


class Manager:
    def __init__(self, registry_path: str, models_root: str | None = None):
        self.registry: dict[str, ModelSpec] = load_registry(registry_path)
        self.models_root = models_root or os.environ.get("DAMAI_MODELS_ROOT") or None
        self._models: dict[str, LoadableModel] = {}
        self._states: dict[str, str] = {n: "loading" for n in self.registry}
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()
        self._sem = threading.BoundedSemaphore(2)  # max concurrent forwards (OOM guard)
        self._executor = ThreadPoolExecutor(max_workers=1)  # sequential model loads
        self._started = False
        self._t0 = time.time()

    # ---------- lifecycle ----------

    def start(self) -> None:
        # Idempotent: init_manager() already kicks loading at import time and
        # the uvicorn startup event calls start() again — a second submission
        # would load every model twice and OOM the GPU (seen live on gpu7).
        if self._started:
            return
        self._started = True
        for name in self.registry:
            self._executor.submit(self._load_one, name)

    def _load_one(self, name: str) -> None:
        spec = self.registry[name]
        cls = LOADERS[spec.loader]
        model = cls(spec, self.models_root)
        t0 = time.time()
        try:
            model.load()
            self._models[name] = model
            self._states[name] = "ready"
            logger.info("model %s ready in %.1fs", name, time.time() - t0)
        except Exception as e:  # noqa: BLE001 — one bad model must not sink the service
            self._states[name] = "error"
            self._errors[name] = f"{type(e).__name__}: {e}"
            logger.exception("model %s failed to load: %s", name, self._errors[name])
            self._release_gpu_cache()

    @staticmethod
    def _release_gpu_cache() -> None:
        """After a failed load, free cached CUDA blocks so later models can load."""
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — cache release is best-effort
            pass

    # ---------- introspection ----------

    def ready(self) -> bool:
        return all(s == "ready" for s in self._states.values())

    def snapshot(self) -> dict[str, Any]:
        return {
            "models": {n: {"state": self._states[n], "error": self._errors.get(n)}
                       for n in self.registry},
            "ready": self.ready(),
            "uptime_s": round(time.time() - self._t0, 3),
        }

    # ---------- embedding ----------

    def parse_and_validate(self, model: str, raw_inputs: list) -> list[dict[str, Any]]:
        """Normalize OpenAI content-part input items into loader input dicts.
        Raises ValueError with an OpenAI-style message on bad input."""
        spec = self.registry.get(model)
        if spec is None:
            raise KeyError(model)
        if "text" not in spec.modalities and "image" not in spec.modalities:
            raise ValueError(f"model {model} supports no known modalities")
        out: list[dict[str, Any]] = []
        for i, item in enumerate(raw_inputs):
            if isinstance(item, str):
                if "text" not in spec.modalities:
                    raise ValueError(
                        f"inputs[{i}]: model {model} does not support text "
                        f"(modalities: {spec.modalities})")
                if not item.strip():
                    raise ValueError(f"inputs[{i}]: empty text")
                out.append({"text": item})
                continue
            if isinstance(item, dict) and item.get("type") == "image_url":
                if "image" not in spec.modalities:
                    raise ValueError(
                        f"inputs[{i}]: model {spec.name} is image-only — text not supported")
                url = (item.get("image_url") or {}).get("url", "")
                payload, scheme = decode_image_url(url)
                if scheme is None:
                    raise ValueError(
                        f"inputs[{i}]: image url must be http(s) or a data:image/...;base64 URL")
                if scheme == "data":
                    out.append({"image_bytes": payload})
                else:  # http(s) — loaders fetch via PIL/qwen_vl_utils
                    out.append({"image_url": url})
                continue
            raise ValueError(
                f"inputs[{i}]: unsupported input type {type(item).__name__}; expected "
                f"string or {{type: 'image_url', image_url: {{url}}}}")
        return out

    def embed(self, model: str, inputs: list[dict[str, Any]], dtype: str) -> list[list[float]]:
        lm = self._models.get(model)
        state = self._states.get(model)
        if lm is None or state != "ready":
            raise ModelNotReadyError(model, state or "unknown", self._errors.get(model))
        with self._sem:
            return lm.embed(inputs, dtype)


manager: Manager | None = None


def init_manager() -> Manager:
    global manager
    if manager is None:
        registry = os.environ.get("DAMAI_REGISTRY",
                                  os.path.join(os.path.dirname(__file__), "..",
                                               "registry", "models.yaml"))
        manager = Manager(registry)
        manager.start()
    return manager
