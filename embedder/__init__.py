"""dam-ai embedder engine (transformers family).

Three loaders cover the on-disk model formats:
- sentence_transformers: ST layout (modules.json) with a vendored wrapper class
  (qwen3-vl-embedding-2b).
- transformers: plain HF transformers layout via AutoModel (siglip-so400m,
  dinov2-base).
- open_clip: open_clip checkpoint files (clip/viT-L-14.pt).
"""

from __future__ import annotations

from registry.schema import ModelSpec


def model_root(spec: ModelSpec, models_root: str | None = None) -> str:
    """Resolve a registry path (/models/<name>) against an optional host root.

    In containers, /models is the mount point and models_root is None. On the
    host (tests/dev), DAMAI_MODELS_ROOT remaps /models -> the real model dir.
    """
    if models_root and spec.path.startswith("/models/"):
        return models_root.rstrip("/") + spec.path[len("/models"):]
    return spec.path


def is_ready() -> bool:
    try:
        from embedder.manager import manager  # noqa: F401
    except Exception:
        return False
    from embedder.manager import manager

    return manager.ready
