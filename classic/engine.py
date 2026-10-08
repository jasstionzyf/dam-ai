"""dam-ai classic engine: task registry + per-image pipeline.

Tasks (T6 scope = color only):
  color      -> ColorModelV2 81-dim features + OPQ/PQ opqCode + ColorPaletteModel hexColors
  colorPalette -> ColorPaletteModel hexColors only

Download/b64 of inputs and the batch envelope (/v1/classify: inputs + item
status + id passthrough, cap 64) live in classic/api.py.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass

from classic.color import (
    FEATURE_DIM,
    ColorModelV2,
    ColorPaletteModel,
    OpqQuantizer,
)

logger = logging.getLogger("damai.classic.engine")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.environ.get(
    "DAMAI_CLASSIC_MODEL_DIR", os.path.join(REPO_ROOT, "models", "color")
)

MAX_BATCH = 64

_COLOR_MODEL: ColorModelV2 | None = None
_PALETTE_MODEL: ColorPaletteModel | None = None
_QUANTIZER: OpqQuantizer | None = None


def _models_dir() -> str:
    return DEFAULT_MODEL_DIR


class EngineUnavailable(RuntimeError):
    """classic deps (faiss/skimage) missing — /v1/classify degrades to 503."""


def _color_model() -> ColorModelV2:
    global _COLOR_MODEL
    if _COLOR_MODEL is None:
        _COLOR_MODEL = ColorModelV2()
    return _COLOR_MODEL


def _palette_model() -> ColorPaletteModel:
    global _PALETTE_MODEL
    if _PALETTE_MODEL is None:
        _PALETTE_MODEL = ColorPaletteModel()
    return _PALETTE_MODEL


def _quantizer() -> OpqQuantizer:
    global _QUANTIZER
    if _QUANTIZER is None:
        _QUANTIZER = OpqQuantizer(_models_dir())
    return _QUANTIZER


@dataclass
class _Task:
    name: str
    version: str


TASKS: dict[str, _Task] = {
    "color": _Task(name="color", version="v2-81d-opq-3x10"),
    "colorPalette": _Task(name="colorPalette", version="kmeans-6"),
}


def decode_image(image_b64: str) -> bytes:
    """base64 (standard or data-url) -> raw bytes; raises ValueError on junk."""
    payload = image_b64
    if payload.startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as e:
        raise ValueError(f"invalid base64 image payload: {e}") from e


def classify_one(task: str, image_bytes: bytes) -> dict:
    """Run one image through a task pipeline. Raises on engine-level errors;
    image-level failures are the caller's per-item status concern."""
    if task == "color":
        result = _color_model().infer([image_bytes])[0]
        if result is None:
            raise EngineUnavailable("color feature extraction failed for this image")
        try:
            opq_code = _quantizer().quantize(
                __import__("numpy").asarray([result["features"]], dtype="float32")
            )
        except Exception as e:
            raise EngineUnavailable(f"OPQ quantization failed: {e}") from e
        return {
            "features": result["features"],
            "opqCode": opq_code,
            "dim": FEATURE_DIM,
        }
    if task == "colorPalette":
        return _palette_model().infer(image_bytes)
    raise KeyError(task)


def run_batch(task: str, images: list[bytes]) -> list[dict | None]:
    """Batch entry used by the API layer; preserves input order."""
    return [classify_one(task, b) for b in images]
