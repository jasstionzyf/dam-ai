"""dam-ai classic /v1/classify: batch color classification over HTTP.

Envelope: {"task": str, "inputs": [{"id": ..., "image_b64" | "image_url"}]}
  -> {"results": [{"id": ..., "status": "ok"|"error", "result": {...}|None,
       "error": str|None}]}
Cap 64 inputs (413 above). A bad image (undecodable / unparsable / engine
failure) is isolated to its own item; the batch continues.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:  # runtime import comes from fastapi inside classify
    from fastapi import Request
    from fastapi.responses import JSONResponse

SERVICE_NAME = "dam-ai-classic"
SERVICE_VERSION = "0.1.0"

logger = logging.getLogger("damai.classic.api")


def _engine_missing(*_a, **_kw):
    raise RuntimeError("classic engine unavailable (faiss/skimage import failed)")


try:  # engine import optional: CI/pytest runs without faiss/skimage
    from classic.engine import MAX_BATCH, TASKS, classify_one, decode_image
except ImportError as _e:  # pragma: no cover
    MAX_BATCH = 64
    TASKS = {}
    classify_one = _engine_missing  # type: ignore[assignment]
    decode_image = _engine_missing  # type: ignore[assignment]
    _ENGINE_IMPORT_ERROR = str(_e)
else:
    _ENGINE_IMPORT_ERROR = None


def _fetch_url(url: str) -> bytes:
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


def _extract_image_bytes(item: dict) -> bytes:
    if item.get("image_b64"):
        return decode_image(item["image_b64"])
    if item.get("image_url"):
        return _fetch_url(item["image_url"])
    raise ValueError("input needs image_b64 or image_url")


async def classify(request: "Request") -> "JSONResponse":
    """POST /v1/classify (see module docstring for the envelope)."""
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse

    if _ENGINE_IMPORT_ERROR:
        # Degraded mode (Phase 4): dict detail with "error" key passes verbatim
        # through the app-level handler -> OpenAI-style 503 model_unavailable.
        raise HTTPException(503, detail={"error": {
            "message": f"classic engine unavailable: {_ENGINE_IMPORT_ERROR}",
            "type": "model_error", "code": "model_unavailable"}})

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "invalid JSON body") from None

    task = body.get("task")
    if task not in TASKS:
        raise HTTPException(400, f"unknown task {task!r}; valid: {sorted(TASKS)}")

    inputs = body.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise HTTPException(400, "inputs must be a non-empty array")
    if len(inputs) > MAX_BATCH:
        raise HTTPException(413, f"inputs cap is {MAX_BATCH}, got {len(inputs)}")

    results = []
    for idx, item in enumerate(inputs):
        item_id = item.get("id", idx)
        started = time.perf_counter()
        try:
            image_bytes = _extract_image_bytes(item)
            result = classify_one(task, image_bytes)
            results.append({"id": item_id, "status": "ok", "result": result, "error": None})
        except Exception as e:  # noqa: BLE001 - one bad input must not kill the batch
            logger.warning("classify item %s failed: %s", item_id, e)
            results.append({"id": item_id, "status": "error", "result": None, "error": str(e)})
        logger.debug("item %s took %.1f ms", item_id, 1000 * (time.perf_counter() - started))

    return JSONResponse({"results": results})
