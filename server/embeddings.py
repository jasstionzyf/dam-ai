"""OpenAI-compatible /v1/embeddings: parse, validate, dispatch to the manager.

Errors follow the OpenAI error envelope: {error: {message, type, code}}.
The manager is accessed via module attribute (not import-time binding) so
init_manager() reassignment is always visible.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import HTTPException

from embedder import manager as manager_module


def openai_error(status: int, message: str, err_type: str,
                 code: str | None = None) -> HTTPException:
    return HTTPException(status_code=status, detail={
        "error": {"message": message, "type": err_type, "code": code}})


def _float_list_to_base64(vec: list[float]) -> str:
    """OpenAI base64 encoding_format: little-endian float32 packed and base64'd."""
    import base64
    import struct

    return base64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode()


def _manager():
    return manager_module.manager


async def embeddings(request_body: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    model = request_body.get("model")
    raw_inputs = request_body.get("input")
    encoding_format = request_body.get("encoding_format", "float")

    if not isinstance(model, str) or not model:
        raise openai_error(400, "'model' is required and must be a string", "invalid_request_error",
                           "invalid_model")
    # OpenAI SDK (>=1.x) defaults to encoding_format=base64 — support it alongside
    # float; float16 is a dam-ai extension (compact transport).
    if encoding_format not in ("float", "float16", "base64"):
        raise openai_error(400, f"unsupported encoding_format: {encoding_format}",
                           "invalid_request_error", "invalid_encoding_format")
    if raw_inputs is None:
        raise openai_error(400, "'input' is required", "invalid_request_error", "invalid_input")
    # OpenAI-compatible: string -> [string]; list of strings/content-parts otherwise
    if isinstance(raw_inputs, str):
        raw_inputs = [raw_inputs]
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise openai_error(400, "'input' must be a non-empty string or array",
                           "invalid_request_error", "invalid_input")
    if len(raw_inputs) > 64:
        raise openai_error(400, f"inputs exceeds hard limit: {len(raw_inputs)} > 64 items "
                                f"(caller must shard)", "invalid_request_error", "batch_too_large")

    try:
        parsed = _manager().parse_and_validate(model, raw_inputs)
    except KeyError:
        raise openai_error(404, f"unknown model: {model}", "invalid_request_error",
                           "model_not_found") from None
    except ValueError as e:
        raise openai_error(400, str(e), "invalid_request_error", "invalid_input") from None

    try:
        vectors = _manager().embed(model, parsed, encoding_format)
    except ModelNotReadyError as e:
        # Degraded mode (T9): backend unavailable (still loading, or failed to
        # load — missing engine deps / bad weights) -> model_unavailable, never
        # a crash. type distinguishes retryable (loading) from permanent (error).
        raise openai_error(
            503, f"model {model} is not ready (state={e.state}"
                 f"{f', error: {e.error}' if e.error else ''}); retry later",
            "service_unavailable" if e.state == "loading" else "model_error",
            "model_unavailable") from None
    except RuntimeError as e:  # loader-level failure (missing deps, etc.)
        raise openai_error(503, str(e), "model_error", "model_unavailable") from None

    dims = _manager().registry[model].dims
    data = []
    for i, v in enumerate(vectors):
        item = {"object": "embedding", "index": i}
        item["embedding"] = _float_list_to_base64(v) if encoding_format == "base64" else v
        data.append(item)
    return {
        "object": "list",
        "data": data,
        "model": model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},  # not tracked for embeddings
        "dam_ai": {"dims": dims, "count": len(vectors),
                   "latency_ms": round((time.time() - t0) * 1000, 1)},
    }


from embedder.manager import ModelNotReadyError  # noqa: E402  (used in except above)
