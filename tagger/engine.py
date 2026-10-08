"""dam-ai tagger engine.

Standalone FastAPI app that fronts a co-located vLLM OpenAI server
(127.0.0.1:<port>, launched alongside by deploy/compose/tagger.yml).

Scope (T3/T5): /v1/chat/completions pass-through (100% OpenAI request/response,
guided decoding via response_format), /v1/models aggregated view,
/healthz /readyz probing the vLLM backend, and the /v1/tagging batch
task-template endpoint (tagger/tagging.py, mounted at import).
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

SERVICE_NAME = "dam-ai-tagger"
SERVICE_VERSION = "0.1.0"
STARTED_AT = time.time()

VLLM_BASE = os.environ.get("DAMAI_VLLM_URL", "http://127.0.0.1:8101").rstrip("/")
VLLM_API = f"{VLLM_BASE}/v1"
# /healthz must stay a cheap process-up probe (liveness) — do not wire it to vLLM.
READY_TIMEOUT_S = float(os.environ.get("DAMAI_VLLM_TIMEOUT", "3.0"))

_CLIENT: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    return _CLIENT


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _CLIENT
    # Always (re)create on startup: a previous shutdown closed the module-level
    # client (tests open/close the app repeatedly; uvicorn does it once).
    _CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    yield
    if _CLIENT is not None:
        await _CLIENT.aclose()
        _CLIENT = None


app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    docs_url="/docs",
    redoc_url=None,
    lifespan=_lifespan,
)


# ---------------------------------------------------------------- health/probes
@app.get("/healthz")
def healthz() -> dict[str, Any]:
    """Process-up probe. Always 200."""
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "vllm": VLLM_BASE,
    }


@app.get("/readyz")
def readyz() -> JSONResponse:
    """Ready when the co-located vLLM server answers /v1/models."""
    try:
        r = httpx.get(f"{VLLM_API}/models", timeout=READY_TIMEOUT_S)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001 - probe boundary, report and degrade
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "service": SERVICE_NAME, "vllm_error": str(e)},
        )
    return JSONResponse(
        content={"status": "ready", "service": SERVICE_NAME, "vllm": VLLM_BASE}
    )


# ------------------------------------------------------------------- /v1/models
@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    """Aggregated view: vLLM's own served models, annotated with engine info.

    vLLM-served model ids are the addressing key for chat/completions, so the
    id list must come from vLLM, not from a static registry file.
    """
    try:
        r = httpx.get(f"{VLLM_API}/models", timeout=READY_TIMEOUT_S)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"vLLM backend unreachable: {e}") from e
    data = payload.get("data", [])
    for m in data:
        m.setdefault("owned_by", "dam-ai")
        m["served_by"] = "vllm"
        m["engine"] = "tagger"
    return {"object": "list", "data": data}


# ---------------------------------------------------------- /v1/chat/completions
def _safe_json(raw: bytes) -> dict[str, Any]:
    import json

    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return {"detail": raw.decode("utf-8", "replace")}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    """100% OpenAI pass-through to the vLLM backend (guided decoding included).

    response_format={type: json_schema, ...} reaches vLLM untouched, where
    guided decoding enforces the schema. stream=true is relayed as an SSE
    byte stream.
    """
    try:
        body: dict[str, Any] = await request.json()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {e}") from e
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    if "model" not in body:
        raise HTTPException(status_code=400, detail="missing required field: model")

    headers: dict[str, str] = {}
    auth = request.headers.get("authorization")
    if auth:
        headers["Authorization"] = auth

    client = _get_client()
    req = client.build_request("POST", f"{VLLM_API}/chat/completions", json=body, headers=headers)
    try:
        resp = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"vLLM backend error: {e}") from e

    if resp.status_code != 200:
        content = await resp.aread()
        await resp.aclose()
        return JSONResponse(status_code=resp.status_code, content=_safe_json(content))

    if body.get("stream"):
        return StreamingResponse(
            resp.aiter_raw(),
            media_type=resp.headers.get("content-type", "text/event-stream"),
            background=BackgroundTask(resp.aclose),
        )
    content = await resp.aread()
    await resp.aclose()
    return JSONResponse(status_code=resp.status_code, content=_safe_json(content))


# ----------------------------------------------------------- /v1/tagging
try:
    from tagger.tagging import tagging as _tagging_handler
except ImportError as e:  # pragma: no cover - jsonschema ships with the vllm env
    _TAGGING_IMPORT_ERROR = str(e)

    @app.post("/v1/tagging")
    def tagging_unavailable() -> JSONResponse:
        """Degrade instead of crash when the tagging deps are missing."""
        return JSONResponse(
            status_code=503,
            content={"error": {
                "message": f"/v1/tagging unavailable: {_TAGGING_IMPORT_ERROR}",
                "type": "model_error", "code": "model_unavailable"}},
        )
else:
    app.add_api_route("/v1/tagging", _tagging_handler, methods=["POST"],
                      name="tagging")
