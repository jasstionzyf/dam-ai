"""dam-ai FastAPI server: T1 skeleton + T2 embedder engine.

healthz/readyz/v1/models from T1; /v1/embeddings lands with the embedder
manager (T2). Registry load failure refuses startup. The server imports fine
without torch; embedder models load in the background and readiness reflects
their state.
"""

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from registry.schema import RegistryError, load_registry
from server.embeddings import embeddings as embeddings_handler
from server.embeddings import openai_error

SERVICE_NAME = "dam-ai"
SERVICE_VERSION = "0.2.0"
DEFAULT_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "registry", "models.yaml")
STARTED_AT = time.time()

try:
    _REGISTRY_PATH = os.environ.get("DAMAI_REGISTRY", DEFAULT_REGISTRY_PATH)
    REGISTRY = load_registry(_REGISTRY_PATH)
except (RegistryError, OSError) as e:  # bad yaml / bad entries / missing file all refuse startup
    raise SystemExit(f"dam-ai server refused to start: {e}") from e

# Embedder engine is optional at import time (API tests / CI run without torch).
# When deps are missing, /v1/embeddings reports model_error instead of crashing.
# Model loading starts on the uvicorn startup event — importing this module
# (e.g. unit tests) must NOT spawn GPU loads.
_manager = None
try:
    from embedder.manager import init_manager

    _manager = init_manager()
except ImportError as e:
    _IMPORT_ERROR = str(e)
except Exception as e:  # registry ok but manager init failed — refuse, it is a deployment bug
    raise SystemExit(f"dam-ai embedder manager failed to init: {e}") from e
else:
    _IMPORT_ERROR = None

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION, docs_url="/docs", redoc_url=None)


@app.on_event("startup")
def _start_model_loading() -> None:
    if _manager is not None:
        _manager.start()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTPException with a dict detail (OpenAI envelope) passes through verbatim."""
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    """Process-up probe. Always 200."""
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@app.get("/readyz")
def readyz() -> JSONResponse:
    """Readiness: registry loaded (startup gate) + per-model embedder states."""
    if _manager is None:
        body: dict[str, Any] = {
            "status": "degraded",
            "registry_models": len(REGISTRY),
            "note": f"embedder unavailable: {_IMPORT_ERROR}",
        }
        return JSONResponse(status_code=200, content=body)
    snap = _manager.snapshot()
    body = {
        "status": "ready" if snap["ready"] else "loading",
        "registry_models": len(REGISTRY),
        "models": snap["models"],
        "uptime_s": snap["uptime_s"],
    }
    return JSONResponse(status_code=200, content=body)


@app.get("/v1/models")
def list_models() -> JSONResponse:
    """OpenAI-style model list, extended with dam_ai metadata."""
    data = []
    for spec in REGISTRY.values():
        data.append({
            "id": spec.name,
            "object": "model",
            "owned_by": SERVICE_NAME,
            "dam_ai": spec.dam_ai_meta,
        })
    return JSONResponse({"object": "list", "data": data})


@app.get("/v1/models/{model_id}")
def get_model(model_id: str) -> dict[str, Any]:
    spec = REGISTRY.get(model_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"unknown model: {model_id}")
    return {
        "id": spec.name,
        "object": "model",
        "owned_by": SERVICE_NAME,
        "dam_ai": {**spec.dam_ai_meta, "path": spec.path, "preprocess": spec.preprocess.to_dict()},
    }


@app.post("/v1/embeddings")
async def v1_embeddings(request: Request) -> JSONResponse:
    if _manager is None:
        raise openai_error(503, f"embedder engine unavailable: {_IMPORT_ERROR}",
                           "model_error", "model_not_ready")
    body = await request.json()
    return JSONResponse(await embeddings_handler(body))
