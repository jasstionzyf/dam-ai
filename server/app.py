"""dam-ai FastAPI server skeleton (T1): /healthz /readyz /v1/models.

Does NOT load real models (readyz is a liveness-style placeholder until the
embedder engine card lands). Registry load failure refuses startup.
"""

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from registry.schema import RegistryError, load_registry

SERVICE_NAME = "dam-ai"
SERVICE_VERSION = "0.1.0"
DEFAULT_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "registry", "models.yaml")
STARTED_AT = time.time()

try:
    _REGISTRY_PATH = os.environ.get("DAMAI_REGISTRY", DEFAULT_REGISTRY_PATH)
    REGISTRY = load_registry(_REGISTRY_PATH)
except (RegistryError, OSError) as e:  # bad yaml / bad entries / missing file all refuse startup
    raise SystemExit(f"dam-ai server refused to start: {e}") from e

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION, docs_url="/docs", redoc_url=None)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    """Process-up probe. Always 200."""
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    """Readiness probe. T1 placeholder: registry loaded => ready; real model
    readiness moves here when the embedder engine card lands."""
    return {
        "status": "ready",
        "registry_models": len(REGISTRY),
        "note": "models not loaded yet (T1 skeleton)",
        "uptime_s": round(time.time() - STARTED_AT, 3),
    }


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
