"""uvicorn entrypoint for a standalone classic service (port 8092, no GPU).

    python -m classic.__main__
The unified server (server.app) also exposes /v1/classify when classic deps
are present; this entrypoint runs classic alone.
"""

from __future__ import annotations

import os

import uvicorn

from server.app import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("DAMAI_HOST", "0.0.0.0"),
        port=int(os.environ.get("DAMAI_CLASSIC_PORT", "8092")),
        log_level=os.environ.get("DAMAI_LOG_LEVEL", "info"),
        access_log=False,
    )
