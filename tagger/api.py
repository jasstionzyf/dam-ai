"""uvicorn entrypoint for the dam-ai tagger proxy app.

Kept as a module so the container command stays declarative:
    python -m tagger.api
"""

from __future__ import annotations

import os

import uvicorn

from tagger.engine import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("DAMAI_PROXY_HOST", "0.0.0.0"),
        port=int(os.environ.get("DAMAI_PROXY_PORT", "8091")),
        log_level=os.environ.get("DAMAI_LOG_LEVEL", "info"),
        access_log=False,
    )
