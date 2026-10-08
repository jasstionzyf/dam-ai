"""T3 tagger engine tests.

Unit-level: proxy app behavior against an in-process fake vLLM upstream.
Live container acceptance (compose up, real vLLM, guided decoding) is the
kanban card's acceptance report, not this file.
"""

from __future__ import annotations

import os
import pathlib
import socket
import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

# Module-level import (not inside the fixture): the test module uses
# `from __future__ import annotations`, so FastAPI must resolve the fake
# upstream's `request: Request` annotation from module globals via
# get_type_hints — a function-local import would not be visible.
from fastapi import FastAPI, Request  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Engine module reads env at import; point it at a dead port by default.
os.environ.setdefault("DAMAI_VLLM_URL", "http://127.0.0.1:9")

from tagger import engine as tagger_engine  # noqa: E402


# ---------------------------------------------------------------- fake upstream
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def fake_vllm() -> dict:
    """Minimal OpenAI-flavored upstream: echoes the request body back."""
    seen: dict = {}

    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "qwen3.5-4b", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        seen["last_body"] = body
        return {
            "id": "cmpl-fake",
            "object": "chat.completion",
            "model": body.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    from uvicorn import Config, Server

    config = Config(app, host="127.0.0.1", port=_free_port(), log_level="error")
    server = Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{config.port}"
    for _ in range(50):
        try:
            httpx.get(f"{base}/v1/models", timeout=0.5)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    yield {"base": base, "seen": seen}
    server.should_exit = True
    t.join(timeout=5)


@pytest.fixture()
def client(fake_vllm, monkeypatch):
    monkeypatch.setattr(tagger_engine, "VLLM_BASE", fake_vllm["base"])
    monkeypatch.setattr(tagger_engine, "VLLM_API", f"{fake_vllm['base']}/v1")
    from fastapi.testclient import TestClient

    with TestClient(tagger_engine.app) as c:
        yield c


# ------------------------------------------------------------------- unit tests
def test_healthz_is_process_probe(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "dam-ai-tagger"
    assert "vllm" in body


def test_readyz_200_when_backend_up(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_readyz_503_when_backend_down(monkeypatch):
    monkeypatch.setattr(tagger_engine, "VLLM_BASE", "http://127.0.0.1:9")
    monkeypatch.setattr(tagger_engine, "VLLM_API", "http://127.0.0.1:9/v1")
    from fastapi.testclient import TestClient

    with TestClient(tagger_engine.app) as c:
        r = c.get("/readyz")
    assert r.status_code == 503
    assert r.json()["status"] == "not_ready"


def test_models_aggregates_vllm_view(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data and data[0]["id"] == "qwen3.5-4b"
    assert data[0]["served_by"] == "vllm"
    assert data[0]["engine"] == "tagger"


def test_chat_passthrough_body_untouched(client, fake_vllm):
    payload = {
        "model": "qwen3.5-4b",
        "messages": [{"role": "user", "content": "describe this"}],
        "temperature": 0.2,
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "ok"
    # 100% pass-through: upstream saw the exact body, no prompt engineering
    assert fake_vllm["seen"]["last_body"] == payload


def test_chat_response_format_reaches_upstream(client, fake_vllm):
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "tags",
            "schema": {
                "type": "object",
                "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
                "required": ["tags"],
            },
        },
    }
    payload = {
        "model": "qwen3.5-4b",
        "messages": [{"role": "user", "content": "tag the image"}],
        "response_format": schema,
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    assert fake_vllm["seen"]["last_body"]["response_format"] == schema


def test_chat_missing_model_400(client):
    r = client.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400


def test_tagging_route_serves_batch(client, fake_vllm):
    """T5: /v1/tagging is the task-template batch endpoint (was a 501 stub in T3).

    This module's fake echoes content="ok", which violates the nsfw_check
    output schema on purpose — the client-side schema gate must mark the item
    output_invalid (proves the T5 endpoint validates model output).
    """
    r = client.post("/v1/tagging", json={"task": "nsfw_check", "inputs": [
        {"id": "t3-check", "image_url": "http://x/a.jpg"}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "nsfw_check"
    assert body["results"][0]["id"] == "t3-check"
    assert body["results"][0]["status"] == "error"
    assert body["results"][0]["code"] == "output_invalid"
    # guided decoding relayed: response_format reaches vLLM untouched
    sent = fake_vllm["seen"]["last_body"]
    assert sent["response_format"]["type"] == "json_schema"


# ----------------------------------------------------- deployment consistency
def test_start_vllm_launcher_flags():
    sh = (ROOT / "deploy" / "tagger" / "start-vllm.sh").read_text()
    for flag in (
        "--model",
        "--served-model-name",
        "--gpu-memory-utilization",
        "--max-model-len",
        "--max-num-seqs",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
    ):
        assert flag in sh, f"start-vllm.sh missing {flag}"
    assert "--quantization" not in sh, "bf16 policy: no quantization (FP8 hurt vision quality)"


def test_compose_service_definition():
    compose = (ROOT / "deploy" / "compose" / "tagger.yml").read_text()
    assert "8091:8091" in compose
    assert "/models:ro" in compose
    assert "HF_HUB_OFFLINE" in compose
    assert "device_ids" in compose
    assert "supervisord.conf" in compose


def test_supervisord_runs_both_processes():
    conf = (ROOT / "deploy" / "tagger" / "supervisord.conf").read_text()
    assert "start-vllm.sh" in conf
    assert "tagger.api" in conf


def test_dockerfile_bakes_code_and_entrypoint():
    df = (ROOT / "deploy" / "tagger" / "Dockerfile").read_text()
    assert "COPY tagger /app/tagger" in df
    assert "supervisord.conf" in df
    assert "EXPOSE 8091" in df
