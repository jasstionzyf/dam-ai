"""T5 /v1/tagging tests.

Unit-level against the same fake vLLM upstream as test_tagger.py: template
channel, inline channel, item isolation, retry, batch limits, locked params.
Live acceptance (real vLLM on gpu7) is the kanban card report, not this file.
"""

from __future__ import annotations

import json
import os
import pathlib
import socket
import sys
import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Module-level import: the fake upstream's route annotations must resolve via
# get_type_hints from module globals (same constraint as test_tagger.py).
from fastapi import FastAPI, Request, Response  # noqa: E402

SCHEMA_COLORS = {
    "type": "object",
    "properties": {"colors": {"type": "array", "items": {"type": "string"}}},
    "required": ["colors"],
    "additionalProperties": False,
}

NSFW_OK = json.dumps({"is_nsfw": False, "confidence": 0.1, "reason": "clean"})

# Engine reads env at import; point it at a dead port by default.
os.environ.setdefault("DAMAI_VLLM_URL", "http://127.0.0.1:9")

from tagger import engine as tagger_engine  # noqa: E402
from tagger import tagging as tagging_module  # noqa: E402


# ---------------------------------------------------------------- fake upstream
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _completion(content: str) -> dict:
    return {"id": "cmpl-fake", "object": "chat.completion", "model": "qwen3.5-4b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}]}


def _valid_instance(schema: dict) -> object:
    """Fabricate a minimal json-schema-valid instance for the fake upstream."""
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    t = schema.get("type")
    if t == "object" or "properties" in schema:
        out = {}
        for name, sub in (schema.get("properties") or {}).items():
            if name in schema.get("required", []):
                out[name] = _valid_instance(sub)
        return out
    if t == "array":
        return [_valid_instance(schema.get("items", {}))]
    if t == "boolean":
        return False
    if t == "number" or t == "integer":
        return schema.get("minimum", 0)
    return schema.get("const", "ok")  # string default


def _resp(status: int, payload: dict) -> Response:
    return Response(content=json.dumps(payload), status_code=status,
                    media_type="application/json")


@pytest.fixture(scope="module")
def fake_vllm() -> dict:
    """OpenAI-flavored upstream: echo + controllable per-call behavior."""
    seen: dict = {}
    state: dict = {}  # test-writable behavior hooks

    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "qwen3.5-4b", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):  # noqa: ANN201 - Response|dict union
        body = await request.json()
        seen["last_body"] = body
        seen.setdefault("bodies", []).append(body)
        hook = state.get("hook")
        if hook is not None:
            return hook(body)
        # schema-faithful fake: produce a minimal instance of the requested
        # output schema so the client-side validation passes for every template
        schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
        return _completion(json.dumps(_valid_instance(schema)))

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
    yield {"base": base, "seen": seen, "state": state}
    server.should_exit = True
    t.join(timeout=5)


@pytest.fixture()
def client(fake_vllm, monkeypatch):
    monkeypatch.setattr(tagger_engine, "VLLM_BASE", fake_vllm["base"])
    monkeypatch.setattr(tagger_engine, "VLLM_API", f"{fake_vllm['base']}/v1")
    tagging_module.ITEM_RETRIES = 0  # deterministic retry counts in unit tests
    from fastapi.testclient import TestClient

    with TestClient(tagger_engine.app) as c:
        yield c


# ------------------------------------------------------------------- template channel
def test_single_image_ok(client, fake_vllm):
    r = client.post("/v1/tagging", json={"task": "nsfw_check", "inputs": [
        {"id": "img-42", "image_url": "http://x/a.jpg"}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "nsfw_check"
    assert body["task_version"] == 1
    assert body["model"] == "qwen3.5-4b"  # template model_default
    res = body["results"][0]
    assert res["status"] == "ok"
    assert res["id"] == "img-42"  # id passes through verbatim
    assert res["output"] == {"is_nsfw": False, "confidence": 0, "reason": "explicit"}
    assert res["task_version"] == 1
    # guided decoding: the schema went to vLLM untouched
    sent = fake_vllm["seen"]["last_body"]
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["name"] == "nsfw_check"


def test_task_params_render_into_prompt(client, fake_vllm):
    r = client.post("/v1/tagging", json={"task": "image_caption_metadata",
                                         "task_params": {"language": "zh", "max_keywords": 5},
                                         "inputs": [{"id": 7, "image_url": "http://x/a.jpg"}]})
    assert r.status_code == 200, r.text
    sent = fake_vllm["seen"]["last_body"]
    assert '"zh"' in sent["messages"][0]["content"][0]["text"]
    assert "5" in sent["messages"][0]["content"][0]["text"]
    # numeric id passes through verbatim (no string coercion)
    assert r.json()["results"][0]["id"] == 7


def test_multi_image_item(client, fake_vllm):
    # translate task templates allow up to 16 images per item (image channel)
    r = client.post("/v1/tagging", json={"task": "translate",
                                         "task_params": {"text": "hello world",
                                                         "target_language": "zh"},
                                         "inputs": [
        {"id": "m1", "images": ["http://x/a.jpg", "http://x/b.jpg"]}]})
    assert r.status_code == 200, r.text
    sent = fake_vllm["seen"]["last_body"]
    parts = sent["messages"][0]["content"]
    assert sum(1 for p in parts if p["type"] == "image_url") == 2


# ---------------------------------------------------------------- request-level 400s
def test_unknown_task_400(client):
    r = client.post("/v1/tagging", json={"task": "nope",
                                         "inputs": [{"id": 1, "image_url": "u"}]})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unknown_task"


def test_model_not_allowed_400(client):
    r = client.post("/v1/tagging", json={"task": "nsfw_check", "model": "clip-vit-l14",
                                         "inputs": [{"id": 1, "image_url": "u"}]})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "model_not_allowed"


def test_65_inputs_400(client):
    inputs = [{"id": i, "image_url": "http://x/a.jpg"} for i in range(65)]
    r = client.post("/v1/tagging", json={"task": "nsfw_check", "inputs": inputs})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "batch_too_large"


def test_64_inputs_accepted(client):
    inputs = [{"id": i, "image_url": "http://x/a.jpg"} for i in range(64)]
    r = client.post("/v1/tagging", json={"task": "nsfw_check", "inputs": inputs})
    assert r.status_code == 200, r.text
    assert len(r.json()["results"]) == 64
    assert all(x["status"] == "ok" for x in r.json()["results"])


def test_images_count_violation_400(client):
    # nsfw_check allows exactly 1 image per item
    r = client.post("/v1/tagging", json={"task": "nsfw_check", "inputs": [
        {"id": 1, "images": ["http://x/a.jpg", "http://x/b.jpg"]}]})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "images_count_violation"


def test_missing_task_and_inline_400(client):
    r = client.post("/v1/tagging", json={"inputs": [{"id": 1, "image_url": "u"}]})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "missing_task"


def test_task_and_inline_conflict_400(client):
    r = client.post("/v1/tagging", json={"task": "nsfw_check", "prompt": "p",
                                         "schema": SCHEMA_COLORS,
                                         "inputs": [{"id": 1, "image_url": "u"}]})
    assert r.status_code == 400


def test_task_params_unknown_key_400(client):
    r = client.post("/v1/tagging", json={"task": "nsfw_check",
                                         "task_params": {"bogus": 1},
                                         "inputs": [{"id": 1, "image_url": "u"}]})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_task_params"


# ------------------------------------------------------------------- item isolation
def test_mixed_failure_isolation(client, fake_vllm):
    state = fake_vllm["state"]

    def hook(body):
        # fail exactly the item whose image_url marks it (URL lives in an
        # image_url content part, not in the prompt text)
        parts = body["messages"][0]["content"]
        urls = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
        if any("BADURL" in u for u in urls):
            return _resp(502, {"error": {"message": "upstream boom"}})
        return _completion(NSFW_OK)

    state["hook"] = hook
    try:
        r = client.post("/v1/tagging", json={"task": "nsfw_check", "inputs": [
            {"id": "good1", "image_url": "http://x/good1.jpg"},
            {"id": "bad", "image_url": "http://BADURL/x.jpg"},
            {"id": "good2", "image_url": "http://x/good2.jpg"},
        ]})
    finally:
        state["hook"] = None
    assert r.status_code == 200, r.text
    results = {x["id"]: x for x in r.json()["results"]}
    assert results["good1"]["status"] == "ok"
    assert results["good2"]["status"] == "ok"
    assert results["bad"]["status"] == "error"
    assert results["bad"]["code"] == "upstream_error"


def test_retry_then_recover(client, fake_vllm, monkeypatch):
    state = fake_vllm["state"]
    calls = {"n": 0}
    monkeypatch.setattr(tagging_module, "ITEM_RETRIES", 2)

    def hook(body):
        calls["n"] += 1
        if calls["n"] < 3:
            return _resp(503, {"error": {"message": "flaky"}})
        return _completion(NSFW_OK)

    state["hook"] = hook
    try:
        r = client.post("/v1/tagging", json={"task": "nsfw_check", "inputs": [
            {"id": "flake", "image_url": "http://x/a.jpg"}]})
    finally:
        state["hook"] = None
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["status"] == "ok", res
    assert calls["n"] == 3


def test_deterministic_4xx_not_retried(client, fake_vllm, monkeypatch):
    state = fake_vllm["state"]
    calls = {"n": 0}
    monkeypatch.setattr(tagging_module, "ITEM_RETRIES", 2)

    def hook(body):
        calls["n"] += 1
        return _resp(400, {"error": {"message": "bad request"}})

    state["hook"] = hook
    try:
        r = client.post("/v1/tagging", json={"task": "nsfw_check", "inputs": [
            {"id": "x", "image_url": "http://x/a.jpg"}]})
    finally:
        state["hook"] = None
    assert calls["n"] == 1  # no retry on deterministic rejection
    assert r.json()["results"][0]["status"] == "error"


def test_invalid_output_retried_then_error(client, fake_vllm, monkeypatch):
    state = fake_vllm["state"]
    monkeypatch.setattr(tagging_module, "ITEM_RETRIES", 1)

    def hook(body):
        return _completion("not json at all")

    state["hook"] = hook
    try:
        r = client.post("/v1/tagging", json={"task": "nsfw_check", "inputs": [
            {"id": "x", "image_url": "http://x/a.jpg"}]})
    finally:
        state["hook"] = None
    res = r.json()["results"][0]
    assert res["status"] == "error"
    assert res["code"] == "output_invalid"


def test_network_error_is_upstream_error(client, monkeypatch):
    # engine client pointed at a dead port for this one request
    monkeypatch.setattr(tagger_engine, "VLLM_API", "http://127.0.0.1:9/v1")
    monkeypatch.setattr(tagging_module, "ITEM_RETRIES", 0)
    r = client.post("/v1/tagging", json={"task": "nsfw_check", "inputs": [
        {"id": "x", "image_url": "http://x/a.jpg"}]})
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["status"] == "error"
    assert res["code"] == "upstream_error"


# ------------------------------------------------------------------- inline channel
def test_inline_channel_ok(client, fake_vllm):
    r = client.post("/v1/tagging", json={
        "prompt": "List the dominant colors as JSON.",
        "schema": SCHEMA_COLORS,
        "model": "qwen3.5-4b",
        "inputs": [{"id": "inl-1", "image_url": "http://x/a.jpg"}],
        "params": {"temperature": 0.7, "max_tokens": 128},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] is None and body["task_version"] is None
    assert body["results"][0]["status"] == "ok"
    sent = fake_vllm["seen"]["last_body"]
    assert sent["temperature"] == 0.7  # inline channel: caller params honored
    assert sent["response_format"]["json_schema"]["schema"] == SCHEMA_COLORS


def test_inline_bad_schema_400(client):
    r = client.post("/v1/tagging", json={
        "prompt": "p", "schema": {"type": "bogus-type-here"}, "model": "qwen3.5-4b",
        "inputs": [{"id": 1, "image_url": "u"}]})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_schema"


def test_inline_missing_model_400(client):
    r = client.post("/v1/tagging", json={
        "prompt": "p", "schema": SCHEMA_COLORS,
        "inputs": [{"id": 1, "image_url": "u"}]})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "missing_model"


# ------------------------------------------------------------------- params lock
def test_params_lock_request_override_ignored(client, fake_vllm):
    r = client.post("/v1/tagging", json={"task": "image_caption_metadata",
                                         "params": {"temperature": 9.9},
                                         "temperature": 9.9,
                                         "inputs": [{"id": 1, "image_url": "u"}]})
    assert r.status_code == 200, r.text
    sent = fake_vllm["seen"]["last_body"]
    assert sent["temperature"] == 0.2  # template value, not the 9.9 override


# ------------------------------------------------------------------- concurrency
def test_concurrency_validated(client):
    r = client.post("/v1/tagging", json={"task": "nsfw_check", "concurrency": 0,
                                         "inputs": [{"id": 1, "image_url": "u"}]})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_concurrency"
    r = client.post("/v1/tagging", json={"task": "nsfw_check", "concurrency": 99,
                                         "inputs": [{"id": 1, "image_url": "u"}]})
    assert r.status_code == 400
