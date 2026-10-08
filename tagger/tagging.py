"""/v1/tagging: batch task-template endpoint fanning out to the co-located vLLM.

One request -> N item-level chat/completions calls, each guided-decoded with
the template's json_schema. Isolation is per item: every item gets its own
status, ids pass through verbatim, and a failing item retries then errors
without affecting siblings. prompt/schema/params are server-locked from the
template (request-body override attempts are ignored); `task_version` is
echoed. An inline channel (prompt+schema in the request, no task) is the
escape hatch for one-off jobs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse
from jsonschema import Draft202012Validator

from registry.tasks import TaskTemplate, TemplateError, load_tasks

logger = logging.getLogger("dam-ai.tagging")
if not logger.handlers:  # uvicorn only configures its own loggers
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

HARD_INPUT_LIMIT = 64
DEFAULT_CONCURRENCY = max(1, int(os.environ.get("DAMAI_TAGGING_CONCURRENCY", "4")))
MAX_CONCURRENCY = 16
# Extra attempts for a failing item after the first try (upstream 5xx/network/
# schema-violation retries). 4xx from vLLM is deterministic and not retried.
ITEM_RETRIES = max(0, int(os.environ.get("DAMAI_TAGGING_ITEM_RETRIES", "2")))

# Refuses startup on any invalid template (T4 contract: bad template => no serve).
TASKS: dict[str, TaskTemplate] = load_tasks()

# Sentinel for "content was not parseable JSON".
_UNPARSEABLE = object()


def _vllm_api() -> str:
    """Late-bound so tests can monkeypatch tagger.engine.VLLM_API (no import cycle)."""
    from tagger import engine

    return engine.VLLM_API


def _get_client() -> httpx.AsyncClient:
    from tagger import engine

    return engine._get_client()


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "invalid_request_error", "code": code}},
    )


def _parse_json_content(content: str) -> Any:
    """Parse the model output; tolerate ```json fences, else _UNPARSEABLE."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return _UNPARSEABLE


def _schema_name(task: str | None) -> str:
    base = re.sub(r"[^a-zA-Z0-9_-]+", "_", task or "inline_output").strip("_")
    return base or "output"


@dataclass
class _Resolved:
    """Everything /v1/tagging needs after request-level validation."""

    model: str
    params: dict[str, Any]
    prompt: str
    schema: dict[str, Any]
    task: str | None
    task_version: int | None
    images_min: int
    images_max: int
    lock_params: bool = True  # False only on the inline channel


def _resolve(body: dict[str, Any]) -> tuple[JSONResponse | None, _Resolved | None]:
    """Task channel vs inline channel; all-or-nothing request-level 400s."""
    has_task = "task" in body and body.get("task") is not None
    has_inline = "prompt" in body or "schema" in body
    if has_task and has_inline:
        return _error(400, "invalid_request",
                      "either 'task' or inline 'prompt'+'schema', not both"), None
    if not has_task and not has_inline:
        return _error(400, "missing_task",
                      "missing required field: task (or inline prompt+schema)"), None

    if has_task:
        task_name = body["task"]
        if not isinstance(task_name, str) or task_name not in TASKS:
            known = ", ".join(sorted(TASKS)) or "(none)"
            return _error(400, "unknown_task",
                          f"unknown task: {task_name!r} (known: {known})"), None
        tmpl = TASKS[task_name]
        model = body.get("model") or tmpl.model_default
        if not isinstance(model, str) or not model:
            return _error(400, "invalid_model", "'model' must be a non-empty string"), None
        if model not in tmpl.allowed_models:
            return _error(400, "model_not_allowed",
                          f"model {model!r} not allowed for task {task_name!r} "
                          f"(allowed: {list(tmpl.allowed_models)})"), None
        try:
            merged = tmpl.validate_params(body.get("task_params") or {})
            prompt = tmpl.render_prompt(merged)
        except TemplateError as e:
            return _error(400, "invalid_task_params", str(e)), None
        # Server-locked sampling params: whatever the request carries under
        # 'params'/'temperature' etc. is never read; template params win.
        return None, _Resolved(model=model, params=dict(tmpl.params), prompt=prompt,
                               schema=tmpl.schema, task=tmpl.name,
                               task_version=tmpl.version,
                               images_min=tmpl.images["min"], images_max=tmpl.images["max"])

    # Inline channel: caller-owned prompt+schema; model explicit, params passable.
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _error(400, "invalid_prompt", "inline 'prompt' must be a non-empty string"), None
    schema = body.get("schema")
    if not isinstance(schema, dict) or not schema:
        return _error(400, "invalid_schema", "inline 'schema' must be a non-empty mapping"), None
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as e:  # noqa: BLE001 - reject before it reaches vLLM
        return _error(400, "invalid_schema",
                      f"inline 'schema' is not valid json-schema: {e}"), None
    model = body.get("model")
    if not isinstance(model, str) or not model:
        return _error(400, "missing_model", "inline requests must specify 'model'"), None
    params = body.get("params") if isinstance(body.get("params"), dict) else {}
    return None, _Resolved(model=model, params=dict(params), prompt=prompt, schema=schema,
                           task=None, task_version=None, images_min=1, images_max=16,
                           lock_params=False)


def _parse_inputs(body: dict[str, Any], res: _Resolved) -> JSONResponse | None:
    """Validate the inputs array shape (all-or-nothing 400s, before any vLLM call)."""
    inputs = body.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        return _error(400, "invalid_input", "'inputs' must be a non-empty array")
    if len(inputs) > HARD_INPUT_LIMIT:
        return _error(400, "batch_too_large",
                      f"inputs exceeds hard limit: {len(inputs)} > {HARD_INPUT_LIMIT} items "
                      f"(caller must shard)")
    for i, item in enumerate(inputs):
        if not isinstance(item, dict) or "id" not in item:
            return _error(400, "invalid_input", f"inputs[{i}] must be an object with an 'id'")
        has_single = isinstance(item.get("image_url"), str) and bool(item["image_url"])
        images = item.get("images")
        has_many = isinstance(images, list) and len(images) > 0 \
            and all(isinstance(u, str) and u for u in images)
        if has_single == has_many:  # exactly one of image_url / images is required
            return _error(400, "invalid_input",
                          f"inputs[{i}] needs exactly one of 'image_url' (string) or "
                          f"'images' (non-empty string array)")
        n = len(images) if has_many else 1
        if not res.images_min <= n <= res.images_max:
            return _error(400, "images_count_violation",
                          f"inputs[{i}] has {n} image(s), task allows "
                          f"{res.images_min}..{res.images_max}")
    return None


async def _run_item(
    item: dict[str, Any], res: _Resolved, sem: asyncio.Semaphore
) -> dict[str, Any]:
    """One item -> guided chat completion with per-item retry. Never raises."""
    urls = [item["image_url"]] if isinstance(item.get("image_url"), str) \
        else list(item["images"])
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": res.prompt}]
        + [{"type": "image_url", "image_url": {"url": u}} for u in urls],
    }]
    req_body: dict[str, Any] = {
        "model": res.model,
        "messages": messages,
        # guided decoding: vLLM enforces the schema server-side
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": _schema_name(res.task), "schema": res.schema},
        },
        **res.params,  # locked template params (inline: caller params)
    }
    validator = Draft202012Validator(res.schema)
    attempts = ITEM_RETRIES + 1
    code, message = "upstream_error", "unreachable"
    async with sem:
        for attempt in range(attempts):
            try:
                r = await _get_client().post(f"{_vllm_api()}/chat/completions", json=req_body)
            except httpx.HTTPError as e:
                code, message = "upstream_error", f"vLLM request failed: {e}"
            else:
                if r.status_code == 200:
                    try:
                        content = r.json()["choices"][0]["message"]["content"]
                    except (ValueError, KeyError, IndexError) as e:
                        code, message = "upstream_error", f"malformed vLLM response: {e}"
                    else:
                        out = _parse_json_content(content)
                        if out is _UNPARSEABLE:
                            code, message = "output_invalid", "model output is not valid JSON"
                        elif not validator.is_valid(out):
                            code, message = "output_invalid", \
                                "model output violates the task output schema"
                        else:
                            return {
                                "id": item["id"],
                                "status": "ok",
                                "output": out,
                                "model": res.model,
                                "task_version": res.task_version,
                            }
                else:
                    code, message = "upstream_error", \
                        f"vLLM HTTP {r.status_code}: {r.text[:200]}"
                    if r.status_code < 500:
                        # deterministic upstream rejection: retrying cannot help
                        break
            if attempt < attempts - 1:
                await asyncio.sleep(0.2 * (attempt + 1))
    return {"id": item["id"], "status": "error", "code": code, "error": message}


async def tagging(request: Request) -> JSONResponse:
    """POST /v1/tagging — batch task execution with item-level status isolation."""
    try:
        body: dict[str, Any] = await request.json()
    except Exception as e:  # noqa: BLE001
        return _error(400, "invalid_json", f"invalid JSON body: {e}")
    if not isinstance(body, dict):
        return _error(400, "invalid_json", "request body must be a JSON object")

    err, res = _resolve(body)
    if err is not None:
        return err
    assert res is not None
    err = _parse_inputs(body, res)
    if err is not None:
        return err
    concurrency = body.get("concurrency", DEFAULT_CONCURRENCY)
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) \
            or not 1 <= concurrency <= MAX_CONCURRENCY:
        return _error(400, "invalid_concurrency",
                      f"'concurrency' must be an integer in 1..{MAX_CONCURRENCY}")

    # Acceptance evidence for the params lock: effective sampling params are
    # logged per request — a caller-supplied temperature never appears here.
    logger.info("tagging task=%s model=%s items=%d effective params (locked): %s",
                res.task, res.model, len(body["inputs"]), res.params)

    sem = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = list(await asyncio.gather(
        *[_run_item(item, res, sem) for item in body["inputs"]]
    ))
    ok = sum(1 for r in results if r["status"] == "ok")
    logger.info("tagging done task=%s ok=%d error=%d", res.task, ok, len(results) - ok)

    return JSONResponse({
        "task": res.task,
        "task_version": res.task_version,
        "model": res.model,
        "results": results,
    })
