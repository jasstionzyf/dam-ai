# Task template guide

`/v1/tagging` executes **task templates**: versioned (prompt + json schema +
sampling params) triplets stored as YAML in Git — not in a database. Locking
these server-side is what makes outputs comparable across calls and models,
which is the precondition for writing results back to asset records.

## Anatomy of a template

A complete, copyable example — this file ships as
[`examples/tasks/alt_text.yaml`](../examples/tasks/alt_text.yaml) and loads
through the real registry (see `tests/test_docs_examples.py`):

```yaml
name: alt_text
version: 1
model_default: qwen3.5-4b
allowed_models: [qwen3.5-4b]
images: {min: 1, max: 1}
description: "Write one concise accessibility alt-text sentence for an image."
task_params:
  - name: language
    type: string
    description: "Language code for the alt text (e.g. en, zh)."
    required: false
    default: en
  - name: max_chars
    type: number
    description: "Soft upper bound of alt text characters."
    required: false
    default: 125
prompt: |
  You write alt text for a digital asset library. Look at the image and write
  exactly one concise alt-text sentence in language code
  "{{ task_params.language }}". Describe the subject and its salient visual
  attributes. No editorials, no "image of ..." phrasing. Keep it under
  {{ task_params.max_chars }} characters.

  Return ONLY a JSON object matching the given schema.
schema:
  type: object
  properties:
    alt_text:
      type: string
      minLength: 3
    language:
      type: string
      minLength: 2
  required: [alt_text, language]
  additionalProperties: false
params: {temperature: 0.2, max_tokens: 128}
```

## Fields

| Field | Meaning |
|---|---|
| `name` | Template id used in the `/v1/tagging` `task` field. Unique per directory. |
| `version` | Positive int. **Bump it whenever prompt/schema/params change** — every response echoes `task_version`, and consumers store it to know which outputs to re-generate. |
| `model_default` | Model used when the request omits `model`. Must be in `allowed_models`. |
| `allowed_models` | Whitelist. A request naming any other model gets 4xx (`model_not_allowed`). Changing models requires validation, hence the explicit list. |
| `images` | `{min, max}` images per input item; `min: 0` = image optional (text-only tasks like translate). |
| `prompt` | Jinja2 source. Declare every variable under `task_params` — rendering with an undeclared variable fails at **load time**, not per-request. |
| `schema` | JSON Schema (draft 2020-12) the output is validated against; also drives vLLM guided decoding. Use `additionalProperties: false` + explicit `required`. |
| `params` | vLLM sampling params. **Locked**: caller overrides are ignored, never merged. |
| `task_params` | The only request-tunable surface: `string` \| `number` \| `boolean`, each optional `default`, optional `required`. Unknown names and type violations are rejected per-request. |
| `description` | Free text, returned by `/v1/tagging` template listings. |

## Locking rules (server-enforced)

- `prompt` / `schema` / `params` are immutable from the request — a request may
  only choose `task` (or `model`, `task_params`).
- `task_params` rendering: declared defaults are filled in first, caller values
  overlay them; unknown names → 4xx; missing `required` params → 4xx.
- Templates are frozen after load — mutating one in-process raises.
- Any template that fails validation at startup **refuses the whole service**
  (bad YAML, bad schema, undeclared Jinja2 variables, `model_default` outside
  the whitelist — all startup failures by design).

## Versioning discipline

1. Behavior change (prompt wording, schema fields, params) → `version += 1`,
   same file.
2. Additive new template → new file, `version: 1`.
3. Never reuse a version number with different semantics — downstream assets
   record `(task, task_version)` and rely on it being stable.
4. Evaluations compare models, not prompts: fix the template version, vary the
   model within `allowed_models`.

## Where templates live

Built-ins (generic, open-source): `registry/tasks.d/*.yaml`. Business/private
templates never enter this repo — they go in an external directory merged at
startup via `DAMAI_TASK_DIRS`; see [external-tasks.md](external-tasks.md).

## Calling it

```bash
curl -s localhost:8091/v1/tagging -H 'Content-Type: application/json' -d '{
  "task": "alt_text",
  "inputs": [{"id": "asset-1", "image_url": "https://example.com/a.jpg"}],
  "task_params": {"language": "zh"}
}'
```

Response items carry `output` (schema-validated JSON), `task_version`, and
per-item `status`/`id` passthrough (batch cap: 64 inputs per request).
