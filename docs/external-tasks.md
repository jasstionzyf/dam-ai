# External task templates (`DAMAI_TASK_DIRS`)

Task templates are part of the prompt contract — business prompt engineering
does not belong in the open-source repo. Built-in templates in
`registry/tasks.d/` stay generic; your own go in **any directory outside this
repository**, merged at startup via the `DAMAI_TASK_DIRS` environment variable.

## How merging works

- `DAMAI_TASK_DIRS` is a **colon-separated** list of directories, e.g.
  `/srv/damai-tasks:/srv/damai-tasks-staging`. Later directories override
  earlier ones.
- Same template `name` in an external dir **overrides the built-in** (or a
  dir scanned earlier); the override is logged at startup, e.g.:
  `task template override: 'nsfw_check' (v1 builtin -> v3 from /srv/damai-tasks)`.
- A new name in an external dir is simply **added** to the registry.
- External templates go through **exactly the same validation** as built-ins:
  schema check, `model_default ∈ allowed_models`, undeclared Jinja2 variables,
  params mapping. One invalid template **refuses startup** — the server never
  serves with a partially-valid template set.
- Everything else applies equally: `allowed_models` whitelist, locked
  prompt/schema/params, `task_version` echo, frozen templates.

## Example template

This copyable example is real: it ships at
[`examples/tasks/alt_text.yaml`](../examples/tasks/alt_text.yaml) and loads
through the registry (`tests/test_docs_examples.py` keeps it verified):

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

Copy it to your external directory and adapt name/prompt/schema. The built-in
templates in `registry/tasks.d/` (`image_caption_metadata`, `nsfw_check`,
`translate`) are additional full references.

## Wiring it up (compose)

Point `DAMAI_TASK_DIRS` at your directory and mount it into the tagger
container:

```yaml
services:
  tagger:
    environment:
      DAMAI_TASK_DIRS: /tasks/business
    volumes:
      - /srv/damai/business-tasks:/tasks/business:ro
```

Restart the service afterwards; the startup log lists what was merged:

```bash
docker logs damai-tagger 2>&1 | grep -i 'override\|external task'
```

## Verify your template loaded

```bash
# local check without starting anything (uses this repo's venv):
.venv/bin/python -c "
from pathlib import Path
from registry.tasks import load_dir
ts = load_dir(Path('/srv/damai/business-tasks'))
print(sorted(ts))            # your template names
print(ts['alt_text'].version)
"
```

An invalid template raises `TemplateError` with the exact violation — same
message the server would refuse startup with.

## Rolling out changes

Bump `version` on every prompt/schema/params change. Responses carry
`task_version`; your asset records store it, so a bump is also your re-generate
cursor. To A/B a risky prompt, put the candidate under a different `name`
(say `alt_text_v2`) in the external dir, compare outputs, then cut over and
retire the old one — do not edit in place without bumping.
