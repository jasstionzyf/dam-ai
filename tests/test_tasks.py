"""T4 tests: task template registry + external dir merge loading (DAMAI_TASK_DIRS).

All acceptance is pytest-level, no GPU needed (per the kanban card).
"""

from __future__ import annotations

import logging
import pathlib
import subprocess
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from registry.tasks import (  # noqa: E402
    TASK_DIRS_ENV,
    TaskTemplate,
    TemplateError,
    load_dir,
    load_tasks,
)

TASKS_DIR = ROOT / "registry" / "tasks.d"

VALID_TEMPLATE = """
name: my_task
version: 2
model_default: qwen3.5-4b
allowed_models: [qwen3.5-4b]
images: {min: 1, max: 4}
prompt: |
  Describe the image in {{ task_params.language }}.
  Return ONLY a JSON object matching the given schema.
schema:
  type: object
  properties:
    caption: {type: string}
  required: [caption]
params: {temperature: 0.2, max_tokens: 512}
task_params:
  - name: language
    type: string
    required: false
    default: en
"""


def _write_template(dir_path: pathlib.Path, raw, filename: str = "t.yaml") -> pathlib.Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    f = dir_path / filename
    f.write_text(raw if isinstance(raw, str) else yaml.safe_dump(raw), encoding="utf-8")
    return f


# ---------- acceptance 1: 3 built-in templates load with all fields ----------

def test_builtin_templates_load():
    ts = load_tasks()
    assert set(ts) == {"image_caption_metadata", "nsfw_check", "translate"}


def test_builtin_templates_fields_complete():
    ts = load_tasks()
    for name, t in ts.items():
        assert isinstance(t, TaskTemplate)
        assert t.name == name
        assert isinstance(t.version, int) and t.version >= 1
        assert t.model_default and t.model_default in t.allowed_models
        assert set(t.images) == {"min", "max"} and 0 <= t.images["min"] <= t.images["max"]
        assert t.prompt.strip()
        assert isinstance(t.schema, dict) and t.schema.get("type") == "object"
        assert isinstance(t.params, dict) and t.params
        assert t.source == "builtin"


# ---------- acceptance 2: Jinja2 task_params injection renders ----------

def test_jinja2_task_params_injected():
    ts = load_tasks()
    rendered = ts["image_caption_metadata"].render_prompt({"language": "zh", "max_keywords": 10})
    assert "zh" in rendered
    assert "10" in rendered


def test_render_fills_declared_defaults_and_required_stays_strict():
    ts = load_tasks()
    # declared optional default (source_language=auto) is filled in
    rendered = ts["translate"].render_prompt(
        {"text": "a red car", "target_language": "ja"})
    assert "ja" in rendered and "auto" in rendered and "a red car" in rendered
    # a required param with no default is NOT silently blanked: rendering
    # without it raises (StrictUndefined), matching validate_params' rejection.
    with pytest.raises(Exception, match="target_language"):
        ts["translate"].render_prompt({"text": "a red car"})


def test_validate_params_merges_defaults_and_rejects_unknown_and_bad_type():
    ts = load_tasks()
    merged = ts["translate"].validate_params({"text": "a red car", "target_language": "ja"})
    assert merged == {"source_language": "auto", "text": "a red car", "target_language": "ja"}

    with pytest.raises(TemplateError, match="unknown task_param"):
        ts["translate"].validate_params({"text": "x", "target_language": "ja", "evil": 1})
    with pytest.raises(TemplateError, match="must be string"):
        ts["translate"].validate_params({"text": "x", "target_language": 42})
    with pytest.raises(TemplateError, match="required task_param"):
        ts["translate"].validate_params({"text": "x"})


def test_validate_output_against_template_schema():
    ts = load_tasks()
    good = {"is_nsfw": False, "confidence": 0.9, "reason": "clean"}
    assert ts["nsfw_check"].validate_output(good) is True
    bad_type = {"is_nsfw": "no", "confidence": 0.9, "reason": "clean"}
    assert ts["nsfw_check"].validate_output(bad_type) is False
    assert ts["nsfw_check"].validate_output({"is_nsfw": False}) is False  # missing required


# ---------- acceptance 3: bad templates refuse to load ----------

@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r: r.pop("prompt"), "missing required field 'prompt'"),
        (lambda r: r.pop("schema"), "missing required field 'schema'"),
        (lambda r: r.pop("params"), "missing required field 'params'"),
        (lambda r: r.update(version="3"), "version must be a positive integer"),
        (lambda r: r.update(model_default=""), "model_default must be a non-empty string"),
        (lambda r: r.update(model_default="not-whitelisted"),
         "model_default 'not-whitelisted' not in allowed_models"),
        (lambda r: r.update(allowed_models=[]), "allowed_models must be a non-empty list"),
        (lambda r: r.update(images={"min": 4, "max": 1}), "images must be"),
        (lambda r: r.update(schema={"type": "not-a-real-type"}), "not a valid json-schema"),
        (lambda r: r.update(prompt="Hello {{ undefined_var }}"), "undeclared variables"),
        (lambda r: r.update(prompt="Hello {{ if % }}}"), "not valid Jinja2"),
        (lambda r: r.update(task_params=[{"name": "x", "type": "array"}]), "type must be one of"),
    ],
)
def test_bad_template_refused(tmp_path, mutate, match):
    raw = yaml.safe_load(VALID_TEMPLATE)
    mutate(raw)
    _write_template(tmp_path, raw)
    with pytest.raises(TemplateError, match=match):
        load_dir(tmp_path)


def test_unparsable_yaml_refused(tmp_path):
    _write_template(tmp_path, "name: [unclosed\n  bad yaml: :\n")
    with pytest.raises(TemplateError, match="unparsable yaml"):
        load_dir(tmp_path)


def test_duplicate_name_within_one_dir_refused(tmp_path):
    _write_template(tmp_path, VALID_TEMPLATE, "a.yaml")
    _write_template(tmp_path, VALID_TEMPLATE, "b.yaml")
    with pytest.raises(TemplateError, match="duplicate template name"):
        load_dir(tmp_path)


def test_missing_tasks_dir_refused(tmp_path):
    with pytest.raises(TemplateError, match="not found"):
        load_dir(tmp_path / "nope")


def test_empty_builtins_refused(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(TemplateError, match="no task templates"):
        load_tasks(tasks_dir=empty)


def test_bad_template_refuses_process_startup(tmp_path):
    """Server entry must SystemExit when a template (builtin or external) is invalid."""
    raw = yaml.safe_load(VALID_TEMPLATE)
    raw["schema"] = {"type": "bogus"}
    bad_dir = tmp_path / "ext"
    _write_template(bad_dir, raw)
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from registry.tasks import load_tasks; "
        "load_tasks(external_dirs=[%r])" % (str(ROOT), str(bad_dir))
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "HOME": str(pathlib.Path.home())})
    assert r.returncode != 0
    assert "not a valid json-schema" in (r.stderr + r.stdout)


# ---------- acceptance 4: external dir override + manifest log + multi-dir order ----------

def _external_version(dir_path: pathlib.Path, version: int, marker: str) -> None:
    raw = yaml.safe_load(VALID_TEMPLATE)
    raw["version"] = version
    raw["prompt"] = raw["prompt"] + f"\nMARKER: {marker}\n"
    _write_template(dir_path, raw)


def test_external_same_name_overrides_builtin_with_log(tmp_path, caplog):
    ext1 = tmp_path / "ext1"
    raw = yaml.safe_load(VALID_TEMPLATE)
    head, tail = VALID_TEMPLATE.split("prompt:", 1)
    _, params_on = VALID_TEMPLATE.split("params:", 1)
    raw.update(name="nsfw_check", version=2,
               prompt=head + "prompt: |\n  OVERRIDE MARKER EXT1\nparams:" + params_on)
    _write_template(ext1, raw, "nsfw_check.yaml")
    with caplog.at_level(logging.INFO, logger="dam-ai.tasks"):
        ts = load_tasks(external_dirs=[str(ext1)])
    assert ts["nsfw_check"].version == 2
    assert "OVERRIDE MARKER EXT1" in ts["nsfw_check"].prompt
    assert ts["nsfw_check"].source == str(ext1)
    # other built-ins survive the merge
    assert {"image_caption_metadata", "translate"} <= set(ts)
    # override manifest logged: template name + winning dir
    logs = "\n".join(r.getMessage() for r in caplog.records)
    assert "nsfw_check" in logs and str(ext1) in logs
    assert "override" in logs.lower()


def test_external_multi_dirs_later_overrides_earlier(tmp_path):
    ext1, ext2 = tmp_path / "ext1", tmp_path / "ext2"
    _external_version(ext1, version=2, marker="FROM_EXT1")
    _external_version(ext2, version=3, marker="FROM_EXT2")
    ts = load_tasks(external_dirs=[str(ext1), str(ext2)])
    assert ts["my_task"].version == 3
    assert "FROM_EXT2" in ts["my_task"].prompt and "FROM_EXT1" not in ts["my_task"].prompt


def test_task_dirs_env_parsed_colon_separated(tmp_path, monkeypatch):
    ext1, ext2 = tmp_path / "ext1", tmp_path / "ext2"
    _external_version(ext1, version=2, marker="E1")
    _external_version(ext2, version=3, marker="E2")
    monkeypatch.setenv(TASK_DIRS_ENV, f"{ext1}:{ext2}")
    ts = load_tasks()
    assert ts["my_task"].version == 3 and "E2" in ts["my_task"].prompt


def test_external_new_template_added_not_just_override(tmp_path):
    ext = tmp_path / "ext"
    raw = yaml.safe_load(VALID_TEMPLATE)
    raw.update(name="business_only_task", version=1)
    _write_template(ext, raw)
    ts = load_tasks(external_dirs=[str(ext)])
    assert "business_only_task" in ts


def test_frozen_template_locks_core_fields():
    ts = load_tasks()
    t = ts["nsfw_check"]
    with pytest.raises(Exception):
        t.prompt = "mutated"
    with pytest.raises(Exception):
        t.params = {}
