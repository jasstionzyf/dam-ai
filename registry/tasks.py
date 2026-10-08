"""Task template registry: YAML templates + Jinja2 prompt rendering + json schema validation.

Templates live in `registry/tasks.d/` (built-in, git-managed); business/private
templates live in external directories injected via `DAMAI_TASK_DIRS` (colon
separated). Same-name external templates override built-ins, with an override
manifest logged at startup. Any template failing validation refuses startup.

Prompt/schema/params are server-locked (comparability of outputs is the
precondition for write-back); `task_version` is returned with every response.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined
from jinja2 import meta as jinja2_meta
from jsonschema import Draft202012Validator

logger = logging.getLogger("dam-ai.tasks")

VALID_TASK_PARAM_TYPES = {"string", "number", "boolean"}

# Server-locked: callers may not override these per-request.
LOCKED_FIELDS = ("prompt", "schema", "params", "name", "version")

# jinja2 environment: task_params injected as `task_params.<name>`, strict on
# undefined so a typo'd template variable fails at load time, not at runtime.
_JINJA_ENV = Environment(undefined=StrictUndefined, keep_trailing_newline=True)

DEFAULT_TASKS_DIR = Path(__file__).resolve().parent / "tasks.d"
TASK_DIRS_ENV = "DAMAI_TASK_DIRS"


class TemplateError(Exception):
    """Raised when the task template registry is missing, unparsable or schema-invalid."""


@dataclass(frozen=True)
class TaskParam:
    name: str
    type: str  # "string" | "number" | "boolean"
    description: str = ""
    required: bool = False
    default: Any = None


@dataclass(frozen=True)
class TaskTemplate:
    name: str
    version: int
    model_default: str
    allowed_models: tuple[str, ...]
    images: dict[str, int]  # {"min": int, "max": int}
    prompt: str  # Jinja2 source; render with task_params injection
    schema: dict[str, Any]  # json schema for guided-decoding output
    params: dict[str, Any]  # vLLM sampling params, server-locked
    task_params: tuple[TaskParam, ...] = ()
    description: str = ""
    source: str = ""  # "builtin" or the external dir it came from

    def render_prompt(self, task_params: dict[str, Any] | None = None) -> str:
        """Render the prompt with task_params injected as `task_params.<name>`.

        Declared defaults are filled in first, caller values overlay them —
        the same merge rule as validate_params.
        """
        merged = {p.name: p.default for p in self.task_params if p.default is not None}
        merged.update(task_params or {})
        return _JINJA_ENV.from_string(self.prompt).render(task_params=merged)

    def validate_params(self, task_params: dict[str, Any]) -> dict[str, Any]:
        """Validate caller-supplied task_params against the template declaration.

        Returns the merged params (declared defaults filled in). Unknown names
        and type violations raise TemplateError — params beyond the declared
        surface are rejected, everything else on the template is locked.
        """
        declared = {p.name: p for p in self.task_params}
        merged: dict[str, Any] = {
            p.name: p.default for p in self.task_params if p.default is not None
        }
        for key, value in (task_params or {}).items():
            if key not in declared:
                raise TemplateError(f"task {self.name!r}: unknown task_param {key!r}")
            p = declared[key]
            ok = {
                "string": lambda v: isinstance(v, str),
                "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
                "boolean": lambda v: isinstance(v, bool),
            }[p.type](value)
            if not ok:
                raise TemplateError(f"task {self.name!r}: task_param {key!r} must be {p.type}")
            merged[key] = value
        for p in self.task_params:
            if p.required and p.name not in merged:
                raise TemplateError(f"task {self.name!r}: required task_param {p.name!r} missing")
        return merged

    def validate_output(self, output: Any) -> bool:
        """True when `output` conforms to the template's json schema."""
        return Draft202012Validator(self.schema).is_valid(output)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "model_default": self.model_default,
            "allowed_models": list(self.allowed_models),
            "images": dict(self.images),
            "task_params": [
                {"name": p.name, "type": p.type, "required": p.required, "default": p.default}
                for p in self.task_params
            ],
            "description": self.description,
            "source": self.source,
        }


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise TemplateError(msg)


def _parse_template(raw: Any, source: str, filename: str) -> TaskTemplate:
    """Validate one raw YAML mapping into a TaskTemplate; raise TemplateError on any violation."""
    _require(isinstance(raw, dict), f"{source}/{filename}: template must be a mapping")
    for key in ("name", "version", "model_default", "allowed_models",
                "images", "prompt", "schema", "params"):
        _require(key in raw, f"{source}/{filename}: missing required field {key!r}")

    name = raw["name"]
    _require(isinstance(name, str) and bool(name.strip()),
             f"{source}/{filename}: name must be a non-empty string")

    version = raw["version"]
    _require(isinstance(version, int) and not isinstance(version, bool) and version >= 1,
             f"{source}/{filename}: version must be a positive integer")

    model_default = raw["model_default"]
    _require(isinstance(model_default, str) and bool(model_default.strip()),
             f"{source}/{filename}: model_default must be a non-empty string")

    allowed = raw["allowed_models"]
    _require(isinstance(allowed, list) and bool(allowed)
             and all(isinstance(m, str) and m for m in allowed),
             f"{source}/{filename}: allowed_models must be a non-empty list of strings")
    _require(model_default in allowed,
             f"{source}/{filename}: model_default {model_default!r} not in allowed_models")

    images = raw["images"]
    _require(isinstance(images, dict) and set(images) == {"min", "max"}
             and all(isinstance(images[k], int) and not isinstance(images[k], bool)
                     for k in ("min", "max"))
             and 0 <= images["min"] <= images["max"],
             f"{source}/{filename}: images must be {{min, max}} with 0 <= min <= max "
             f"(min 0 = image optional)")

    prompt = raw["prompt"]
    _require(isinstance(prompt, str) and bool(prompt.strip()),
             f"{source}/{filename}: prompt must be a non-empty string")
    try:
        parsed = _JINJA_ENV.parse(prompt)
    except Exception as e:
        raise TemplateError(f"{source}/{filename}: prompt is not valid Jinja2: {e}") from e
    # StrictUndefined: a rendered-but-undeclared variable must fail at load time.
    free = set(jinja2_meta.find_undeclared_variables(parsed))
    declared_names = {
        p["name"] for p in raw.get("task_params") or [] if isinstance(p, dict) and "name" in p
    }
    undeclared = free - declared_names - {"task_params"}
    _require(not undeclared,
             f"{source}/{filename}: prompt uses undeclared variables {sorted(undeclared)} "
             f"(declare them under task_params)")

    schema = raw["schema"]
    _require(isinstance(schema, dict) and bool(schema),
             f"{source}/{filename}: schema must be a non-empty mapping")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as e:
        raise TemplateError(f"{source}/{filename}: schema is not a valid json-schema: {e}") from e

    params = raw["params"]
    _require(isinstance(params, dict),
             f"{source}/{filename}: params must be a mapping (vLLM sampling params)")

    raw_task_params = raw.get("task_params") or []
    _require(isinstance(raw_task_params, list), f"{source}/{filename}: task_params must be a list")
    task_params: list[TaskParam] = []
    seen: set[str] = set()
    for i, p in enumerate(raw_task_params):
        _require(isinstance(p, dict) and "name" in p and "type" in p,
                 f"{source}/{filename}: task_params[{i}] needs name+type")
        _require(p["name"] not in seen, f"{source}/{filename}: duplicate task_param {p['name']!r}")
        seen.add(p["name"])
        _require(p["type"] in VALID_TASK_PARAM_TYPES,
                 f"{source}/{filename}: task_param {p['name']!r} type must be one of "
                 f"{sorted(VALID_TASK_PARAM_TYPES)}")
        _require(isinstance(p.get("required", False), bool),
                 f"{source}/{filename}: task_param {p['name']!r} required must be boolean")
        task_params.append(TaskParam(
            name=p["name"], type=p["type"],
            description=str(p.get("description", "")),
            required=p.get("required", False),
            default=p.get("default"),
        ))

    return TaskTemplate(
        name=name, version=version, model_default=model_default,
        allowed_models=tuple(allowed), images={"min": images["min"], "max": images["max"]},
        prompt=prompt, schema=schema, params=dict(params),
        task_params=tuple(task_params),
        description=str(raw.get("description", "")),
        source=source,
    )


def load_dir(path: Path) -> dict[str, TaskTemplate]:
    """Load every *.yaml/*.yml in one directory. Missing dir raises; bad file raises."""
    templates: dict[str, TaskTemplate] = {}
    if not path.is_dir():
        raise TemplateError(f"task dir not found: {path}")
    files = sorted(p for p in path.iterdir() if p.suffix in (".yaml", ".yml"))
    for f in files:
        try:
            raw = yaml.safe_load(f.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise TemplateError(f"{path}/{f.name}: unparsable yaml: {e}") from e
        t = _parse_template(raw, source=str(path), filename=f.name)
        _require(t.name not in templates,
                 f"{path}/{f.name}: duplicate template name {t.name!r} within one dir")
        templates[t.name] = t
    return templates


def load_tasks(
    tasks_dir: Path | str | None = None, external_dirs: list[str] | None = None
) -> dict[str, TaskTemplate]:
    """Built-in tasks.d merged with DAMAI_TASK_DIRS (later dirs override earlier).

    Override manifest (template + winner dir) is logged, per design.md:
    same-name external overrides built-in; any validation failure refuses startup.
    """
    base = Path(tasks_dir) if tasks_dir else DEFAULT_TASKS_DIR
    merged = load_dir(base)
    for name, t in merged.items():
        object.__setattr__(t, "source", "builtin")  # frozen dataclass; tag after load

    dir_list = list(external_dirs if external_dirs is not None
                    else [d for d in os.environ.get(TASK_DIRS_ENV, "").split(":") if d.strip()])
    for d in dir_list:
        overrides = load_dir(Path(d))
        for name, t in overrides.items():
            if name in merged:
                logger.info("task template override: %r (v%s builtin -> v%s from %s)",
                            name, merged[name].version, t.version, d)
            merged[name] = t
    overrides_list = [n for n, t in merged.items() if t.source not in ("builtin",)]
    if overrides_list:
        logger.info("external task templates loaded: %s", sorted(overrides_list))
    if not merged:
        raise TemplateError(f"no task templates found in {base}")
    return merged
