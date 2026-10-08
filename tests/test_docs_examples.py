"""T8 tests: docs guides claim real things — verify them against the code.

docs/external-tasks.md ships a copyable example template; docs/model-onboarding.md
and docs/template-guide.md quote registry shapes. This suite pins those claims:
the doc example loads through the real registry, and the doc-quoted model/
template fields match the actual code.
"""

from __future__ import annotations

import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from registry.tasks import load_dir, load_tasks  # noqa: E402

DOCS = ROOT / "docs"
EXAMPLES = ROOT / "examples" / "tasks"


def _doc_text(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


def _fenced_yaml_blocks(text: str) -> list[dict]:
    """Every ```yaml fenced block in a doc, parsed."""
    blocks = re.findall(r"```yaml\n(.*?)```", text, flags=re.S)
    return [yaml.safe_load(b) for b in blocks]


# ---------- acceptance 1: the shipped example template loads ----------

def test_example_template_loads_via_registry():
    ts = load_dir(EXAMPLES)
    t = ts["alt_text"]
    assert t.name == "alt_text" and t.version == 1
    assert t.model_default in t.allowed_models
    assert t.source == str(EXAMPLES)
    # task_params render + defaults merge like any registry template
    rendered = t.render_prompt({"language": "zh", "max_chars": 80})
    assert "zh" in rendered and "80" in rendered
    merged = t.validate_params({})
    assert merged == {"language": "en", "max_chars": 125}
    assert t.validate_output({"alt_text": "A red bicycle against a brick wall",
                              "language": "en"}) is True
    assert t.validate_output({"alt_text": "x"}) is False  # too short


def test_example_template_directory_not_picked_up_by_default_builtins():
    ts = load_tasks()
    assert set(ts) == {"image_caption_metadata", "nsfw_check", "translate"}


# ---------- acceptance 2: every complete yaml template inside the docs loads ----------

def test_docs_yaml_templates_are_registry_valid():
    for doc in ("template-guide.md", "external-tasks.md", "model-onboarding.md"):
        for i, raw in enumerate(_fenced_yaml_blocks(_doc_text(doc))):
            if not isinstance(raw, dict) or "name" not in raw or "prompt" not in raw:
                continue  # config snippet, not a template
            from registry.tasks import _parse_template
            t = _parse_template(raw, source=f"docs/{doc}#block{i}", filename=doc)
            assert t.model_default in t.allowed_models


# ---------- acceptance 3: doc-quoted code shapes match the code ----------

def test_model_onboarding_doc_fields_exist_in_schema():
    text = _doc_text("model-onboarding.md")
    assert "registry/models.yaml" in text and "DAMAI_MODELS_ROOT" in text
    # quoted model field names all exist in the real registry yaml
    sys_path = ROOT
    sys.path.insert(0, str(sys_path))
    from registry.schema import load_registry
    reg = load_registry(str(ROOT / "registry" / "models.yaml"))
    spec = reg["clip-vit-l14"]
    have = (set(spec.dam_ai_meta) | {"path", "preprocess", "weights"}
            | set(spec.preprocess.to_dict()))
    for field in re.findall(r"`([a-z_]+)`", text):
        if field in ("dims", "modalities", "normalized", "loader", "engine",
                     "weights", "preprocess", "resolution", "mean", "std",
                     "pooling", "path"):
            assert field in have or hasattr(spec, field), \
                f"doc field {field!r} not in registry entry"


def test_template_guide_doc_fields_exist_in_parser():
    text = _doc_text("template-guide.md")
    from registry.tasks import LOCKED_FIELDS
    for field in ("prompt", "schema", "params"):
        assert field in text and field in LOCKED_FIELDS
    for field in ("name", "version", "model_default", "allowed_models", "images"):
        assert f"`{field}`" in text  # documented as a template field
