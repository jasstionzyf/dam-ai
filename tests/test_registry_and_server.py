"""dam-ai T1 tests: registry schema validation + server skeleton endpoints."""

from __future__ import annotations

import copy
import pathlib
import subprocess
import sys

import pytest
import yaml
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from registry.schema import RegistryError, load_registry  # noqa: E402
from server.app import app  # noqa: E402

REGISTRY_YAML = ROOT / "registry" / "models.yaml"

client = TestClient(app)


# ---------- registry loading: 4 entries, all fields present ----------

def test_registry_has_exactly_4_models():
    reg = load_registry(REGISTRY_YAML)
    assert set(reg) == {"qwen3-vl-embedding-2b", "siglip-so400m", "clip-vit-l14", "dinov2-base"}


def test_registry_entries_all_fields_valid():
    reg = load_registry(REGISTRY_YAML)
    for name, spec in reg.items():
        assert spec.name == name
        assert spec.path.startswith("/models/")
        assert spec.loader in {"sentence_transformers", "open_clip", "transformers"}
        assert spec.engine in {"transformers", "open_clip", "vllm"}
        assert isinstance(spec.dims, int) and spec.dims > 0
        assert spec.modalities and set(spec.modalities) <= {"image", "text"}
        assert isinstance(spec.normalized, bool)
        pp = spec.preprocess
        assert pp.resolution > 0 and len(pp.mean) == 3 and len(pp.std) == 3
        assert pp.pooling in {"cls", "mean", "pooler", "last_token"}
        assert isinstance(spec.loader_config, dict)


def test_registry_known_values():
    reg = load_registry(REGISTRY_YAML)
    assert reg["qwen3-vl-embedding-2b"].dims == 2048
    assert reg["qwen3-vl-embedding-2b"].loader == "sentence_transformers"
    assert reg["siglip-so400m"].dims == 1152
    assert reg["clip-vit-l14"].dims == 768
    assert reg["clip-vit-l14"].weights == "ViT-L-14.pt"
    assert reg["clip-vit-l14"].path == "/models/clip"
    assert reg["dinov2-base"].dims == 768
    assert reg["dinov2-base"].modalities == ("image",)  # image-only


# ---------- bad yaml / bad entries refuse to load ----------

def _write_and_load(tmp_path: pathlib.Path, doc) -> None:
    p = tmp_path / "models.yaml"
    if isinstance(doc, str):
        p.write_text(doc, encoding="utf-8")  # intentionally unparsable
    else:
        with open(p, "w") as f:
            yaml.safe_dump(doc, f, allow_unicode=True)
    load_registry(p)


BAD_YAML_TEXT = "models: [ {name: broken, ::\n  - unbalanced brackets: [[["


def test_bad_yaml_text_refused(tmp_path):
    p = tmp_path / "broken.yaml"
    p.write_text(BAD_YAML_TEXT, encoding="utf-8")
    with pytest.raises(RegistryError, match="YAML parse error"):
        load_registry(p)


def test_missing_file_refused(tmp_path):
    with pytest.raises(RegistryError, match="not found"):
        load_registry(tmp_path / "nope.yaml")


def test_unknown_field_refused(tmp_path):
    doc = yaml.safe_load(REGISTRY_YAML.read_text())
    doc["models"][0]["bogus_field"] = 1
    with pytest.raises(RegistryError, match="unknown fields"):
        _write_and_load(tmp_path, doc)


def test_missing_required_field_refused(tmp_path):
    doc = yaml.safe_load(REGISTRY_YAML.read_text())
    del doc["models"][1]["dims"]
    with pytest.raises(RegistryError, match="missing required field"):
        _write_and_load(tmp_path, doc)


def test_duplicate_model_name_refused(tmp_path):
    doc = yaml.safe_load(REGISTRY_YAML.read_text())
    doc["models"].append(copy.deepcopy(doc["models"][0]))
    with pytest.raises(RegistryError, match="duplicate model name"):
        _write_and_load(tmp_path, doc)


def test_bad_loader_refused(tmp_path):
    doc = yaml.safe_load(REGISTRY_YAML.read_text())
    doc["models"][2]["loader"] = "magic_loader"
    with pytest.raises(RegistryError, match="loader"):
        _write_and_load(tmp_path, doc)


def test_bad_dims_refused(tmp_path):
    doc = yaml.safe_load(REGISTRY_YAML.read_text())
    doc["models"][0]["dims"] = -1
    with pytest.raises(RegistryError, match="dims"):
        _write_and_load(tmp_path, doc)


def test_bad_preprocess_pooling_refused(tmp_path):
    doc = yaml.safe_load(REGISTRY_YAML.read_text())
    doc["models"][3]["preprocess"]["pooling"] = "avg"
    with pytest.raises(RegistryError, match="pooling"):
        _write_and_load(tmp_path, doc)


def test_empty_registry_refused(tmp_path):
    with pytest.raises(RegistryError):
        _write_and_load(tmp_path, {"models": []})


# ---------- bad registry refuses process startup (SystemExit) ----------

def test_bad_registry_refuses_startup(tmp_path):
    """Server entry must SystemExit (non-zero) when registry is invalid."""
    doc = yaml.safe_load(REGISTRY_YAML.read_text())
    doc["models"][0]["dims"] = "not-an-int"
    p = tmp_path / "models.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(doc, f, allow_unicode=True)

    env = {"PATH": "/usr/bin:/bin", "DAMAI_REGISTRY": str(p)}
    env.update({k: v for k, v in __import__("os").environ.items()
                if k.startswith(("PYTHON", "UV_", "LD_"))})
    r = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), "-c",
         "import sys; sys.path.insert(0, '.'); import server.app"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert r.returncode != 0, f"server must refuse startup on bad registry, got: {r.stdout}"
    assert "refused to start" in (r.stderr + r.stdout)


# ---------- server endpoints ----------

def test_healthz_always_200():
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["service"] == "dam-ai"


def test_readyz_ok():
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["registry_models"] == 4


def test_v1_models_returns_4_with_dam_ai_meta():
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list" and len(body["data"]) == 4
    yaml_doc = yaml.safe_load(REGISTRY_YAML.read_text())
    by_name = {m["name"]: m for m in yaml_doc["models"]}
    for m in body["data"]:
        assert m["object"] == "model" and m["owned_by"] == "dam-ai"
        meta = m["dam_ai"]
        assert set(meta) == {"modalities", "dims", "normalized", "engine", "loader"}
        src = by_name[m["id"]]
        assert meta["modalities"] == src["modalities"]
        assert meta["dims"] == src["dims"]
        assert meta["normalized"] == src["normalized"]
        assert meta["engine"] == src["engine"]
        assert meta["loader"] == src["loader"]


def test_v1_models_single_lookup_and_404():
    r = client.get("/v1/models/clip-vit-l14")
    assert r.status_code == 200
    assert r.json()["dam_ai"]["dims"] == 768
    assert client.get("/v1/models/no-such-model").status_code == 404
