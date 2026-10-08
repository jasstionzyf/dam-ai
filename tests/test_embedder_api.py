"""dam-ai T2 tests: embedder engine (mocked backends), /v1/embeddings contract.

Mocked-layer tests run anywhere (no torch needed — manager/lazy imports are
torch-free). Real model loading is covered by tests/test_embedder_live.py
(GPU box, marked skip-by-default).
"""

from __future__ import annotations

import base64
import importlib
import pathlib
import sys
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from registry.schema import RegistryError, load_registry  # noqa: E402
from server.app import app  # noqa: E402

REGISTRY_YAML = ROOT / "registry" / "models.yaml"

client = TestClient(app)

# a 2x2 red PNG
_PNG_2X2 = base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000002000000020802000000fdd49a730000000c4944415408d763"
    "a8a9a904000006ff014730e8e2ba0000000049454e44ae426082")).decode()
DATA_URL = f"data:image/png;base64,{_PNG_2X2}"


# ---------- fixtures: fake manager injected over the module attribute ----------

class FakeModel:
    def __init__(self, dims=8):
        self.dims = dims
        self.calls: list[list[dict[str, Any]]] = []

    def embed(self, inputs, dtype):
        self.calls.append(inputs)
        return [[0.5] * self.dims for _ in inputs]


class FakeRegistry(dict):
    pass


def _fake_spec(name="m1", modalities=("image", "text")):
    from registry.schema import ModelSpec, Preprocess

    return ModelSpec(
        name=name, path=f"/models/{name}", loader="transformers", engine="transformers",
        dims=8, modalities=tuple(modalities), normalized=True,
        preprocess=Preprocess(resolution=224, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                              pooling="cls"))


class FakeManager:
    def __init__(self):
        self.registry = {"m1": _fake_spec("m1"), "imgonly": _fake_spec("imgonly", ("image",)),
                         "txtonly": _fake_spec("txtonly", ("text",))}
        self.model = FakeModel()

    def parse_and_validate(self, model, raw_inputs):
        from embedder.engine import decode_image_url

        spec = self.registry.get(model)
        if spec is None:
            raise KeyError(model)
        out = []
        for i, item in enumerate(raw_inputs):
            if isinstance(item, str):
                if "text" not in spec.modalities:
                    raise ValueError(f"inputs[{i}]: model {model} is image-only")
                if not item.strip():
                    raise ValueError(f"inputs[{i}]: empty text")
                out.append({"text": item})
            elif isinstance(item, dict) and item.get("type") == "image_url":
                if "image" not in spec.modalities:
                    raise ValueError(f"inputs[{i}]: model {model} is text-only")
                payload, scheme = decode_image_url((item.get("image_url") or {}).get("url", ""))
                if scheme is None:
                    raise ValueError(f"inputs[{i}]: image url must be http(s) or data: URL")
                out.append({"image_bytes": payload or b"http"})
            else:
                raise ValueError(f"inputs[{i}]: unsupported")
        return out

    def embed(self, model, inputs, dtype):
        return self.model.embed(inputs, dtype)

    def snapshot(self):
        return {"models": {n: {"state": "ready", "error": None} for n in self.registry},
                "ready": True, "uptime_s": 1.0}


@pytest.fixture()
def fake_app(monkeypatch):
    from embedder import manager as manager_module

    fm = FakeManager()
    monkeypatch.setattr(manager_module, "manager", fm)
    yield fm


# ---------- registry changes vs T1 ----------

def test_registry_accepts_transformers_loader_and_loader_config():
    reg = load_registry(REGISTRY_YAML)
    assert reg["siglip-so400m"].loader == "transformers"
    assert reg["dinov2-base"].loader == "transformers"
    assert reg["qwen3-vl-embedding-2b"].loader_config["wrapper"] == \
        "qwen3_vl_embedding.Qwen3VLEmbedder"


def test_registry_new_names_match_disk_layout():
    reg = load_registry(REGISTRY_YAML)
    assert set(reg) == {"qwen3-vl-embedding-2b", "siglip-so400m", "clip-vit-l14", "dinov2-base"}
    assert reg["clip-vit-l14"].weights == "ViT-L-14.pt"


def test_bad_loader_config_refused(tmp_path):
    doc = yaml.safe_load(REGISTRY_YAML.read_text())
    doc["models"][0]["loader_config"] = "not-a-mapping"
    p = tmp_path / "bad.yaml"
    yaml.safe_dump(doc, p.open("w"))
    with pytest.raises(RegistryError, match="loader_config"):
        load_registry(p)


# ---------- /v1/embeddings happy paths (fake manager) ----------

def test_embeddings_text_string_shorthand(fake_app):
    r = client.post("/v1/embeddings", json={"model": "m1", "input": "hello world"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list" and body["model"] == "m1"
    assert len(body["data"]) == 1
    assert body["data"][0]["object"] == "embedding" and body["data"][0]["index"] == 0
    assert len(body["data"][0]["embedding"]) == 8
    assert body["dam_ai"]["dims"] == 8 and body["dam_ai"]["count"] == 1


def test_embeddings_batched_and_order(fake_app):
    r = client.post("/v1/embeddings",
                    json={"model": "m1", "input": ["a", "b", "c"], "encoding_format": "float16"})
    assert r.status_code == 200
    data = r.json()["data"]
    assert [d["index"] for d in data] == [0, 1, 2]
    assert fake_app.model.calls[-1][1] == {"text": "b"}


def test_embeddings_image_data_url(fake_app):
    r = client.post("/v1/embeddings", json={"model": "m1", "input": [
        {"type": "image_url", "image_url": {"url": DATA_URL}}]})
    assert r.status_code == 200
    assert len(r.json()["data"][0]["embedding"]) == 8


def test_embeddings_mixed_modalities_batch(fake_app):
    r = client.post("/v1/embeddings", json={"model": "m1", "input": [
        "a text", {"type": "image_url", "image_url": {"url": DATA_URL}}]})
    assert r.status_code == 200
    assert len(r.json()["data"]) == 2


# ---------- validation errors (OpenAI envelope) ----------

def _post(**kw):
    payload: dict[str, Any] = {"model": "m1", "input": "x"}
    payload.update(kw)
    return client.post("/v1/embeddings", json=payload)


def test_err_unknown_model_404():
    r = _post(model="nope")
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error" and err["code"] == "model_not_found"


def test_err_missing_model_400():
    r = client.post("/v1/embeddings", json={"input": "x"})
    assert r.status_code == 400 and "error" in r.json()


def test_err_empty_input_400(fake_app):
    assert _post(input=[]).status_code == 400
    r = _post(input="   ")
    assert r.status_code == 400
    assert "inputs[0]: empty text" in r.json()["error"]["message"]


def test_err_unsupported_input_part_400(fake_app):
    r = _post(input=[{"type": "audio_url", "audio_url": {"url": "http://x"}}])
    assert r.status_code == 400
    assert "inputs[0]" in r.json()["error"]["message"]


def test_err_text_to_image_only_model_400(fake_app):
    r = _post(model="imgonly", input="some text")
    assert r.status_code == 400 and "image-only" in r.json()["error"]["message"]


def test_err_image_to_text_only_model_400(fake_app):
    r = _post(model="txtonly",
              input=[{"type": "image_url", "image_url": {"url": DATA_URL}}])
    assert r.status_code == 400 and "text-only" in r.json()["error"]["message"]


def test_err_bad_image_scheme_400(fake_app):
    r = _post(input=[{"type": "image_url", "image_url": {"url": "ftp://x/y.png"}}])
    assert r.status_code == 400 and "inputs[0]" in r.json()["error"]["message"]


def test_err_batch_too_large_400():
    r = _post(input=["x"] * 65)
    assert r.status_code == 400 and r.json()["error"]["code"] == "batch_too_large"


def test_err_bad_encoding_format_400():
    assert _post(encoding_format="int8").status_code == 400


def test_err_model_loading_503(monkeypatch):
    from embedder import manager as manager_module

    class LoadingManager(FakeManager):
        def embed(self, model, inputs, dtype):
            from embedder.manager import ModelNotReadyError

            raise ModelNotReadyError(model, "loading", None)

    monkeypatch.setattr(manager_module, "manager", LoadingManager())
    r = _post()
    assert r.status_code == 503
    err = r.json()["error"]
    assert err["type"] == "service_unavailable" and "state=loading" in err["message"]


def test_err_model_failed_503(monkeypatch):
    from embedder import manager as manager_module

    class FailedManager(FakeManager):
        def embed(self, model, inputs, dtype):
            from embedder.manager import ModelNotReadyError

            raise ModelNotReadyError(model, "error", "RuntimeError: cuda oom")

    monkeypatch.setattr(manager_module, "manager", FailedManager())
    r = _post()
    assert r.status_code == 503
    assert "cuda oom" in r.json()["error"]["message"]


# ---------- data-url decoding ----------

def test_decode_data_url_roundtrip():
    from embedder.engine import decode_image_url

    payload, scheme = decode_image_url(DATA_URL)
    assert scheme == "data" and payload == base64.b64decode(_PNG_2X2)
    assert decode_image_url("https://x/y.jpg") == (None, "http")
    assert decode_image_url("file:///etc/passwd") == (None, None)


# ---------- server skeleton still intact (T1 regressions) ----------

def test_t1_healthz_and_models_endpoints():
    assert client.get("/healthz").status_code == 200
    r = client.get("/v1/models")
    assert r.status_code == 200 and len(r.json()["data"]) == 4
    ids = {m["id"] for m in r.json()["data"]}
    assert ids == {"qwen3-vl-embedding-2b", "siglip-so400m", "clip-vit-l14", "dinov2-base"}


def test_t1_server_imports_without_torch(monkeypatch):
    """Simulate a torch-less box: importlib reload of app with heavy deps hidden."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def blocked(name, *a, **k):
        if name.split(".")[0] in ("torch", "open_clip", "transformers", "PIL",
                                  "sentence_transformers", "qwen_vl_utils"):
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, *a, **k)

    import builtins

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(builtins, "__import__", blocked)
        for mod in list(sys.modules):
            if mod.startswith(("embedder", "server")):
                del sys.modules[mod]
        try:
            mod = importlib.import_module("server.app")
            with TestClient(mod.app) as c:
                assert c.get("/healthz").status_code == 200
                r = c.post("/v1/embeddings", json={"model": "clip-vit-l14", "input": "x"})
                assert r.status_code == 503
                assert r.json()["error"]["code"] == "model_not_ready"
        finally:
            for m in list(sys.modules):
                if m.startswith(("embedder", "server")):
                    del sys.modules[m]
            import server.app  # noqa: F401 — restore real modules for other tests
