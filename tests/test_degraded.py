"""T9 degraded-mode tests: engines absent from a deployment -> OpenAI-style
503 model_unavailable, process stays up.

The degraded server is exercised in a SUBPROCESS with the heavy deps blocked
(torch/open_clip/transformers/faiss/skimage) — exactly what CI (a CPU-only
runner) and a no-GPU deployment look like. A subprocess keeps this hermetic:
re-importing server.app in-process would swap the module objects that
tests/test_embedder_api.py holds bindings to (order-dependent pollution).

Root-compose checks (profiles, env parity with the standalone files, /models
read-only mounts) run in-process — pure YAML, no docker daemon needed.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

fastapi = pytest.importorskip("fastapi")
yaml = pytest.importorskip("yaml")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_PROBE = r"""
import builtins, json, os, sys

sys.path.insert(0, {root!r})

real_import = builtins.__import__

def blocked(name, *a, **k):
    if name.split(".")[0] in ("torch", "open_clip", "transformers", "PIL",
                              "sentence_transformers", "qwen_vl_utils",
                              "faiss", "skimage"):
        raise ImportError(f"blocked in degraded-mode probe: {{name}}")
    return real_import(name, *a, **k)

builtins.__import__ = blocked
if {embedder_off}:
    os.environ["DAMAI_EMBEDDER"] = "0"

from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)
embed = client.post("/v1/embeddings", json={{"model": "clip-vit-l14", "input": "x"}})
classify_req = {{"task": "color", "inputs": [{{"id": "1", "image_url": "https://x/y.jpg"}}]}}
classify_resp = client.post("/v1/classify", json=classify_req)
out = {{
    "healthz": client.get("/healthz").json(),
    "readyz": client.get("/readyz").json(),
    "models_count": len(client.get("/v1/models").json()["data"]),
    "embeddings": embed.json(),
    "embeddings_status": embed.status_code,
    "classify_status": classify_resp.status_code,
    "classify": classify_resp.json(),
}}
print("PROBE_JSON<<" + json.dumps(out) + ">>PROBE_JSON")
"""


def _run_probe(embedder_off: bool) -> dict:
    script = _PROBE.format(root=str(ROOT), embedder_off=embedder_off)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, f"probe crashed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    line = next(x for x in r.stdout.splitlines() if x.startswith("PROBE_JSON<<"))
    return json.loads(line.removeprefix("PROBE_JSON<<").removesuffix(">>PROBE_JSON"))


# ------------------------------------------------------- degraded 503s (probe)

def test_degraded_probe_engine_imports_blocked():
    """No torch/etc: process up, healthz green, both model endpoints 503
    model_unavailable, registry listing intact."""
    out = _run_probe(embedder_off=False)
    assert out["healthz"]["status"] == "ok"
    assert out["readyz"]["status"] in ("degraded", "ok")  # reports, never crashes
    assert out["models_count"] == 4  # registry stays truthful without engines

    assert out["embeddings_status"] == 503
    err = out["embeddings"]["error"]
    assert err["code"] == "model_unavailable" and err["type"] == "model_error"

    assert out["classify_status"] == 503
    err = out["classify"]["error"]
    assert err["code"] == "model_unavailable" and err["type"] == "model_error"


def test_degraded_probe_embedder_disabled_by_env():
    """DAMAI_EMBEDDER=0 (classic-only deployment, root compose classic service):
    same contract, embedder manager never init'd."""
    out = _run_probe(embedder_off=True)
    assert out["healthz"]["status"] == "ok"
    assert out["embeddings_status"] == 503
    assert out["embeddings"]["error"]["code"] == "model_unavailable"
    assert "unavailable" in out["embeddings"]["error"]["message"]


# --------------------------------------------------------- root compose checks

def _root_compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_root_compose_has_three_engines_with_profiles():
    doc = _root_compose()
    assert set(doc["services"]) == {"classic", "embedder", "tagger"}
    # classic = degraded-mode anchor: always up, CPU-only, embedder manager off
    classic = doc["services"]["classic"]
    assert classic["environment"]["DAMAI_EMBEDDER"] == "0"
    assert "8092:8092" in classic["ports"]
    # GPU engines are opt-in profiles so a CPU box one-shot `up` never pulls them
    assert doc["services"]["embedder"]["profiles"] == ["embedder"]
    assert doc["services"]["tagger"]["profiles"] == ["tagger"]


def test_root_compose_matches_standalone_files():
    """Root file must mirror the standalone engine compose files (ports/env)."""
    for svc, standalone, port in [("classic", "classic.yml", "8092:8092"),
                                  ("embedder", "embedder.yml", "8090:8090"),
                                  ("tagger", "tagger.yml", "8091:8091")]:
        doc = yaml.safe_load((ROOT / "deploy" / "compose" / standalone)
                             .read_text(encoding="utf-8"))
        body = next(iter(doc["services"].values()))
        root = _root_compose()["services"][svc]
        assert port in (body.get("ports") or []), standalone
        assert port in (root.get("ports") or []), svc
        for key in ("DAMAI_ENGINE",):
            if key in (body.get("environment") or {}):
                assert root["environment"][key] == body["environment"][key]
        # tagger vLLM sizing must not drift between the two files
        if svc == "tagger":
            for key in ("DAMAI_VLLM_GPU_UTIL", "DAMAI_VLLM_MAX_MODEL_LEN",
                        "DAMAI_VLLM_MAX_NUM_SEQS"):
                assert root["environment"][key] == body["environment"][key]


def test_root_compose_models_mount_readonly():
    doc = _root_compose()
    for svc in ("embedder", "tagger"):
        mounts = doc["services"][svc]["volumes"]
        assert any(v.startswith("/data1/mlib_data/zhaoyufei_cache/soujpg/models")
                   and v.endswith(":ro") for v in mounts), svc
