"""dam-ai T2 live-model integration tests (GPU box / real weights).

Run explicitly: pytest tests/test_embedder_live.py -v
Skipped automatically when torch is unavailable or DAMAI_MODELS_ROOT is unset
(keep CI green on torch-less runners).
"""

from __future__ import annotations

import base64
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS_ROOT = os.environ.get("DAMAI_MODELS_ROOT",
                             "/data1/mlib_data/zhaoyufei_cache/soujpg/models")

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("DAMAI_LIVE") != "1", reason="live model tests opt-in (DAMAI_LIVE=1)"),
]

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

pytestmark.append(pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed"))


@pytest.fixture(scope="module")
def live_client():
    os.environ["DAMAI_MODELS_ROOT"] = MODELS_ROOT
    for mod in list(sys.modules):
        if mod.startswith(("embedder", "server")):
            del sys.modules[mod]
    from embedder.manager import init_manager

    init_manager()
    from fastapi.testclient import TestClient

    from server.app import app

    with TestClient(app) as c:
        yield c


def _wait_ready(client, names, timeout_s=600):
    import time

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = client.get("/readyz").json()
        states = {n: s["state"] for n, s in r.get("models", {}).items()}
        if all(states.get(n) == "ready" for n in names):
            return states
        if any(states.get(n) == "error" for n in names):
            pytest.fail(f"models failed to load: { {n: s for n, s in r['models'].items()} }")
        time.sleep(2)
    pytest.fail(f"models not ready after {timeout_s}s: {states}")


# a 1x1 white PNG
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU"
    "5ErkJggg==")


def test_all_four_models_load_and_embed(live_client):
    states = _wait_ready(live_client,
                         ["qwen3-vl-embedding-2b", "siglip-so400m", "clip-vit-l14", "dinov2-base"])
    assert all(v == "ready" for v in states.values())


def test_clip_text_image_dims(live_client):
    r = live_client.post("/v1/embeddings", json={"model": "clip-vit-l14", "input": "a red apple"})
    assert r.status_code == 200
    assert len(r.json()["data"][0]["embedding"]) == 768


def test_siglip_text_image_dims(live_client):
    r = live_client.post("/v1/embeddings", json={
        "model": "siglip-so400m",
        "input": ["a red apple", {"type": "image_url",
                                  "image_url": {"url": "data:image/png;base64,"
                                                        + base64.b64encode(_PNG).decode()}}]})
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data) == 2 and all(len(d["embedding"]) == 1152 for d in data)


def test_dinov2_image_only(live_client):
    url = "data:image/png;base64," + base64.b64encode(_PNG).decode()
    r = live_client.post("/v1/embeddings", json={
        "model": "dinov2-base",
        "input": [{"type": "image_url", "image_url": {"url": url}}]})
    assert r.status_code == 200
    assert len(r.json()["data"][0]["embedding"]) == 768
    # text to an image-only model must 400
    assert live_client.post("/v1/embeddings",
                            json={"model": "dinov2-base", "input": "text"}).status_code == 400


def test_qwen3vl_text_and_image(live_client):
    url = "data:image/png;base64," + base64.b64encode(_PNG).decode()
    r = live_client.post("/v1/embeddings", json={
        "model": "qwen3-vl-embedding-2b",
        "input": ["a red apple", {"type": "image_url", "image_url": {"url": url}}]})
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data) == 2 and all(len(d["embedding"]) == 2048 for d in data)


def test_embeddings_are_normalized(live_client):
    r = live_client.post("/v1/embeddings", json={"model": "clip-vit-l14", "input": "a red apple"})
    vec = r.json()["data"][0]["embedding"]
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-2
