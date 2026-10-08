# dam-ai

AI inference layer for open-source DAM (Digital Asset Management):
**multimodal tagging** (vLLM) + **multimodal embeddings** (transformers), unified OpenAI-compatible API.

## Positioning

dam-ai is the AI engine of the open-source DAM stack. Two independent inference engines behind one API:

- **tagger** — generative multimodal LLM via vLLM: captioning, metadata extraction,
  NSFW/aesthetic checks, structured outputs.
- **embedder** — feature extraction via transformers/sentence-transformers + open_clip:
  text/image embeddings for semantic search and visual dedup (CLIP-family and
  DINO-family models).

## API surface (design)

| Endpoint | Purpose |
|---|---|
| `POST /v1/embeddings` | OpenAI-compatible; `input` accepts strings and `{"type":"image_url"}` content parts |
| `GET /v1/models` | Model registry self-description: dims, modalities, normalized |
| `POST /v1/tagging` | Batch tagging with versioned task templates (prompt + json schema + inference params) |
| `POST /v1/chat/completions` | Pass-through to vLLM (escape hatch, OpenAI-compatible) |
| `POST /v1/classify` | Classical CV models (color features / palette) via image + task |
| `GET /healthz` `GET /readyz` | Liveness / model-loaded readiness |

Design principles:

1. Text-only requests are 100% OpenAI-compatible (openai SDK / LangChain work unchanged).
2. Multimodal input uses chat-style content parts (`{"type": "image_url", ...}`) — minimal extension.
3. Tagging templates are first-class: versioned prompt+schema+params locked server-side,
   stored as YAML in Git (registry/tasks.d/), with per-template model whitelists.
4. Model capabilities are declared by modality, not by model name: callers pick
   `model`, the service enforces `modalities: [text, image]` vs `[image]`.
5. Preprocessing (resolution / mean-std / pooling) is locked in the model registry —
   must match model training.

## Quick start

All engines share one server codebase; each compose service picks its engine
via `DAMAI_ENGINE` and its port. Development runs use the repo venv
(`uv venv --python 3.11 && uv pip install fastapi uvicorn httpx jinja2
jsonschema pyyaml numpy pillow scikit-image scikit-learn scipy faiss-cpu` —
see `deploy/*/requirements.txt` for the pinned set).

```bash
# 1. classic engine (CPU, no weights) — :8092
DAMAI_EMBEDDER=0 .venv/bin/python -m uvicorn server.app:app --port 8092 &
curl -s localhost:8092/healthz     # {"status":"ok", ...}

# 2. embedder engine (needs /models + GPU, or DAMAI_MODELS_ROOT) — :8090
curl -s localhost:8090/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model": "clip-vit-l14", "input": ["a red bicycle"]}'

# 3. tagger engine (co-located vLLM, see deploy/tagger-README.md) — :8091
curl -s localhost:8091/v1/tagging -H 'Content-Type: application/json' -d '{
  "task": "image_caption_metadata",
  "inputs": [{"id": "a1", "image_url": "https://example.com/a.jpg"}]}'
```

Container bring-up (the supported production path) is one compose file per
engine: `docker compose -f deploy/compose/classic.yml up -d` (same pattern for
`embedder.yml` / `tagger.yml`; engine ports 8090/8091/8092). One file for the
whole stack: `docker compose -f docker-compose.yml up -d` — classic always
starts (CPU anchor); `--profile embedder` / `--profile tagger` add the GPU
engines. On a 16GB card the embedder and tagger are mutually exclusive (see
docker-compose.yml header); a GPU-less host runs classic-only and every model
endpoint answers `503 {"error": {"code": "model_unavailable"}}` instead of
crashing (pinned by tests/test_degraded.py).

More:

- add a model → [docs/model-onboarding.md](docs/model-onboarding.md)
- write a tagging template → [docs/template-guide.md](docs/template-guide.md)
- private/business templates → [docs/external-tasks.md](docs/external-tasks.md)

## Deployment

Container + docker compose only (no host venv services). One compose service per
model; GPUs optional per service.

## License

AGPL-3.0 (aligned with the DAM open-source stack).
