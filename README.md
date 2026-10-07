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

## Deployment

Container + docker compose only (no host venv services). One compose service per
model; GPUs optional per service.

## License

AGPL-3.0 (aligned with the DAM open-source stack).
