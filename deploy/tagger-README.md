> **Why a proxy in front of vLLM?**
> vLLM owns the heavy lifting (continuous batching, guided decoding), but its
> process lifecycle and API surface are model-server specific. The thin
> `tagger` app owns service concerns: cheap `/healthz` liveness vs vLLM-aware
> `/readyz`, a `/v1/models` view annotated for the dam-ai registry, a stable
> place for future cross-cutting behavior (auth, quotas, task fan-out). Each
> dam-ai engine ships the same pattern, so upstream LB wiring is uniform
> (damai-tag -> :8091).

# T3 — tagger engine (vLLM) deployment

One compose service, **two processes in one container** (supervisord):

1. `vllm serve /models/qwen3.5-4b ...` — OpenAI-compatible server on 127.0.0.1:8101 (container-internal)
2. `python -m tagger.api` — dam-ai tagger proxy on 0.0.0.0:8091 (the service port)

## Local bring-up (gpu7:4080 shares ~4.2GB with the dinov2/siglip2/tools tenants)

```bash
# gpu7: image already present (soujpg/qwen-vl:v0.22.0, vllm 0.19.1) — tag it, no build/pull
docker tag soujpg/qwen-vl:v0.22.0 damai-tagger:v0.1

# offline env for the model mount (no hub access needed)
mkdir -p /data/projects/dam-ai/deploy/tagger-offline
cat > /data/projects/dam-ai/deploy/tagger-offline/env <<'EOF'
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
EOF

cd /data/projects/dam-ai/deploy
sudo /usr/libexec/docker/cli-plugins/docker-compose -f compose/tagger.yml up -d
```

## Layout

```
deploy/
  compose/tagger.yml          # service definition (port 8091)
  tagger/supervisord.conf     # 2-process supervision (vLLM + proxy)
  tagger/start-vllm.sh        # vLLM launcher (env-tunable, offline env sourced)
  tagger/offline.env          # HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1
  tagger/requirements.txt     # pip install -r into the runtime image
  tagger/Dockerfile           # FROM soujpg/qwen-vl:v0.22.0 + dam-ai code
  tagger/build.sh             # docker build -f tagger/Dockerfile -t damai-tagger:v0.1 ..
```

## Endpoints (proxy, :8091)

- `GET /healthz` — process up (always 200, never touches vLLM)
- `GET /readyz` — 200 once vLLM answers /v1/models, 503 while loading
- `GET /v1/models` — vLLM served models, annotated
- `POST /v1/chat/completions` — 100% OpenAI pass-through (stream + response_format/json_schema included)
- `POST /v1/tagging` — batch task-template execution (T5): `{task|prompt+schema,
  inputs[{id, image_url|images[]}], model?, task_params?, params?, concurrency?}`;
  items get per-item status + verbatim id + retry isolation; hard cap 64 inputs

## vLLM parameters (gpu7:4080, ~4.2GB shared with dinov2/siglip2/tools; qwen3.5-4b bf16 ~8.1GB)

- `--gpu-memory-utilization 0.70` — vLLM 0.19 startup check requires
  `util * total <= free-at-launch`; free is ~11.4GiB of 15.57GiB (the
  embedding tenants sit outside the torch pool), so 0.80+ is rejected outright
  and 0.55 leaves no KV room after 8.1GB weights. 0.70 -> ~2.8GB KV.
- no `--quantization fp8` — see memory: FP8 measurably hurt this model's
  vision quality; bf16
- `--max-model-len 8192` — image + caption prompts fit comfortably
- no guided-decoding backend flag needed: vLLM 0.19 auto-selects (xgrammar)

## Sibling files

- `tagger/Dockerfile`, `tagger/build.sh` — image build (only needed where the
  base image isn't already present; on gpu7 we reuse the existing image)
- `deploy/tagger/supervisord.conf` — process supervision inside the container
