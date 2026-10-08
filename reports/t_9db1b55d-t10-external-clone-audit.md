# T10 — External Clone Audit (t_9db1b55d)

Date: 2026-10-08 · Auditor: kanban worker (soujpg profile) · Host: gpu0 (dev)
Clone: `git -c http.proxy=http://127.0.0.1:11700 clone https://github.com/jasstionzyf/dam-ai.git /tmp/damai-t10`
HEAD audited: `924e47e` (same as local main tip — GitHub remote is in sync)

## Verdict

**PASS with 2 bugs found (BUG-1 functional, BUG-2 correctness).**
Every README quick-start command was live-executed from a clean clone on a
never-seen machine layout; all endpoints answered as documented. The two
bugs are both in the http-image-input path of /v1/embeddings.

## A1. Clean clone + fresh venv — PASS

- Clone from GitHub via proxy OK; tree identical to local repo (same HEAD).
- `uv venv --python 3.11` + README dep list (fastapi uvicorn httpx jinja2
  jsonschema pyyaml numpy pillow scikit-image scikit-learn scipy faiss-cpu)
  OK on first pin. NOTE for doc readers: this list is missing `pytest`
  (dev extra) and the embedder loader deps (`open_clip_torch`,
  `sentence-transformers`, `protobuf`, `sentencepiece` — the embedder
  Dockerfile pins `open_clip_torch==3.3.0`). Without them the open_clip /
  transformers loaders fail with `state=error` → 503 model_unavailable
  (degraded envelope works as designed, so nothing crashes).
- Network note: plain PyPI stalled 15+ min on faiss-cpu download from this
  host; tuna mirror + UV_HTTP_TIMEOUT=120 fixed it. Doc claim "uv pip
  install …" is correct modulo mirror choice.

## A2. README §1 classic engine :8092 — PASS

- `DAMAI_EMBEDDER=0 uvicorn server.app:app --port 8092`
- `GET /healthz` → `{"status":"ok","service":"dam-ai","version":"0.2.0"}` ✓
- `GET /readyz` → 200 ✓
- `GET /v1/models` → 4 models (qwen3-vl-embedding-2b / siglip-so400m /
  clip-vit-l14 / dinov2-base) with modalities+dims+normalized+loader ✓
- `POST /v1/embeddings` on classic → HTTP 503
  `{"error":{...,"code":"model_unavailable"}}` exactly as the README
  degraded-mode claim (pinned by tests/test_degraded.py) ✓

## A3. README §2 embedder engine — PASS (on :8093)

- Port 8090 is occupied on gpu0 by the pre-existing qwen3-vl-embedding
  container (PLAN.md baseline). Audit ran the embedder on **:8093** (same
  engine, `DAMAI_ENGINE=embedder DAMAI_MODELS_ROOT=/data1/mlib_data/
  zhaoyufei_cache/soujpg/models`). /healthz ok.
- gpu0:3090 has no free VRAM (vLLM :30000 = 13GB), so the audit ran the
  loaders with `CUDA_VISIBLE_DEVICES=""` (CPU) — PLAN.md-sanctioned
  fallback; what is verified is the API contract + real weights forward,
  not GPU latency.
- Text embedding: `clip-vit-l14` × "a red bicycle" → real 768-dim vector,
  OpenAI envelope, ~ms latency after load ✓ (first call 503 state=loading
  → then 200; retry-later semantics match T9 design)
- Image embedding via data URL: red test JPEG → 768-dim vector ✓

## A4. tagger engine on gpu7 — PASS

gpu7:4080 was already running dam-ai-embedder (7.9G) from T7/T9 residue;
README states tagger+embedder are mutually exclusive on 16G. Sequence:
stop embedder → start tagger (vLLM boot ~3min) → readyz 200 → verify →
restore embedder. Final state restored: damai-classic (healthy) +
dam-ai-embedder running, tagger stopped (clean exit).

- `GET /healthz` → `{"status":"ok","service":"dam-ai-tagger",...}` ✓
- `GET /readyz` → 503 while vLLM loads → 200 after boot ✓
- `GET /v1/models` → qwen3.5-4b annotated view ✓
- `POST /v1/tagging` task=image_caption_metadata, real image
  (red test JPEG served from gpu0 LAN) → status=ok, correct
  title/description/keywords ("Solid Red Background"…) ✓
- Batch of 2 inputs → both ok, per-item results ✓
- `POST /v1/chat/completions` pass-through → completion returned (thinking
  model; answer "pong" after `</think>`) ✓
- First tagging attempt used an https CDN image URL → vLLM upstream 500
  "Connection reset" — gpu7 egress to img.soujpg.com is blocked (known:
  no proxy in tagger env). Not a code bug; documented as deploy note:
  tagger needs either an egress route to the CDN or intra-LAN image URLs.

## A5. Test suite on the clean clone — PASS

`pytest -q` (after adding pytest): **119 passed, 6 skipped, 0 failed**
(7.87s). CI-green claim holds on a fresh machine.

## Bugs found

### BUG-1 (functional): http(s) image inputs silently dropped by open_clip and transformers loaders

- `embedder/manager.py:130` normalizes http(s) image parts to
  `{"image_url": url}`.
- `embedder/engine.py` `OpenClipModel.embed` (lines 187-190) and
  `TransformersAutoModel.embed` (132-133) only read `text` and
  `image_bytes` — no `image_url` branch. Only `Qwen3VLEmbeddingModel`
  handles it (engine.py:94-95, via qwen_vl_utils).
- Observed: `POST /v1/embeddings` with an https image part on clip-vit-l14
  → HTTP 200, `data: [], count=0, latency 0.1ms` — the input is silently
  discarded. Expected per docs/design.md (multimodal extension) and
  principle 2 of README: either fetch the URL (like the qwen wrapper does)
  or return a 4xx. Silent 200-empty is the worst of the three.

### BUG-2 (correctness): mixed text+image batches return vectors in engine-order, not input-order

- Both engines collect all texts first, then all images
  (engine.py:132-133, 145-158 / 187-200), and `server/embeddings.py:86-89`
  numbers `index` by output position.
- Observed with input `[IMG, "a red bicycle", "blue sky"]` on clip-vit-l14:
  returned `data[0]` matches "blue sky" (cos 1.000), `data[1]` matches
  "a red bicycle" (1.000), `data[2]` matches IMG (1.000) — i.e. output
  order `[text2, text1, img]` while indices claim 0,1,2. An OpenAI client
  pairing by index gets wrong vectors. text-only and image-only batches
  keep order; data-URL image-only also fine.
- Fix direction: make `parse_and_validate` preserve per-item slots and
  have each loader write results back into those slots (or reorder in
  `server/embeddings.py` using the parsed item kinds — no engine change).

## Doc nits (non-blocking)

1. README dev-dep list lacks pytest + embedder loader deps (see A1).
2. tagger image-fetch egress requirement (needs proxy or LAN URLs) is
   undocumented in tagger-README.
3. README says "embedding engines are mutually exclusive on a 16GB card"
   — true; on gpu7 the residue of both running made the first tagger boot
   OOM (vLLM start check correctly refused: "Free memory 3.68/15.57 GiB <
   desired 0.70"). After stopping embedder, boot was clean. Suggest a
   sentence in tagger-README: "if both started, `docker compose -f
   compose/embedder.yml stop` before starting tagger".

## Acceptance mapping (per card body)

| Acceptance item | Result |
|---|---|
| README quick-start executable on clean machine, every claim verified | PASS (A2-A5) |
| Found bugs graded + reproducible path | 2 bugs, both with repro in this file |
| Real call-chain verification (not mocked) | all curl/live, raw commands + outputs above |
| gpu7 state restored | yes (classic + embedder up, tagger stopped) |

## Raw command log (key excerpts)

```
$ curl -s localhost:8092/healthz
{"status":"ok","service":"dam-ai","version":"0.2.0"}
$ curl -s -o /dev/null -w '%{http_code}' localhost:8092/readyz
200
$ curl -s localhost:8092/v1/embeddings -d '{"model":"clip-vit-l14","input":["a red bicycle"]}' -w '\n%{http_code}'
{"error":{"message":"embedder engine unavailable: None","type":"model_error","code":"model_unavailable"}}
503
$ curl -s localhost:8093/v1/embeddings -d '{"model":"clip-vit-l14","input":[{"type":"image_url","image_url":{"url":"https://img.soujpg.com/image/thumbnails/..."}}]}'
{"object":"list","data":[],"model":"clip-vit-l14","usage":{...},"dam_ai":{"dims":768,"count":0,"latency_ms":0.1}}   # BUG-1
$ # mixed [IMG,"a red bicycle","blue sky"] → best-match per slot:
mix[0] best-match: blue sky (cos 1.000)   # BUG-2: should be IMG
mix[1] best-match: a red bicycle (cos 1.000)
mix[2] best-match: IMG (cos 1.000)        # should be blue sky
$ curl -s localhost:8091/v1/tagging -d '{"task":"image_caption_metadata","inputs":[{"id":"t10","image_url":"http://192.168.18.33:8123/t10img.jpg"}]}'
{"task":"image_caption_metadata","task_version":1,"model":"qwen3.5-4b","results":[{"id":"t10","status":"ok","output":{"title":"Solid Red Background",...}}]}
$ pytest -q
119 passed, 6 skipped, 8 warnings in 7.87s
```
