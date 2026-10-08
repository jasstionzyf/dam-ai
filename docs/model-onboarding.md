# Model onboarding guide

Adding a model to dam-ai means **one entry in `registry/models.yaml`** — no code
changes. The registry is the single addressing entry (model name → on-disk
path), and preprocessing is **locked per model**: it must match the model's
training recipe, because changing it invalidates every embedding already
computed with it.

## The registry entry

Every model is a mapping under `models:` in `registry/models.yaml`. A complete,
copyable example — SigLIP2-B/16 in HF transformers format:

```yaml
models:
  - name: siglip2-base
    path: /models/siglip2-base
    loader: transformers
    engine: transformers
    dims: 768
    modalities: [image, text]
    normalized: true
    description: SigLIP2 B/16 (HF transformers format) — image via get_image_features, text via get_text_features
    weights: ""
    preprocess:
      resolution: 224
      mean: [0.5, 0.5, 0.5]
      std: [0.5, 0.5, 0.5]
      pooling: cls
```

Field reference:

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | Serving name callers put in the `model` field. Must match the on-disk directory name under `/models`. |
| `path` | yes | Container-absolute path (convention: `/models/<name>`). |
| `loader` | yes | `sentence_transformers` (ST layout, `modules.json`) \| `open_clip` (checkpoint files) \| `transformers` (plain HF layout, `AutoModel`). |
| `engine` | yes | `transformers` \| `open_clip` — which code path loads it. |
| `dims` | yes | Output dimensionality (positive int). |
| `modalities` | yes | Non-empty subset of `[image, text]`. `text`-less models (e.g. DINOv2) reject text input at the API with a 4xx. |
| `normalized` | yes | Whether outputs are L2-normalized (all current models: `true`). |
| `weights` | no | Checkpoint filename inside `path` (used by the open_clip loader, e.g. `ViT-L-14.pt`). |
| `loader_config` | no | Loader-specific extras (e.g. the qwen wrapper module, dtype, `max_pixels`). |
| `preprocess` | yes | Locked preprocessing — exactly `resolution`, `mean`, `std`, `pooling`. `pooling` ∈ `cls` \| `last_token` \| `mean`. |

## Rules the registry enforces

The server **refuses to start** on any invalid entry (load happens at import
time; failures raise and the process exits):

- unknown fields are rejected (typo = startup failure, not silent ignore);
- `loader`/`engine`/`pooling`/`modalities` values are enum-checked;
- `dims`/`resolution` positive ints, `mean`/`std` lists of exactly 3 numbers;
- no duplicate names, no duplicate modalities in one entry.

## Path resolution: containers vs host

Inside the container `/models` is the read-only mount of the model root
(`/data1/mlib_data/zhaoyufei_cache/soujpg/models/` — same path on gpu0 and
gpu7, so compose files never need per-host paths). On a host without that
mount (tests/dev), set:

```bash
DAMAI_MODELS_ROOT=/data1/mlib_data/zhaoyufei_cache/soujpg/models
```

`embedder.model_root()` then resolves `/models/<name>` →
`$DAMAI_MODELS_ROOT/<name>`. Model loading is offline: the deploy compose
services set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so a bare
directory with complete model files is all you need — no hub mirror.

## Onboarding checklist

1. Put the model directory under the model root, named exactly `name`
   (complete files: config + tokenizer/processor + weights; for the qwen
   wrapper, the `scripts/` subdirectory inside the model dir is part of the
   model — rsync it too).
2. Add the registry entry (copy the example above; copy `preprocess` from the
   model card/config — do not guess).
3. Restart the service. Startup validates the entry; a bad one refuses to
   start with the reason.
4. Verify:

```bash
curl -s localhost:8090/v1/models | python3 -m json.tool   # model listed with dims/modalities
curl -s localhost:8090/readyz                              # wait for status=ready
```

5. Re-embedding: anything you change in `preprocess` changes every vector the
   model produces — a preprocess change is a **new embedding campaign**, not a
   config tweak. Prefer onboarding a new model name and migrating.
