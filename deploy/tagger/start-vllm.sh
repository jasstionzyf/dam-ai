#!/bin/bash
# Launch the co-located vLLM OpenAI server inside the dam-ai tagger container.
# All knobs come from environment (see deploy/compose/tagger.yml); offline env
# is forced here so a bare /models mount is enough (PLAN.md offline rule).
set -euo pipefail

# vLLM runtime python: the conda env from the base image (vllm 0.19.1, torch 2.10).
PY="${DAMAI_VLLM_PYTHON:-/data/apps/miniconda3/envs/vllm/bin/python}"

MODEL="${DAMAI_VLLM_MODEL:-/models/qwen3.5-4b}"
SERVED_NAME="${DAMAI_VLLM_SERVED_NAME:-qwen3.5-4b}"
PORT="${DAMAI_VLLM_PORT:-8101}"
GPU_UTIL="${DAMAI_VLLM_GPU_UTIL:-0.70}"
MAX_MODEL_LEN="${DAMAI_VLLM_MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${DAMAI_VLLM_MAX_NUM_SEQS:-8}"
DTYPE="${DAMAI_VLLM_DTYPE:-bfloat16}"
EXTRA_ARGS="${DAMAI_VLLM_EXTRA_ARGS:-}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL="${DAMAI_LOG_LEVEL:-info}"
# vLLM compiles kernels into ~/.cache — keep it inside the writable layer
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/.triton}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

echo "[start-vllm] model=$MODEL served_as=$SERVED_NAME port=$PORT gpu_util=$GPU_UTIL dtype=$DTYPE"

exec "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  $EXTRA_ARGS
