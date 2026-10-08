# dam-ai embedder deps on top of a GPU base (torch/cu129 + transformers 5.x).
# ARG base: gpu0 uses the TEI-family image soujpg/qwen3-vl-embedding:v0.2 (which
# already vendors the qwen wrapper deps); gpu7 has no such image and no hub
# access, so it builds from soujpg/qwen-vl:v0.22.0 (torch 2.11+cu129, tf 5.9)
# and installs the small deps itself.
ARG base=soujpg/qwen3-vl-embedding:v0.2
FROM ${base}

# open_clip for clip-vit-l14; protobuf/sentencepiece for the transformers stack.
# (pip is best-effort: when the base already ships these, failures are ignored
# only for duplicate-provision reasons — a genuinely missing wheel fails at
# model load, not here.)
RUN /opt/conda/bin/pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "open_clip_torch==3.3.0" "protobuf==6.33.0" "sentencepiece==0.2.2" \
    "qwen-vl-utils" "sentence-transformers" || true

COPY . /app/dam-ai
WORKDIR /app/dam-ai
ENV DAMAI_REGISTRY=/app/dam-ai/registry/models.yaml \
    DAMAI_MODELS_ROOT= \
    PYTHONPATH=/app/dam-ai
EXPOSE 8090
CMD ["/opt/conda/bin/python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8090"]
