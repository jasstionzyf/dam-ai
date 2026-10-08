FROM soujpg/qwen3-vl-embedding:v0.2

# dam-ai embedder deps on top of the GPU base (torch/cu129 + transformers 5.9 + qwen-vl-utils present).
# open_clip for clip-vit-l14; numpy present in base.
RUN /opt/conda/bin/pip install --no-cache-dir "open_clip_torch==3.3.0" "protobuf==6.33.0" "sentencepiece==0.2.2"

COPY . /app/dam-ai
WORKDIR /app/dam-ai
ENV DAMAI_REGISTRY=/app/dam-ai/registry/models.yaml \
    DAMAI_MODELS_ROOT= \
    PYTHONPATH=/app/dam-ai
EXPOSE 8090
CMD ["/opt/conda/bin/python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8090"]
