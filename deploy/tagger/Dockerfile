FROM soujpg/qwen-vl:v0.22.0

# dam-ai tagger engine image: vLLM runtime (base image) + dam-ai code + proxy deps.
# Build (repo root): deploy/tagger/build.sh

COPY pyproject.toml /app/pyproject.toml
COPY tagger /app/tagger
COPY server /app/server
COPY registry /app/registry
COPY deploy /app/deploy

COPY deploy/tagger/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && chmod +x /app/deploy/tagger/start-vllm.sh

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

WORKDIR /app

EXPOSE 8091

# supervisord runs vllm serve + the tagger proxy (see deploy/tagger/supervisord.conf)
CMD ["/usr/bin/supervisord", "-c", "/app/deploy/tagger/supervisord.conf"]
