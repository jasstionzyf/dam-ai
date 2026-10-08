FROM gpu7.dev.yufei.com:5000/open-clip:v0.1-12.1.1-dev1.01

# dam-ai classic engine image: open-clip base already ships the exact tools
# runtime (numpy 1.26.4 / scikit-image 0.24.0 / faiss-cpu 1.9.0.post1 /
# scikit-learn 1.5.2 / pillow 11.0.0 in /data/apps/miniconda3/envs/sj) — only
# dam-ai code is added. No GPU.

COPY pyproject.toml /app/pyproject.toml
COPY classic /app/classic
COPY server /app/server
COPY registry /app/registry
COPY models /app/models
COPY deploy /app/deploy

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    DAMAI_CLASSIC_MODEL_DIR=/app/models/color

WORKDIR /app

EXPOSE 8092

# the sj conda env python is the tools-parity runtime
CMD ["/data/apps/miniconda3/envs/sj/bin/python", "-m", "classic.__main__"]
