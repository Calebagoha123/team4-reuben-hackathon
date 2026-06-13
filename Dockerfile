# syntax=docker/dockerfile:1
#
# MediSnap EHR — for deploying the local MedGemma OCR path on a LINUX host with a
# CUDA GPU. Docker on macOS cannot access the Mac GPU — run locally with `uv run`
# there instead. (For the Claude OCR path you don't need the GPU image at all.)
#
# Build:  docker build -t medisnap .
# Run:    docker run --rm --gpus all -p 8000:8000 -e HF_TOKEN=hf_... medisnap
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Python (uv will install the right version into the venv)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock* ./
# This image is the on-device MedGemma path, so pull the heavy ML extra.
RUN uv sync --extra medgemma --no-install-project --no-dev

COPY app.py data.py ocr.py labs.py patients.py ./
COPY templates ./templates

ENV PATH="/app/.venv/bin:$PATH"
# Cache model weights on a mounted volume so they survive container restarts:
#   docker run ... -v hf-cache:/root/.cache/huggingface ...
ENV HF_HOME=/root/.cache/huggingface
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
