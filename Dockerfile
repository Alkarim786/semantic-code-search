# syntax=docker/dockerfile:1
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models CODESEARCH_DATA=/data
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY codesearch ./codesearch
RUN pip install ".[ml]"
RUN useradd -m app && mkdir -p /data /models && chown -R app /data /models
USER app
EXPOSE 8000
CMD ["uvicorn", "codesearch.api:app", "--host", "0.0.0.0", "--port", "8000"]
