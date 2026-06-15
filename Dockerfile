# Production Dockerfile — Fly.io single-container deploy
# Packages both Qdrant and FastAPI together.
# Local dev: run Qdrant via Docker (scripts/start_qdrant.sh) + uvicorn separately.

FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tar && \
    rm -rf /var/lib/apt/lists/*

# Download Qdrant binary (musl = no glibc dependency)
RUN curl -L \
    https://github.com/qdrant/qdrant/releases/download/v1.9.1/qdrant-x86_64-unknown-linux-musl.tar.gz \
    | tar -xz -C /usr/local/bin/ && \
    chmod +x /usr/local/bin/qdrant

WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY . .
RUN mkdir -p data/raw data/processed evals qdrant_data frontend && \
    chmod +x scripts/start_services.sh

EXPOSE 8000 6333

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["bash", "scripts/start_services.sh"]
