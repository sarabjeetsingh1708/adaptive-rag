#!/bin/bash
# scripts/start_services.sh
# Used by Fly.io to start both Qdrant and the API in one container.
# For local dev use: bash scripts/start_qdrant.sh + uvicorn separately.

set -e

echo "Starting Qdrant in background..."
/usr/local/bin/qdrant &
QDRANT_PID=$!

echo "Waiting for Qdrant to be ready..."
until curl -sf http://localhost:6333/health > /dev/null 2>&1; do
  sleep 1
done
echo "Qdrant ready."

echo "Starting FastAPI..."
exec uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1
