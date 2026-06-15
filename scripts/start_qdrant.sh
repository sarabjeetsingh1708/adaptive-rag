#!/bin/bash
# scripts/start_qdrant.sh
# Starts Qdrant vector DB locally via Docker.
# Data is persisted in ./qdrant_data so restarts don't lose your index.

set -e

echo "Starting Qdrant vector database..."

docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/qdrant_data:/qdrant/storage" \
  qdrant/qdrant:latest

echo ""
echo "✓ Qdrant running at http://localhost:6333"
echo "  Dashboard: http://localhost:6333/dashboard"
echo ""
echo "To stop: docker stop qdrant && docker rm qdrant"
