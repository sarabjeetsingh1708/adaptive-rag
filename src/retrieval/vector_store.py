"""
Vector store — Qdrant wrapper.

Qdrant runs locally via Docker (see scripts/start_qdrant.sh).
We wrap it here so the rest of the codebase never touches qdrant-client directly.
Swap the backend by replacing this file — nothing else changes.

Senior note: the payload stored alongside each vector IS the Chunk data.
  This means we never need a separate metadata DB for retrieval.
  For production you'd add a PostgreSQL store for analytics & logging.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
)

from config import cfg
from src.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Qdrant-backed vector store.

    Usage:
        store = VectorStore()
        store.index(chunks, vectors)
        results = store.search("what is RAG?", query_vector, top_k=5)
    """

    def __init__(self) -> None:
        self.client = QdrantClient(
            host=cfg.vector_db.host,
            port=cfg.vector_db.port,
            timeout=30,
        )
        self.collection = cfg.vector_db.collection_name
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=cfg.vector_db.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self.collection}")
        else:
            count = self.client.count(self.collection).count
            logger.info(f"Collection '{self.collection}' exists with {count} vectors.")

    def index(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """
        Upsert chunks + their vectors into Qdrant.
        Uses chunk_id as the point ID (hashed to int).
        """
        assert len(chunks) == len(vectors), "chunks and vectors must align"

        points = []
        for chunk, vector in zip(chunks, vectors):
            # Qdrant needs integer IDs — hash the string chunk_id
            point_id = abs(hash(chunk.chunk_id)) % (2**53)
            points.append(PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=chunk.model_dump(),   # store full chunk as payload
            ))

        # Batch upsert in chunks of 256
        batch_size = 256
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=self.collection, points=batch)
            logger.info(f"  Indexed {min(i+batch_size, len(points))}/{len(points)} vectors")

        logger.info(f"Indexing complete. Total vectors in collection: {self.count()}")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: Optional[int] = None,
        filter_arxiv_ids: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """
        Search for nearest neighbours.
        Optionally filter to specific arxiv_ids (useful for per-paper Q&A).
        """
        k = top_k or cfg.retrieval.top_k

        search_filter = None
        if filter_arxiv_ids:
            search_filter = Filter(
                must=[FieldCondition(
                    key="arxiv_id",
                    match=MatchValue(value=filter_arxiv_ids[0])
                )]
            )

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector[0].tolist(),
            limit=k,
            score_threshold=cfg.retrieval.score_threshold,
            query_filter=search_filter,
            search_params=SearchParams(hnsw_ef=128),   # higher = more accurate
            with_payload=True,
        )

        retrieved = []
        for r in results:
            try:
                chunk = Chunk.model_validate(r.payload)
                retrieved.append(RetrievedChunk(chunk=chunk, score=r.score))
            except Exception as e:
                logger.warning(f"Failed to parse payload: {e}")

        return retrieved

    def count(self) -> int:
        return self.client.count(self.collection).count

    def delete_collection(self) -> None:
        """Nuclear option — wipe and recreate. Useful during dev."""
        self.client.delete_collection(self.collection)
        logger.warning(f"Deleted collection: {self.collection}")
        self._ensure_collection()
