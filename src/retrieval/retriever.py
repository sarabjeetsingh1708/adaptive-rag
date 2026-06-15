"""
Naive retriever — Week 1 baseline.

Straight vector search: query → embed → top-k chunks.
No grading. No re-routing. No fallback.
This is our control group. We measure it, then beat it.

Senior note: keeping naive RAG clean and separate means we can
  A/B test it against adaptive RAG at any time without code conflicts.
"""
from __future__ import annotations

import logging

from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.models import RetrievedChunk

logger = logging.getLogger(__name__)


class NaiveRetriever:
    """
    Baseline retriever: embed query → cosine search → return top-k chunks.
    No self-evaluation. No re-routing. Measures our floor.
    """

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        logger.debug(f"[NaiveRetriever] query='{query[:60]}...' top_k={top_k}")
        query_vector = self.embedder.embed_query(query)
        results = self.store.search(query_vector, top_k=top_k)
        logger.debug(f"[NaiveRetriever] retrieved {len(results)} chunks")
        return results
