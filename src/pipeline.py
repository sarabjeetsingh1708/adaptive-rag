"""
Naive RAG pipeline — Week 1 complete.

Wires: query → embed → retrieve → generate → QueryResult

This is the baseline we will measure with RAGAS.
Every adaptive feature in Week 2 wraps or extends this.

Senior note: pipeline composition lives here.
  Components (retriever, generator) are injected — not instantiated inside.
  This makes the pipeline testable and swappable.
"""
from __future__ import annotations

import logging
import time

from src.retrieval.retriever import NaiveRetriever
from src.generation.generator import GeminiGenerator
from src.models import QueryResult

logger = logging.getLogger(__name__)


class NaiveRAGPipeline:
    """
    Baseline RAG: retrieve → generate. No self-evaluation.

    Args:
        retriever: NaiveRetriever instance
        generator: GeminiGenerator instance
        top_k: number of chunks to retrieve
    """

    def __init__(
        self,
        retriever: NaiveRetriever,
        generator: GeminiGenerator,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def query(self, question: str) -> QueryResult:
        t0 = time.perf_counter()

        # Step 1: Retrieve
        retrieved = self.retriever.retrieve(question, top_k=self.top_k)
        logger.info(f"Retrieved {len(retrieved)} chunks for: '{question[:60]}'")

        # Step 2: Generate
        answer = self.generator.generate(question, retrieved)

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"Pipeline complete in {latency_ms:.0f}ms")

        return QueryResult(
            query=question,
            answer=answer,
            retrieved_chunks=retrieved,
            sources=list({rc.chunk.arxiv_id for rc in retrieved}),
            retrieval_path="direct",
            latency_ms=latency_ms,
        )
