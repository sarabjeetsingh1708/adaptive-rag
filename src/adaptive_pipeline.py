"""
Adaptive RAG Pipeline — Week 2 complete.

Execution flow:
                         ┌─────────────────────┐
  query ──► embed ──► search │  Vector Retrieval  │
                         └──────────┬────────────┘
                                    │ top-k chunks
                         ┌──────────▼────────────┐
                         │  [L1] Retrieval Grader │  LLM judges relevance
                         └──────────┬────────────┘
                           ┌────────┴────────┐
                        pass             fail (< 2 relevant)
                           │                 │
                           │      ┌──────────▼────────────┐
                           │      │  [L2] Query Rewriter   │  LLM rewrites query
                           │      └──────────┬────────────┘
                           │           re-retrieve + re-grade
                           │                 │
                           │      still failing?
                           │                 │
                           │      ┌──────────▼────────────┐
                           │      │  [L3] Web Fallback     │  Tavily search
                           │      └──────────┬────────────┘
                           └────────┬────────┘
                                    │ best available context
                         ┌──────────▼────────────┐
                         │     Generation         │  Gemini 1.5 Flash
                         └──────────┬────────────┘
                                    │ answer
                         ┌──────────▼────────────┐
                         │  [L4] Hallucination    │  LLM grades groundedness
                         │       Detector         │  re-generates if score < 0.6
                         └──────────┬────────────┘
                                    │
                              QueryResult

Senior note:
  Each layer is independently testable and swappable.
  The pipeline orchestrates — it has no business logic of its own.
  Retrieval path is logged in QueryResult so we can measure how often
  each branch is taken (useful for the eval dashboard in Week 3).
"""
from __future__ import annotations

import logging
import time

from src.retrieval.retriever import NaiveRetriever
from src.retrieval.grader import RetrievalGrader
from src.retrieval.rewriter import QueryRewriter
from src.retrieval.web_fallback import WebSearchFallback
from src.generation.generator import GeminiGenerator
from src.generation.hallucination import HallucinationDetector
from src.models import QueryResult, RetrievedChunk

logger = logging.getLogger(__name__)


class AdaptiveRAGPipeline:
    """
    Self-healing RAG pipeline.
    Wraps NaiveRetriever with grading, rewriting, fallback, and hallucination checking.

    Args:
        retriever:   NaiveRetriever (embed + vector search)
        generator:   GeminiGenerator
        grader:      RetrievalGrader (LLM-as-judge for chunk relevance)
        rewriter:    QueryRewriter (LLM-based query improvement)
        web_search:  WebSearchFallback (Tavily)
        detector:    HallucinationDetector
        top_k:       chunks to retrieve per attempt
    """

    def __init__(
        self,
        retriever: NaiveRetriever,
        generator: GeminiGenerator,
        grader: RetrievalGrader,
        rewriter: QueryRewriter,
        web_search: WebSearchFallback,
        detector: HallucinationDetector,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.grader = grader
        self.rewriter = rewriter
        self.web_search = web_search
        self.detector = detector
        self.top_k = top_k

    def query(self, question: str) -> QueryResult:
        t0 = time.perf_counter()
        retrieval_path = "direct"

        # ── Layer 1: Initial retrieval ───────────────────────────────────
        retrieved = self.retriever.retrieve(question, top_k=self.top_k)
        logger.info(f"[1/4] Retrieved {len(retrieved)} chunks")

        # ── Layer 1: Retrieval grading ───────────────────────────────────
        graded = self.grader.grade_chunks(question, retrieved)
        passing_chunks: list[RetrievedChunk] = [g.retrieved for g in graded if g.passes]
        logger.info(f"[1/4] Grader: {len(passing_chunks)}/{len(graded)} chunks passed")

        # ── Layer 2: Query rewriting (if grading failed) ─────────────────
        if self.grader.needs_rerouting(graded):
            logger.info("[2/4] Retrieval poor — rewriting query...")
            retrieval_path = "rewritten"
            rewritten_query = self.rewriter.rewrite(question)

            # Retry retrieval with rewritten query
            retrieved_v2 = self.retriever.retrieve(rewritten_query, top_k=self.top_k)
            graded_v2 = self.grader.grade_chunks(question, retrieved_v2)  # grade vs ORIGINAL question
            passing_v2 = [g.retrieved for g in graded_v2 if g.passes]

            # Use whichever attempt produced more passing chunks
            if len(passing_v2) >= len(passing_chunks):
                passing_chunks = passing_v2
                logger.info(f"[2/4] Rewritten query improved results: {len(passing_chunks)} passing")
            else:
                logger.info(f"[2/4] Rewrite didn't help — keeping original {len(passing_chunks)} chunks")

        # ── Layer 3: Web search fallback (if still failing) ──────────────
        if len(passing_chunks) < self.grader.MIN_PASSING and self.web_search.available:
            logger.info("[3/4] Still insufficient context — falling back to web search...")
            retrieval_path = "web_fallback"
            web_chunks = self.web_search.search(question, max_results=5)
            if web_chunks:
                passing_chunks = web_chunks
                logger.info(f"[3/4] Web fallback returned {len(web_chunks)} results")
            else:
                logger.warning("[3/4] Web fallback also returned nothing")
                # Use whatever we have from vector search (even if poor)
                passing_chunks = [g.retrieved for g in graded] or retrieved

        elif len(passing_chunks) < self.grader.MIN_PASSING:
            # No web search available — use best available from vector search
            logger.warning("[3/4] Insufficient chunks, no web fallback — using best available")
            passing_chunks = sorted(
                [g.retrieved for g in graded],
                key=lambda r: r.score,
                reverse=True,
            )[:self.top_k]

        # ── Generation ────────────────────────────────────────────────────
        logger.info(f"[Gen] Generating with {len(passing_chunks)} context chunks...")
        answer = self.generator.generate(question, passing_chunks)

        # ── Layer 4: Hallucination detection ─────────────────────────────
        hal_result = self.detector.check(answer, passing_chunks)
        logger.info(f"[4/4] Hallucination score: {hal_result.score:.2f} ({hal_result.verdict})")

        if hal_result.regenerate:
            logger.info("[4/4] Score below threshold — regenerating with strict prompt...")
            answer = self.detector.regenerate(question, passing_chunks, self.generator)
            # Re-check after regeneration
            hal_result = self.detector.check(answer, passing_chunks)
            logger.info(f"[4/4] Post-regen score: {hal_result.score:.2f}")

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Pipeline complete | path={retrieval_path} | "
            f"hal={hal_result.score:.2f} | {latency_ms:.0f}ms"
        )

        return QueryResult(
            query=question,
            answer=answer,
            retrieved_chunks=passing_chunks,
            sources=list({rc.chunk.arxiv_id for rc in passing_chunks}),
            retrieval_path=retrieval_path,
            hallucination_score=hal_result.score,
            latency_ms=latency_ms,
        )
