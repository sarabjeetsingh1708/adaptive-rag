"""
Pipeline factory — assembles naive or adaptive pipeline from config.

Why a factory?
  Both pipelines share Embedder + VectorStore (expensive to initialise).
  The factory builds shared components once and injects them into both.
  Tests and scripts import from here — not scattered across the codebase.
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import NaiveRetriever
from src.retrieval.grader import RetrievalGrader
from src.retrieval.rewriter import QueryRewriter
from src.retrieval.web_fallback import WebSearchFallback
from src.generation.generator import GeminiGenerator
from src.generation.hallucination import HallucinationDetector
from src.pipeline import NaiveRAGPipeline
from src.adaptive_pipeline import AdaptiveRAGPipeline

load_dotenv()
logger = logging.getLogger(__name__)


def build_naive_pipeline(top_k: int = 5) -> NaiveRAGPipeline:
    """Build the Week 1 baseline pipeline."""
    logger.info("Building NaiveRAGPipeline...")
    embedder = Embedder()
    store = VectorStore()
    retriever = NaiveRetriever(embedder, store)
    generator = GeminiGenerator()
    return NaiveRAGPipeline(retriever, generator, top_k=top_k)


def build_adaptive_pipeline(top_k: int = 5) -> AdaptiveRAGPipeline:
    """
    Build the Week 2 adaptive pipeline.
    Shares Embedder + VectorStore with naive for fair comparison.
    """
    logger.info("Building AdaptiveRAGPipeline...")
    embedder = Embedder()
    store = VectorStore()
    retriever = NaiveRetriever(embedder, store)
    generator = GeminiGenerator()
    grader = RetrievalGrader()
    rewriter = QueryRewriter()
    web_search = WebSearchFallback()
    detector = HallucinationDetector()
    return AdaptiveRAGPipeline(
        retriever=retriever,
        generator=generator,
        grader=grader,
        rewriter=rewriter,
        web_search=web_search,
        detector=detector,
        top_k=top_k,
    )
