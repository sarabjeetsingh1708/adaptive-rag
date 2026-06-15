"""
Web Search Fallback — Week 2, Layer 3.

Last resort when both direct retrieval and rewritten-query retrieval
fail the grader. Calls Tavily Search API (free tier) and converts
web results into RetrievedChunk objects so the rest of the pipeline
treats them identically to vector-search results.

Design decision:
  We normalise web results into the same RetrievedChunk type used throughout.
  This means the generator, hallucination checker, and evaluator never need
  to know where the context came from. Clean abstraction boundary.

Free tier: https://tavily.com — 1000 searches/month, no card required.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import requests

from src.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

TAVILY_API_URL = "https://api.tavily.com/search"


class WebSearchFallback:
    """
    Tavily-powered web search that returns results as RetrievedChunks.

    Usage:
        fallback = WebSearchFallback(api_key="...")
        chunks = fallback.search("retrieval augmented generation survey")
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            logger.warning(
                "TAVILY_API_KEY not set — web fallback disabled. "
                "Get a free key at https://tavily.com"
            )
        else:
            logger.info("WebSearchFallback ready (Tavily)")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 5) -> list[RetrievedChunk]:
        """
        Search the web and return results as RetrievedChunk objects.
        Returns empty list if unavailable or on error.
        """
        if not self.available:
            logger.warning("Web fallback unavailable — no Tavily key.")
            return []

        # Focus search on AI/ML academic sources
        augmented_query = f"{query} machine learning research"

        try:
            response = requests.post(
                TAVILY_API_URL,
                json={
                    "api_key": self.api_key,
                    "query": augmented_query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_domains": [
                        "arxiv.org", "papers.nips.cc", "openreview.net",
                        "huggingface.co", "ai.googleblog.com", "openai.com",
                        "anthropic.com", "deepmind.com",
                    ],
                },
                timeout=10,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

        chunks: list[RetrievedChunk] = []
        for i, r in enumerate(results):
            url = r.get("url", "")
            title = r.get("title", "Web Result")
            content = r.get("content", "")

            if not content.strip():
                continue

            # Derive a pseudo arxiv_id from the URL for traceability
            pseudo_id = f"web_{i}_{url.split('/')[-1][:20]}"

            chunk = Chunk(
                chunk_id=f"{pseudo_id}_0",
                arxiv_id=pseudo_id,
                title=title,
                text=f"Source: {url}\nTitle: {title}\n\n{content[:800]}",
                chunk_index=0,
                total_chunks=1,
                char_count=len(content),
                source_section="web",
            )
            # Web results don't have a similarity score — use 0.5 as neutral
            chunks.append(RetrievedChunk(chunk=chunk, score=0.5))

        logger.info(f"Web fallback: {len(chunks)} results for '{query[:50]}'")
        return chunks
