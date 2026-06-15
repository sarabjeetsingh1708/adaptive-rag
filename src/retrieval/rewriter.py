"""
Query Rewriter — Week 2, Layer 2.

When the retrieval grader finds that retrieved chunks are poor,
the rewriter transforms the original query to improve retrieval.

Strategies applied (in order):
  1. Decompose — break a complex query into a simpler core question
  2. Expand    — add relevant ML/AI terminology and synonyms
  3. Rephrase  — change sentence structure to match document vocabulary

Why this beats just retrying:
  The original query and the re-retrieved query are semantically different
  in embedding space. A rephrased query explores a different neighbourhood
  of the vector index, often surfacing relevant chunks that were missed.
"""
from __future__ import annotations

import logging
import os
import textwrap

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

REWRITER_SYSTEM = textwrap.dedent("""
    You are a query optimisation expert for a RAG system over AI/ML research papers.

    Your task: rewrite a user's query to improve retrieval from a vector database
    of ArXiv papers. The rewritten query should:
      1. Be more specific and technical
      2. Include relevant ML/AI terminology
      3. Decompose complex multi-part questions into the most retrievable core question
      4. Avoid conversational language — use the style of an academic search query

    Return ONLY the rewritten query string. No preamble, no explanation, no quotes.
""").strip()

REWRITER_USER = textwrap.dedent("""
    Original query: {query}

    Rewrite this query to maximise retrieval quality from an ArXiv AI/ML paper database:
""").strip()


# ---------------------------------------------------------------------------
# Rewriter
# ---------------------------------------------------------------------------

class QueryRewriter:
    """
    Rewrites a query when initial retrieval is poor.

    Usage:
        rewriter = QueryRewriter(api_key="...")
        better_query = rewriter.rewrite("what is rag?")
        # → "retrieval augmented generation architecture knowledge base"
    """

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=key)
        self.config = types.GenerateContentConfig(
            system_instruction=REWRITER_SYSTEM,
            temperature=0.3,       # slight creativity for diverse rewrites
            max_output_tokens=100,
        )
        logger.info("QueryRewriter ready")

    def rewrite(self, query: str) -> str:
        """Return an improved version of the query for re-retrieval."""
        prompt = REWRITER_USER.format(query=query)
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=self.config,
            )
            rewritten = response.text.strip().strip('"').strip("'")
            logger.info(f"Query rewritten:\n  original : {query}\n  rewritten: {rewritten}")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewriter failed: {e}. Using original query.")
            return query   # fail safe — use original
