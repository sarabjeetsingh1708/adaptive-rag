"""
Generation step — Gemini 1.5 Flash (free tier).

Responsibilities:
  - Format retrieved chunks into a structured prompt
  - Call Gemini API
  - Parse and return the answer

Senior note: the prompt template is a first-class engineering artifact.
  It lives here, is version-controlled, and is the first thing to tune
  when answer quality is poor. Never bury prompts in API call strings.
"""
from __future__ import annotations

import logging
import os
import textwrap

from google import genai
from google.genai import types

from config import cfg
from src.models import RetrievedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = textwrap.dedent("""
    You are a precise research assistant specializing in AI and machine learning.
    Answer questions using ONLY the provided context from ArXiv papers.

    Rules:
    1. Ground every claim in the provided context. Never hallucinate.
    2. If the context does not contain enough information, say so explicitly.
    3. Cite papers by title when referencing specific claims.
    4. Be concise but complete. Prefer 2-4 paragraphs.
    5. Use technical language appropriate for an ML researcher.
""").strip()

RAG_USER_TEMPLATE = textwrap.dedent("""
    CONTEXT (retrieved ArXiv papers):
    {context}

    ---
    QUESTION: {question}

    Answer based strictly on the context above:
""").strip()


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a readable context block."""
    parts = []
    for i, rc in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Title: {rc.chunk.title}\n"
            f"    Score: {rc.score:.3f}\n"
            f"    Text: {rc.chunk.text[:800]}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class GeminiGenerator:
    """
    Wraps Gemini 1.5 Flash for RAG generation.
    Uses the current google-genai SDK (replaces deprecated google.generativeai).

    Usage:
        gen = GeminiGenerator(api_key="...")
        answer = gen.generate(question, retrieved_chunks)
    """

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY not set. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
        self.client = genai.Client(api_key=key)
        self.model_name = cfg.generation.model
        self.config = types.GenerateContentConfig(
            system_instruction=RAG_SYSTEM_PROMPT,
            temperature=cfg.generation.temperature,
            max_output_tokens=cfg.generation.max_output_tokens,
        )
        logger.info(f"Gemini generator ready: {self.model_name}")

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Generate an answer given a question and retrieved context chunks."""
        if not chunks:
            return (
                "I could not find relevant information in the knowledge base "
                "to answer this question."
            )

        prompt = RAG_USER_TEMPLATE.format(
            context=_format_context(chunks),
            question=question,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.config,
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
