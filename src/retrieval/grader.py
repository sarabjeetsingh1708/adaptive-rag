"""
Retrieval Grader — Week 2, Layer 1.

An LLM judges whether each retrieved chunk is actually relevant
to the user's query before we pass anything to generation.

Why this matters:
  Vector similarity ≠ relevance. A chunk can be close in embedding space
  but not contain useful information for the specific question asked.
  The grader catches these false positives before they corrupt the answer.

Design:
  - Input:  query + chunk text
  - Output: GradeResult(relevant: bool, confidence: float, reason: str)
  - Batch grades all retrieved chunks in parallel (sequential for now, async in prod)
  - Returns only chunks that pass the relevance threshold
"""
from __future__ import annotations

import logging
import os
import textwrap
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.models import RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class GradeResult(BaseModel):
    relevant: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class GradedChunk(BaseModel):
    retrieved: RetrievedChunk
    grade: GradeResult

    @property
    def passes(self) -> bool:
        return self.grade.relevant and self.grade.confidence >= 0.5


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GRADER_SYSTEM = textwrap.dedent("""
    You are a retrieval quality grader for a RAG system over AI/ML research papers.
    Your job: decide if a retrieved text chunk is RELEVANT to a user's question.

    Relevant means: the chunk contains information that directly helps answer the question.
    Irrelevant means: the chunk is about a different topic, too vague, or off-target.

    Respond ONLY with valid JSON matching this schema:
    {
      "relevant": true | false,
      "confidence": 0.0 to 1.0,
      "reason": "one sentence explanation"
    }

    Be strict. A chunk that mentions the topic tangentially is NOT relevant.
    Only mark relevant=true if the chunk would materially help answer the question.
""").strip()

GRADER_USER = textwrap.dedent("""
    QUESTION: {question}

    RETRIEVED CHUNK:
    {chunk_text}

    Grade this chunk:
""").strip()


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class RetrievalGrader:
    """
    Uses Gemini Flash to grade whether retrieved chunks are relevant.

    Usage:
        grader = RetrievalGrader(api_key="...")
        graded = grader.grade_chunks(query, retrieved_chunks)
        passing = [g for g in graded if g.passes]
    """

    MIN_PASSING = 2   # if fewer than this pass, trigger re-routing

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=key)
        self.config = types.GenerateContentConfig(
            system_instruction=GRADER_SYSTEM,
            temperature=0.0,          # grading must be deterministic
            max_output_tokens=150,
            response_mime_type="application/json",
        )
        logger.info("RetrievalGrader ready")

    def _grade_one(self, question: str, chunk: RetrievedChunk) -> GradeResult:
        prompt = GRADER_USER.format(
            question=question,
            chunk_text=chunk.chunk.text[:600],   # cap to save tokens
        )
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=self.config,
            )
            import json
            data = json.loads(response.text)
            return GradeResult(**data)
        except Exception as e:
            logger.warning(f"Grader failed for chunk {chunk.chunk.chunk_id}: {e}")
            # Fail open — treat as relevant if grader errors
            return GradeResult(relevant=True, confidence=0.5, reason="grader error, defaulting to relevant")

    def grade_chunks(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> list[GradedChunk]:
        """Grade all chunks. Returns GradedChunk list with pass/fail on each."""
        graded: list[GradedChunk] = []
        for chunk in chunks:
            grade = self._grade_one(question, chunk)
            graded.append(GradedChunk(retrieved=chunk, grade=grade))
            logger.debug(
                f"  chunk={chunk.chunk.chunk_id[:30]} "
                f"relevant={grade.relevant} conf={grade.confidence:.2f}"
            )
        passing = sum(1 for g in graded if g.passes)
        logger.info(f"Grader: {passing}/{len(graded)} chunks passed for '{question[:50]}'")
        return graded

    def needs_rerouting(self, graded: list[GradedChunk]) -> bool:
        """True if not enough chunks passed — triggers query rewriter."""
        passing = sum(1 for g in graded if g.passes)
        return passing < self.MIN_PASSING
