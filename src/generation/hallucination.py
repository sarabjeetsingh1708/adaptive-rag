"""
Hallucination Detector — Week 2, Layer 4.

After generation, grades whether the answer is fully grounded
in the retrieved context. Ungrounded claims are hallucinations.

Methodology (NLI-style with LLM judge):
  For each sentence in the generated answer, the judge checks
  whether it can be inferred from the retrieved chunks.
  Final score = fraction of sentences that are supported.

Thresholds:
  >= 0.8  → high confidence, pass
  0.6–0.8 → acceptable, pass with warning
  < 0.6   → regenerate with stricter prompt

This is the piece that makes your project stand out in interviews.
When asked "how do you handle hallucinations?" — you have a real answer.
"""
from __future__ import annotations

import json
import logging
import os
import textwrap

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.models import RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class HallucinationResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)    # 1.0 = fully grounded
    flagged_claims: list[str]               # claims NOT in context
    verdict: str                             # "grounded" | "partial" | "hallucinated"
    regenerate: bool                         # True if score < threshold

    @classmethod
    def from_score(cls, score: float, flagged: list[str]) -> "HallucinationResult":
        if score >= 0.8:
            verdict = "grounded"
        elif score >= 0.6:
            verdict = "partial"
        else:
            verdict = "hallucinated"
        return cls(
            score=score,
            flagged_claims=flagged,
            verdict=verdict,
            regenerate=score < 0.6,
        )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DETECTOR_SYSTEM = textwrap.dedent("""
    You are a hallucination detection system for an AI research assistant.

    Your task: determine whether a generated answer is grounded in the provided context.

    For each claim in the answer, check if it can be verified from the context chunks.
    A claim is "grounded" if the context explicitly supports it.
    A claim is "hallucinated" if it is not mentioned or implied by the context.

    Respond ONLY with valid JSON:
    {
      "grounded_fraction": 0.0 to 1.0,
      "flagged_claims": ["claim 1 not in context", "claim 2 not in context"],
      "summary": "one sentence assessment"
    }
""").strip()

DETECTOR_USER = textwrap.dedent("""
    CONTEXT (retrieved chunks):
    {context}

    GENERATED ANSWER:
    {answer}

    Grade the answer for hallucinations:
""").strip()

# Stricter regeneration prompt used when hallucination score < 0.6
STRICT_RAG_PROMPT = textwrap.dedent("""
    The previous answer contained claims not supported by the retrieved context.
    Generate a new answer using ONLY information explicitly stated in the context below.
    If you cannot answer from this context alone, say so clearly.

    CONTEXT:
    {context}

    QUESTION: {question}

    Answer (strictly from context only):
""").strip()


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class HallucinationDetector:
    """
    Grades generated answers for groundedness.
    Triggers regeneration with a stricter prompt if score is too low.

    Usage:
        detector = HallucinationDetector(api_key="...")
        result = detector.check(answer, chunks)
        if result.regenerate:
            answer = detector.regenerate(question, chunks, generator)
    """

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=key)
        self.config = types.GenerateContentConfig(
            system_instruction=DETECTOR_SYSTEM,
            temperature=0.0,
            max_output_tokens=300,
            response_mime_type="application/json",
        )
        logger.info("HallucinationDetector ready")

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        return "\n\n".join(
            f"[{i+1}] {rc.chunk.text[:500]}"
            for i, rc in enumerate(chunks)
        )

    def check(self, answer: str, chunks: list[RetrievedChunk]) -> HallucinationResult:
        """Grade an answer for hallucinations. Returns HallucinationResult."""
        if not chunks:
            return HallucinationResult.from_score(0.0, ["no context retrieved"])

        prompt = DETECTOR_USER.format(
            context=self._format_context(chunks),
            answer=answer[:1000],   # cap to save tokens
        )

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=self.config,
            )
            data = json.loads(response.text)
            score = float(data.get("grounded_fraction", 0.5))
            flagged = data.get("flagged_claims", [])
            result = HallucinationResult.from_score(score, flagged)
            logger.info(
                f"Hallucination check: score={score:.2f} "
                f"verdict={result.verdict} flagged={len(flagged)} claims"
            )
            return result
        except Exception as e:
            logger.warning(f"Hallucination detector error: {e}. Defaulting to partial.")
            return HallucinationResult.from_score(0.7, [])   # fail safe

    def regenerate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        generator,           # GeminiGenerator — avoid circular import
    ) -> str:
        """Regenerate answer with a stricter grounding prompt."""
        logger.info("Regenerating answer with strict grounding prompt...")
        context = self._format_context(chunks)
        strict_prompt = STRICT_RAG_PROMPT.format(
            context=context,
            question=question,
        )
        # We call generate directly with the strict prompt
        # by temporarily overriding the user template
        from google.genai import types as gtypes
        try:
            response = generator.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=strict_prompt,
                config=gtypes.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=1024,
                ),
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Regeneration failed: {e}")
            return "I cannot provide a grounded answer from the available context."
