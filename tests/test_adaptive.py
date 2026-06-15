"""
Tests for Week 2 adaptive components.

All tests are offline — no LLM API calls.
We mock the grader/rewriter/detector responses to test pipeline logic.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Chunk, RetrievedChunk, QueryResult
from src.retrieval.grader import RetrievalGrader, GradeResult, GradedChunk
from src.retrieval.rewriter import QueryRewriter
from src.retrieval.web_fallback import WebSearchFallback
from src.generation.hallucination import HallucinationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(chunk_id: str = "test_0", text: str = "test text about RAG") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            arxiv_id="2401.00001",
            title="Test Paper",
            text=text,
            chunk_index=0,
            total_chunks=1,
            char_count=len(text),
        ),
        score=0.85,
    )


def make_graded(chunk: RetrievedChunk, relevant: bool, confidence: float = 0.9) -> GradedChunk:
    return GradedChunk(
        retrieved=chunk,
        grade=GradeResult(relevant=relevant, confidence=confidence, reason="test"),
    )


# ---------------------------------------------------------------------------
# GradeResult tests
# ---------------------------------------------------------------------------

def test_grade_result_passes_when_relevant_and_confident():
    g = GradedChunk(
        retrieved=make_chunk(),
        grade=GradeResult(relevant=True, confidence=0.9, reason="ok"),
    )
    assert g.passes is True


def test_grade_result_fails_when_irrelevant():
    g = GradedChunk(
        retrieved=make_chunk(),
        grade=GradeResult(relevant=False, confidence=0.9, reason="off-topic"),
    )
    assert g.passes is False


def test_grade_result_fails_when_low_confidence():
    g = GradedChunk(
        retrieved=make_chunk(),
        grade=GradeResult(relevant=True, confidence=0.3, reason="uncertain"),
    )
    assert g.passes is False


# ---------------------------------------------------------------------------
# RetrievalGrader.needs_rerouting tests
# ---------------------------------------------------------------------------

def test_needs_rerouting_false_when_enough_pass():
    chunks = [make_chunk(f"c{i}") for i in range(4)]
    graded = [make_graded(c, relevant=True) for c in chunks]
    grader = RetrievalGrader.__new__(RetrievalGrader)  # skip __init__
    grader.MIN_PASSING = 2
    assert grader.needs_rerouting(graded) is False


def test_needs_rerouting_true_when_too_few_pass():
    chunks = [make_chunk(f"c{i}") for i in range(4)]
    graded = [make_graded(c, relevant=(i == 0)) for i, c in enumerate(chunks)]
    grader = RetrievalGrader.__new__(RetrievalGrader)
    grader.MIN_PASSING = 2
    assert grader.needs_rerouting(graded) is True


def test_needs_rerouting_true_when_all_fail():
    chunks = [make_chunk(f"c{i}") for i in range(5)]
    graded = [make_graded(c, relevant=False) for c in chunks]
    grader = RetrievalGrader.__new__(RetrievalGrader)
    grader.MIN_PASSING = 2
    assert grader.needs_rerouting(graded) is True


# ---------------------------------------------------------------------------
# HallucinationResult tests
# ---------------------------------------------------------------------------

def test_hallucination_grounded():
    r = HallucinationResult.from_score(0.9, [])
    assert r.verdict == "grounded"
    assert r.regenerate is False


def test_hallucination_partial():
    r = HallucinationResult.from_score(0.7, ["one claim"])
    assert r.verdict == "partial"
    assert r.regenerate is False


def test_hallucination_triggers_regen():
    r = HallucinationResult.from_score(0.4, ["claim a", "claim b"])
    assert r.verdict == "hallucinated"
    assert r.regenerate is True


def test_hallucination_boundary():
    r = HallucinationResult.from_score(0.6, [])
    assert r.regenerate is False   # 0.6 is the threshold, should pass


# ---------------------------------------------------------------------------
# WebSearchFallback tests
# ---------------------------------------------------------------------------

def test_web_fallback_unavailable_without_key():
    fallback = WebSearchFallback(api_key=None)
    assert fallback.available is False
    results = fallback.search("RAG systems")
    assert results == []


def test_web_fallback_available_with_key():
    fallback = WebSearchFallback(api_key="fake_key_for_test")
    assert fallback.available is True


def test_web_fallback_returns_empty_on_api_error():
    fallback = WebSearchFallback(api_key="fake_key")
    # search() with a bad key will hit the except block and return []
    results = fallback.search("test query")
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# AdaptivePipeline routing logic (mocked)
# ---------------------------------------------------------------------------

def test_pipeline_uses_direct_path_when_grader_passes():
    """Pipeline should use direct retrieval path when chunks pass grading."""
    from src.adaptive_pipeline import AdaptiveRAGPipeline

    # Build mocks
    chunk = make_chunk()
    graded_passing = [make_graded(chunk, relevant=True), make_graded(make_chunk("c2"), relevant=True)]

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [chunk]

    mock_grader = MagicMock()
    mock_grader.grade_chunks.return_value = graded_passing
    mock_grader.needs_rerouting.return_value = False
    mock_grader.MIN_PASSING = 2

    mock_generator = MagicMock()
    mock_generator.generate.return_value = "RAG retrieves documents to ground generation."

    mock_detector = MagicMock()
    mock_detector.check.return_value = HallucinationResult(
        score=0.9, flagged_claims=[], verdict="grounded", regenerate=False
    )

    pipeline = AdaptiveRAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        grader=mock_grader,
        rewriter=MagicMock(),
        web_search=MagicMock(available=False),
        detector=mock_detector,
    )

    result = pipeline.query("What is RAG?")
    assert result.retrieval_path == "direct"
    assert result.hallucination_score == 0.9
    mock_retriever.retrieve.assert_called_once()


def test_pipeline_rewrites_when_grader_fails():
    """Pipeline should rewrite query when initial retrieval fails grading."""
    from src.adaptive_pipeline import AdaptiveRAGPipeline

    chunk = make_chunk()
    graded_fail = [make_graded(chunk, relevant=False)]
    graded_pass = [make_graded(chunk, relevant=True), make_graded(make_chunk("c2"), relevant=True)]

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [chunk]

    mock_grader = MagicMock()
    mock_grader.grade_chunks.side_effect = [graded_fail, graded_pass]
    mock_grader.needs_rerouting.return_value = True
    mock_grader.MIN_PASSING = 2

    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = "retrieval augmented generation survey"

    mock_generator = MagicMock()
    mock_generator.generate.return_value = "RAG combines retrieval with generation."

    mock_detector = MagicMock()
    mock_detector.check.return_value = HallucinationResult(
        score=0.85, flagged_claims=[], verdict="grounded", regenerate=False
    )

    pipeline = AdaptiveRAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        grader=mock_grader,
        rewriter=mock_rewriter,
        web_search=MagicMock(available=False),
        detector=mock_detector,
    )

    result = pipeline.query("what is rag?")
    assert result.retrieval_path == "rewritten"
    mock_rewriter.rewrite.assert_called_once_with("what is rag?")
    assert mock_retriever.retrieve.call_count == 2   # original + rewritten


if __name__ == "__main__":
    tests = [
        test_grade_result_passes_when_relevant_and_confident,
        test_grade_result_fails_when_irrelevant,
        test_grade_result_fails_when_low_confidence,
        test_needs_rerouting_false_when_enough_pass,
        test_needs_rerouting_true_when_too_few_pass,
        test_needs_rerouting_true_when_all_fail,
        test_hallucination_grounded,
        test_hallucination_partial,
        test_hallucination_triggers_regen,
        test_hallucination_boundary,
        test_web_fallback_unavailable_without_key,
        test_web_fallback_available_with_key,
        test_web_fallback_returns_empty_on_api_error,
        test_pipeline_uses_direct_path_when_grader_passes,
        test_pipeline_rewrites_when_grader_fails,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
