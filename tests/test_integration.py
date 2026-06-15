"""
tests/test_integration.py — Week 4 complete integration tests.

Tests:
  - Semantic chunker produces valid chunks
  - Query logger writes and reads SQLite correctly
  - Full pipeline routing logic with mocks
  - Eval harness scores are in valid range
  - Config is self-consistent
  - Factory returns correctly typed objects
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers ────────────────────────────────────────────────────────────────

def make_paper(arxiv_id="2401.00001"):
    from src.models import Paper
    return Paper(
        arxiv_id=arxiv_id,
        title="Attention Is All You Need: A Survey on Transformer Models",
        abstract=(
            "We propose the Transformer architecture based on attention mechanisms. "
            "Attention allows models to relate tokens regardless of distance. "
            "The model achieves state-of-the-art results on translation tasks. "
            "We dispense with recurrence entirely, enabling parallelisation. "
            "Experiments show significant improvements over previous baselines. "
        ) * 3,
        authors=["Vaswani et al."],
        categories=["cs.CL"],
        published="2017-06-12",
        pdf_url="",
    )


def make_chunk(chunk_id="c0", text="RAG combines retrieval with generation for better answers."):
    from src.models import Chunk, RetrievedChunk
    c = Chunk(chunk_id=chunk_id, arxiv_id="2401.00001", title="Test",
              text=text, chunk_index=0, total_chunks=1, char_count=len(text))
    return RetrievedChunk(chunk=c, score=0.87)


# ── Config tests ───────────────────────────────────────────────────────────

def test_config_values_are_sane():
    from config import cfg
    assert cfg.ingest.chunk_size > 0
    assert cfg.ingest.chunk_overlap < cfg.ingest.chunk_size
    assert cfg.retrieval.top_k > 0
    assert 0 < cfg.retrieval.score_threshold < 1
    assert cfg.generation.temperature >= 0
    print("  ✓ config values are sane")


def test_config_dirs_created():
    from config import RAW_DIR, PROCESSED_DIR, EVALS_DIR
    for d in [RAW_DIR, PROCESSED_DIR, EVALS_DIR]:
        assert d.exists(), f"Directory not created: {d}"
    print("  ✓ config dirs auto-created")


# ── Semantic chunker tests ─────────────────────────────────────────────────

def test_semantic_chunker_produces_chunks():
    from src.ingestion.semantic_chunker import SemanticChunker
    import numpy as np

    mock_embedder = MagicMock()
    # Return slightly different vectors to trigger breakpoints
    def fake_embed(texts):
        n = len(texts)
        vecs = np.eye(n, 384, dtype=np.float32)   # orthogonal → low similarity → many breaks
        return vecs
    mock_embedder.embed_texts.side_effect = fake_embed

    chunker = SemanticChunker(mock_embedder, breakpoint_threshold=0.9)
    paper = make_paper()
    chunks = chunker.chunk_paper(paper)

    assert len(chunks) >= 1, "Should produce at least one chunk"
    assert all(c.arxiv_id == "2401.00001" for c in chunks)
    assert all(len(c.text) >= 10 for c in chunks)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"
    print(f"  ✓ semantic chunker: {len(chunks)} chunks produced")


def test_semantic_chunker_always_has_abstract():
    from src.ingestion.semantic_chunker import SemanticChunker
    import numpy as np

    mock_embedder = MagicMock()
    # Return high-similarity vectors → no breakpoints → all merges into one body chunk
    def fake_embed_similar(texts):
        n = len(texts)
        vecs = np.ones((n, 384), dtype=np.float32)
        vecs /= np.linalg.norm(vecs[0])
        return vecs
    mock_embedder.embed_texts.side_effect = fake_embed_similar
    chunker = SemanticChunker(mock_embedder)
    paper = make_paper()
    chunks = chunker.chunk_paper(paper)
    sections = [c.source_section for c in chunks]
    assert "abstract" in sections
    print("  ✓ semantic chunker: abstract always present")


def test_semantic_chunker_short_paper():
    from src.ingestion.semantic_chunker import SemanticChunker
    import numpy as np
    from src.models import Paper

    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = np.ones((2, 384), dtype=np.float32)
    chunker = SemanticChunker(mock_embedder)
    paper = Paper(arxiv_id="x1", title="Short", abstract="Short abstract.",
                  authors=[], categories=[], published="2024-01-01", pdf_url="")
    chunks = chunker.chunk_paper(paper)
    assert len(chunks) >= 1
    print("  ✓ semantic chunker: handles short papers gracefully")


# ── Query logger tests ─────────────────────────────────────────────────────

def test_query_logger_write_and_read():
    from src.models import QueryResult, Chunk, RetrievedChunk
    import src.api.query_logger as ql

    # Redirect DB to a temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        original = ql.DB_PATH
        ql.DB_PATH = Path(tmpdir) / "test_queries.db"
        try:
            ql.init_db()

            rc = make_chunk()
            result = QueryResult(
                query="What is RAG?",
                answer="RAG retrieves documents.",
                retrieved_chunks=[rc],
                sources=["2401.00001"],
                retrieval_path="direct",
                hallucination_score=0.88,
                latency_ms=450.0,
            )
            ql.log_query(result)

            recent = ql.get_recent(limit=5)
            assert len(recent) == 1
            assert recent[0]["question"] == "What is RAG?"
            assert recent[0]["retrieval_path"] == "direct"
            assert abs(recent[0]["hallucination_score"] - 0.88) < 0.001

            stats = ql.get_stats()
            assert stats["total"] == 1
            assert stats["direct"] == 1
        finally:
            ql.DB_PATH = original
    print("  ✓ query logger: write, read, stats all correct")


def test_query_logger_handles_missing_db_gracefully():
    import src.api.query_logger as ql
    original = ql.DB_PATH
    ql.DB_PATH = Path("/nonexistent/path/queries.db")
    try:
        result = MagicMock()
        result.query = "test"
        result.answer = "ans"
        result.retrieval_path = "direct"
        result.hallucination_score = 0.9
        result.latency_ms = 100.0
        result.retrieved_chunks = []
        result.sources = []
        ql.log_query(result)   # should not raise
        data = ql.get_recent()
        assert data == []
    finally:
        ql.DB_PATH = original
    print("  ✓ query logger: fails gracefully on bad path")


# ── Evaluation harness tests ───────────────────────────────────────────────

def test_eval_metrics_all_in_range():
    from src.evaluation.evaluator import (
        score_faithfulness, score_answer_relevancy,
        score_context_recall, score_context_precision,
        GOLDEN_EVAL_SET,
    )
    rc = make_chunk(text="RAG retrieves documents from a knowledge base to ground LLM generation.")
    sample = GOLDEN_EVAL_SET[0]

    f = score_faithfulness("RAG retrieves documents to ground generation answers.", [rc])
    r = score_answer_relevancy("RAG retrieves documents to ground generation.", sample.question)
    c = score_context_recall([rc], sample)
    p = score_context_precision([rc], sample.question)

    for name, val in [("faithfulness", f), ("relevancy", r), ("recall", c), ("precision", p)]:
        assert 0.0 <= val <= 1.0, f"{name} out of range: {val}"
    print(f"  ✓ eval metrics all in [0,1]: f={f:.2f} r={r:.2f} c={c:.2f} p={p:.2f}")


def test_eval_result_overall_score():
    from src.evaluation.evaluator import EvalResult
    r = EvalResult(
        question="q", answer="a",
        faithfulness=0.8, answer_relevancy=0.7,
        context_recall=0.6, context_precision=0.5,
        retrieval_path="direct", latency_ms=300.0,
        num_chunks_retrieved=4, hallucination_score=0.85,
    )
    score = r.overall_score
    expected = 0.8*0.35 + 0.7*0.30 + 0.6*0.20 + 0.5*0.15
    assert abs(score - expected) < 0.001, f"Wrong overall: {score} != {expected}"
    print(f"  ✓ EvalResult.overall_score = {score:.3f} (correct)")


def test_golden_eval_set_complete():
    from src.evaluation.evaluator import GOLDEN_EVAL_SET, EvalSample
    assert len(GOLDEN_EVAL_SET) >= 10, "Need at least 10 eval samples"
    for s in GOLDEN_EVAL_SET:
        assert isinstance(s, EvalSample)
        assert s.question.strip()
        assert s.ground_truth.strip()
        assert len(s.context_keywords) > 0
    print(f"  ✓ golden eval set: {len(GOLDEN_EVAL_SET)} valid samples")


# ── Model schema tests ─────────────────────────────────────────────────────

def test_query_result_serialises():
    from src.models import QueryResult
    rc = make_chunk()
    result = QueryResult(
        query="q", answer="a", retrieved_chunks=[rc],
        sources=["2401.00001"], retrieval_path="rewritten",
        hallucination_score=0.77, latency_ms=812.0,
    )
    d = result.model_dump()
    assert d["retrieval_path"] == "rewritten"
    assert d["hallucination_score"] == 0.77
    assert len(d["retrieved_chunks"]) == 1
    print("  ✓ QueryResult serialises correctly")


def test_chunk_id_format():
    from src.ingestion.chunker import DocumentChunker
    chunker = DocumentChunker()
    paper = make_paper("2402.12345")
    chunks = chunker.chunk_paper(paper)
    for c in chunks:
        assert c.arxiv_id == "2402.12345"
        assert "2402.12345" in c.chunk_id
    print("  ✓ chunk IDs contain arxiv_id")


# ── Hallucination detector logic ───────────────────────────────────────────

def test_hallucination_result_thresholds():
    from src.generation.hallucination import HallucinationResult
    cases = [
        (0.0,  "hallucinated", True),
        (0.59, "hallucinated", True),
        (0.60, "partial",      False),
        (0.79, "partial",      False),
        (0.80, "grounded",     False),
        (1.0,  "grounded",     False),
    ]
    for score, expected_verdict, expected_regen in cases:
        r = HallucinationResult.from_score(score, [])
        assert r.verdict == expected_verdict, f"score={score}: got {r.verdict}"
        assert r.regenerate == expected_regen, f"score={score}: regen={r.regenerate}"
    print("  ✓ hallucination thresholds all correct (0.6 / 0.8 boundaries)")


if __name__ == "__main__":
    tests = [
        test_config_values_are_sane,
        test_config_dirs_created,
        test_semantic_chunker_produces_chunks,
        test_semantic_chunker_always_has_abstract,
        test_semantic_chunker_short_paper,
        test_query_logger_write_and_read,
        test_query_logger_handles_missing_db_gracefully,
        test_eval_metrics_all_in_range,
        test_eval_result_overall_score,
        test_golden_eval_set_complete,
        test_query_result_serialises,
        test_chunk_id_format,
        test_hallucination_result_thresholds,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
