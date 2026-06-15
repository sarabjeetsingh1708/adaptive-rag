"""
tests/test_api.py — Week 3 API tests.

Tests the new /eval/* endpoints and response schema
using FastAPI's test client (no live server needed).
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient


def make_mock_pipeline():
    from src.models import QueryResult, RetrievedChunk, Chunk
    mock = MagicMock()
    chunk = RetrievedChunk(
        chunk=Chunk(
            chunk_id="test_0", arxiv_id="2401.00001", title="Test Paper",
            text="RAG combines retrieval and generation.", chunk_index=0,
            total_chunks=1, char_count=40,
        ),
        score=0.88,
    )
    mock.query.return_value = QueryResult(
        query="What is RAG?",
        answer="RAG retrieves documents to ground generation, reducing hallucinations.",
        retrieved_chunks=[chunk],
        sources=["2401.00001"],
        retrieval_path="direct",
        hallucination_score=0.91,
        latency_ms=320.0,
    )
    return mock


def get_client():
    """Build test client with mocked pipeline."""
    import src.api.app as app_module

    with patch.multiple(
        "src.api.app",
        Embedder=MagicMock(), VectorStore=MagicMock(),
        NaiveRetriever=MagicMock(), GeminiGenerator=MagicMock(),
        RetrievalGrader=MagicMock(), QueryRewriter=MagicMock(),
        WebSearchFallback=MagicMock(), HallucinationDetector=MagicMock(),
        AdaptiveRAGPipeline=MagicMock(return_value=make_mock_pipeline()),
    ):
        from src.api.app import app
        app_module.pipeline = make_mock_pipeline()
        return TestClient(app)


# ── Tests ──────────────────────────────────────────────────────────────────

def test_health():
    client = get_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("  ✓ GET /health")


def test_query_returns_expected_fields():
    client = get_client()
    r = client.post("/query", json={"question": "What is RAG?"})
    assert r.status_code == 200
    data = r.json()
    for field in ["question", "answer", "sources", "retrieval_path",
                  "hallucination_score", "latency_ms", "num_chunks", "retrieved_chunks"]:
        assert field in data, f"Missing field: {field}"
    assert data["retrieval_path"] == "direct"
    assert data["hallucination_score"] == 0.91
    assert len(data["retrieved_chunks"]) == 1
    assert data["retrieved_chunks"][0]["score"] == 0.88
    print("  ✓ POST /query — all fields present")


def test_query_empty_rejected():
    client = get_client()
    r = client.post("/query", json={"question": "   "})
    assert r.status_code == 400
    print("  ✓ POST /query — empty question rejected")


def test_eval_summary_no_data():
    client = get_client()
    r = client.get("/eval/summary")
    # Returns message or empty dict when no eval files exist
    assert r.status_code == 200
    print("  ✓ GET /eval/summary — handles missing data gracefully")


def test_eval_samples_404_when_missing():
    client = get_client()
    r = client.get("/eval/samples?tag=nonexistent")
    assert r.status_code == 404
    print("  ✓ GET /eval/samples — 404 on missing tag")


def test_stats_returns_pipeline_info():
    client = get_client()
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert "pipeline" in data
    assert "embedding_model" in data
    assert "queries_total" in data
    print("  ✓ GET /stats — all fields present")


def test_frontend_served():
    """index.html should be served at root if frontend/ dir exists."""
    from pathlib import Path
    frontend = Path(__file__).parent.parent / "frontend" / "index.html"
    assert frontend.exists(), "frontend/index.html not found"
    content = frontend.read_text()
    assert "Adaptive RAG" in content
    assert "eval/summary" in content   # dashboard calls this endpoint
    print("  ✓ frontend/index.html exists and references /eval/summary")


if __name__ == "__main__":
    tests = [
        test_health,
        test_query_returns_expected_fields,
        test_query_empty_rejected,
        test_eval_summary_no_data,
        test_eval_samples_404_when_missing,
        test_stats_returns_pipeline_info,
        test_frontend_served,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
