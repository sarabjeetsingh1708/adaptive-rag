"""
FastAPI application — production-complete.

Endpoints:
  GET  /                   — chat + eval dashboard UI
  POST /query              — adaptive RAG query
  GET  /eval/summary       — naive vs adaptive metric comparison
  GET  /eval/history       — all saved eval runs
  GET  /eval/samples       — per-question breakdown
  GET  /eval/recent        — last N live queries (from SQLite)
  GET  /stats              — corpus + pipeline + live query stats
  GET  /health             — liveness probe
  GET  /docs               — auto-generated OpenAPI docs
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from config import EVALS_DIR, BASE_DIR
from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import NaiveRetriever
from src.retrieval.grader import RetrievalGrader
from src.retrieval.rewriter import QueryRewriter
from src.retrieval.web_fallback import WebSearchFallback
from src.generation.generator import GeminiGenerator
from src.generation.hallucination import HallucinationDetector
from src.adaptive_pipeline import AdaptiveRAGPipeline
from src.api.query_logger import init_db, log_query, get_recent, get_stats
from src.models import QueryResult

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

pipeline: Optional[AdaptiveRAGPipeline] = None
_start_time = time.time()
FRONTEND_DIR = BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    init_db()
    logger.info("Initialising Adaptive RAG pipeline...")
    embedder = Embedder()
    store   = VectorStore()
    pipeline = AdaptiveRAGPipeline(
        retriever  = NaiveRetriever(embedder, store),
        generator  = GeminiGenerator(),
        grader     = RetrievalGrader(),
        rewriter   = QueryRewriter(),
        web_search = WebSearchFallback(),
        detector   = HallucinationDetector(),
    )
    logger.info(f"Pipeline ready. Vectors indexed: {store.count()}")
    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="Adaptive RAG — ArXiv AI/ML",
    description="Self-healing RAG with retrieval grading, query rewriting, and hallucination detection.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Serve frontend ────────────────────────────────────────────────────────
if FRONTEND_DIR.exists():
    @app.get("/", include_in_schema=False)
    def root():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Schemas ───────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    retrieval_path: str
    hallucination_score: Optional[float]
    latency_ms: float
    num_chunks: int
    retrieved_chunks: list[dict]


# ── Query ─────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialised")
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    result: QueryResult = pipeline.query(req.question)
    log_query(result)

    return QueryResponse(
        question        = result.query,
        answer          = result.answer,
        sources         = result.sources,
        retrieval_path  = result.retrieval_path,
        hallucination_score = result.hallucination_score,
        latency_ms      = result.latency_ms or 0.0,
        num_chunks      = len(result.retrieved_chunks),
        retrieved_chunks = [
            {
                "title":     rc.chunk.title,
                "score":     round(rc.score, 3),
                "arxiv_id":  rc.chunk.arxiv_id,
                "text":      rc.chunk.text[:200],
                "section":   rc.chunk.source_section,
            }
            for rc in result.retrieved_chunks
        ],
    )


# ── Eval endpoints ────────────────────────────────────────────────────────
@app.get("/eval/summary")
def eval_summary():
    """Naive vs adaptive metric comparison — powers the dashboard KPIs."""
    out = {}
    for tag in ["naive", "adaptive"]:
        p = EVALS_DIR / f"summary_{tag}.json"
        if p.exists():
            out[tag] = json.loads(p.read_text())
    if not out:
        return {"message": "No eval data yet. Run scripts/run_eval.py first."}
    return out


@app.get("/eval/history")
def eval_history():
    """All saved eval summaries — for trend tracking."""
    summaries = []
    for f in sorted(EVALS_DIR.glob("summary_*.json")):
        try:
            summaries.append(json.loads(f.read_text()))
        except Exception:
            pass
    return summaries


@app.get("/eval/samples")
def eval_samples(tag: str = "adaptive"):
    """Per-question breakdown for a given eval run."""
    csv_path = EVALS_DIR / f"eval_{tag}.csv"
    if not csv_path.exists():
        raise HTTPException(404, f"No eval CSV for tag '{tag}'")
    df = pd.read_csv(csv_path)
    return df.fillna(0).to_dict(orient="records")


@app.get("/eval/recent")
def eval_recent(limit: int = 20):
    """Most recent live queries from the SQLite log."""
    return get_recent(limit=limit)


# ── Stats + health ────────────────────────────────────────────────────────
@app.get("/stats")
def stats():
    store_count = 0
    if pipeline:
        try:
            store_count = pipeline.retriever.store.count()
        except Exception:
            pass

    db_stats = get_stats()

    return {
        "vectors_indexed":    store_count,
        "embedding_model":    "all-MiniLM-L6-v2",
        "llm":                "gemini-2.0-flash",
        "pipeline":           "adaptive-rag-v3",
        "uptime_s":           round(time.time() - _start_time),
        "queries_total":      db_stats.get("total", 0),
        "avg_hallucination":  round(db_stats.get("avg_hal") or 0, 3),
        "avg_latency_ms":     round(db_stats.get("avg_lat") or 0, 1),
        "retrieval_paths": {
            "direct":       db_stats.get("direct", 0),
            "rewritten":    db_stats.get("rewritten", 0),
            "web_fallback": db_stats.get("web_fallback", 0),
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time(), "version": "3.0.0"}
