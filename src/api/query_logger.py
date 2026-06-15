"""
Query logger — persists every query result to SQLite.

Why SQLite and not just the in-memory list in app.py?
  - Survives server restarts
  - Powers the dashboard's "recent queries" feed
  - Lets you compute daily/weekly metrics post-hoc
  - One file, zero infrastructure, works on Fly.io's volume

Schema is append-only — we never update rows.
Analytics queries read from it; the pipeline only writes.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

from config import BASE_DIR
from src.models import QueryResult

logger = logging.getLogger(__name__)

DB_PATH = BASE_DIR / "data" / "queries.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create table if not exists. Call once on startup."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           REAL    NOT NULL,
                question     TEXT    NOT NULL,
                answer       TEXT    NOT NULL,
                retrieval_path TEXT  NOT NULL,
                hallucination_score REAL,
                latency_ms   REAL,
                num_chunks   INTEGER,
                sources      TEXT    -- JSON array
            )
        """)
        conn.commit()
    logger.info(f"Query DB initialised at {DB_PATH}")


def log_query(result: QueryResult) -> None:
    """Append one query result. Fire-and-forget — errors are swallowed."""
    try:
        with _get_conn() as conn:
            conn.execute("""
                INSERT INTO queries
                  (ts, question, answer, retrieval_path,
                   hallucination_score, latency_ms, num_chunks, sources)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(),
                result.query,
                result.answer,
                result.retrieval_path,
                result.hallucination_score,
                result.latency_ms,
                len(result.retrieved_chunks),
                json.dumps(result.sources),
            ))
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to log query: {e}")


def get_recent(limit: int = 20) -> list[dict]:
    """Return most recent queries for the dashboard feed."""
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM queries ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_stats() -> dict:
    """Aggregate stats for the /stats endpoint."""
    try:
        with _get_conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                        AS total,
                    AVG(hallucination_score)        AS avg_hal,
                    AVG(latency_ms)                 AS avg_lat,
                    SUM(retrieval_path='direct')    AS direct,
                    SUM(retrieval_path='rewritten') AS rewritten,
                    SUM(retrieval_path='web_fallback') AS web_fallback
                FROM queries
            """).fetchone()
        return dict(row) if row else {}
    except Exception:
        return {}
