"""
Shared data models (Pydantic).
Every stage of the pipeline speaks these types — no raw dicts passed around.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Paper(BaseModel):
    """Raw paper as fetched from ArXiv."""
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str
    pdf_url: str
    fetched_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class Chunk(BaseModel):
    """A text chunk ready for embedding and storage."""
    chunk_id: str                   # "{arxiv_id}_{chunk_index}"
    arxiv_id: str
    title: str
    text: str
    chunk_index: int
    total_chunks: int
    char_count: int
    source_section: str = "body"    # abstract | body


class RetrievedChunk(BaseModel):
    """A chunk returned by vector search, with its similarity score."""
    chunk: Chunk
    score: float


class QueryResult(BaseModel):
    """Full result returned to the user / API."""
    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    sources: list[str]              # arxiv_ids used
    retrieval_path: str = "direct"  # direct | rewritten | web_fallback
    hallucination_score: Optional[float] = None
    latency_ms: Optional[float] = None
