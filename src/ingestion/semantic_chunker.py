"""
Semantic chunker — splits text at meaningful topic boundaries.

Week 1 used fixed-size chunking (simple, measurable baseline).
This replaces it with sentence-level grouping that respects topic shifts —
a real engineering improvement you can speak to in interviews.

Strategy:
  1. Split into sentences
  2. Embed each sentence
  3. Compute cosine similarity between adjacent sentences
  4. Split at similarity drops > threshold (topic boundaries)
  5. Merge small fragments into surrounding chunks

Reference: "Semantic Chunking" — LlamaIndex / Greg Kamradt
"""
from __future__ import annotations

import logging
import re

import numpy as np

from config import cfg
from src.models import Paper, Chunk

logger = logging.getLogger(__name__)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping reasonable minimum length."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    merged: list[str] = []
    buf = ""
    for s in raw:
        buf = (buf + " " + s).strip() if buf else s
        if len(buf) >= 80:   # minimum sentence length
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return merged


class SemanticChunker:
    """
    Splits paper text at semantic topic boundaries using embedding similarity.

    Usage:
        chunker = SemanticChunker(embedder)
        chunks = chunker.chunk_paper(paper)

    Set breakpoint_threshold higher → fewer, larger chunks.
    Set it lower → more, smaller chunks.
    """

    def __init__(self, embedder, breakpoint_threshold: float = 0.75) -> None:
        self.embedder = embedder
        self.threshold = breakpoint_threshold

    def _find_breakpoints(self, sentences: list[str]) -> list[int]:
        """Return indices where a new chunk should start."""
        if len(sentences) <= 2:
            return []

        # Embed all sentences at once (efficient)
        vecs = self.embedder.embed_texts(sentences)   # (N, D)

        breakpoints = []
        for i in range(len(sentences) - 1):
            sim = _cosine_sim(vecs[i], vecs[i + 1])
            if sim < self.threshold:
                breakpoints.append(i + 1)   # start new chunk here
        return breakpoints

    def _build_chunks_from_splits(
        self,
        sentences: list[str],
        breakpoints: list[int],
        paper: Paper,
        offset: int = 0,
    ) -> list[Chunk]:
        """Group sentences between breakpoints into Chunk objects."""
        splits: list[list[str]] = []
        prev = 0
        for bp in breakpoints:
            splits.append(sentences[prev:bp])
            prev = bp
        splits.append(sentences[prev:])

        chunks: list[Chunk] = []
        for i, group in enumerate(splits):
            text = " ".join(group).strip()
            if len(text) < cfg.ingest.min_chunk_length:
                # Merge tiny fragments into previous chunk
                if chunks:
                    prev_chunk = chunks[-1]
                    merged_text = prev_chunk.text + " " + text
                    chunks[-1] = prev_chunk.model_copy(update={
                        "text": merged_text,
                        "char_count": len(merged_text),
                    })
                continue

            chunks.append(Chunk(
                chunk_id=f"{paper.arxiv_id}_sem_{offset + i}",
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                text=text,
                chunk_index=offset + i,
                total_chunks=0,   # updated after
                char_count=len(text),
                source_section="semantic",
            ))

        # Update total_chunks
        n = len(chunks)
        chunks = [c.model_copy(update={"total_chunks": n}) for c in chunks]
        return chunks

    def chunk_paper(self, paper: Paper) -> list[Chunk]:
        """Chunk a paper using semantic boundary detection."""
        # Always keep abstract as its own chunk
        abstract_text = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
        abstract_chunk = Chunk(
            chunk_id=f"{paper.arxiv_id}_sem_abstract",
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            text=abstract_text,
            chunk_index=0,
            total_chunks=1,
            char_count=len(abstract_text),
            source_section="abstract",
        )

        # Semantic split of body
        body = f"{paper.title}. {paper.abstract}"
        sentences = _split_sentences(body)

        if len(sentences) < 3:
            return [abstract_chunk]

        breakpoints = self._find_breakpoints(sentences)
        body_chunks = self._build_chunks_from_splits(sentences, breakpoints, paper, offset=1)

        all_chunks = [abstract_chunk] + body_chunks
        # Fix total_chunks across all
        n = len(all_chunks)
        all_chunks = [c.model_copy(update={"total_chunks": n}) for c in all_chunks]
        return all_chunks

    def chunk_papers(self, papers: list[Paper]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for paper in papers:
            try:
                all_chunks.extend(self.chunk_paper(paper))
            except Exception as e:
                logger.warning(f"SemanticChunker failed on {paper.arxiv_id}: {e}")
        logger.info(f"Semantic chunking: {len(papers)} papers → {len(all_chunks)} chunks")
        return all_chunks
