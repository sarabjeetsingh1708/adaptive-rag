"""
Text chunking pipeline.

Strategy (Week 1 — baseline):
  Fixed-size token chunking with overlap.
  We keep this simple intentionally so we can measure it and improve it.

Senior note: chunking strategy is one of the biggest levers in RAG quality.
  We'll experiment with semantic chunking in Week 2 once we have eval baselines.
  Never optimize what you haven't measured.
"""
from __future__ import annotations

import hashlib
import logging
import re

from config import cfg
from src.models import Paper, Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer-free word-based splitter
# (avoids heavy tokenizer dependency for baseline; swap to tiktoken later)
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def _split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into chunks of ~chunk_size words with overlap.
    Uses sentence boundaries where possible to avoid mid-sentence cuts.
    """
    # Split on sentence boundaries first
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks: list[str] = []
    current_words: list[str] = []

    for sentence in sentences:
        s_words = sentence.split()
        current_words.extend(s_words)

        if _word_count(" ".join(current_words)) >= chunk_size:
            chunk_text = " ".join(current_words)
            chunks.append(chunk_text.strip())
            # Keep last `overlap` words for context continuity
            current_words = current_words[-overlap:]

    # Don't discard the last partial chunk
    if current_words:
        remaining = " ".join(current_words).strip()
        if len(remaining) >= cfg.ingest.min_chunk_length:
            chunks.append(remaining)

    return chunks


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

class DocumentChunker:
    """
    Converts Paper objects → Chunk objects ready for embedding.

    Each paper produces:
      - 1 abstract chunk (always — abstracts are dense signal)
      - N body chunks from title + abstract (our text corpus for now)

    When we add PDF parsing (scripts/fetch_pdfs.py), body chunks
    will come from full paper text instead.
    """

    def __init__(self) -> None:
        self.chunk_size = cfg.ingest.chunk_size // 2   # word-based ≈ half token count
        self.overlap = cfg.ingest.chunk_overlap // 2

    def chunk_paper(self, paper: Paper) -> list[Chunk]:
        chunks: list[Chunk] = []

        # --- Abstract chunk (always kept whole) ---
        if len(paper.abstract) >= cfg.ingest.min_chunk_length:
            chunks.append(Chunk(
                chunk_id=f"{paper.arxiv_id}_abstract",
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                text=f"Title: {paper.title}\n\nAbstract: {paper.abstract}",
                chunk_index=0,
                total_chunks=1,    # updated below
                char_count=len(paper.abstract),
                source_section="abstract",
            ))

        # --- Body chunks (title + abstract split for longer coverage) ---
        body_text = f"{paper.title}. {paper.abstract}"
        body_splits = _split_into_chunks(body_text, self.chunk_size, self.overlap)

        for i, split in enumerate(body_splits):
            if len(split) < cfg.ingest.min_chunk_length:
                continue
            chunks.append(Chunk(
                chunk_id=f"{paper.arxiv_id}_body_{i}",
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                text=split,
                chunk_index=i,
                total_chunks=len(body_splits),
                char_count=len(split),
                source_section="body",
            ))

        # Update total_chunks on abstract chunk now we know the real count
        if chunks:
            chunks[0] = chunks[0].model_copy(update={"total_chunks": len(chunks)})

        return chunks

    def chunk_papers(self, papers: list[Paper]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for paper in papers:
            try:
                c = self.chunk_paper(paper)
                all_chunks.extend(c)
            except Exception as e:
                logger.warning(f"Failed to chunk {paper.arxiv_id}: {e}")

        logger.info(
            f"Chunked {len(papers)} papers → {len(all_chunks)} chunks "
            f"(avg {len(all_chunks)/max(len(papers),1):.1f} chunks/paper)"
        )
        return all_chunks
