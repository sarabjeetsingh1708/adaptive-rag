"""
ArXiv ingestion pipeline.

Responsibilities:
  - Fetch paper metadata from ArXiv API
  - Download abstracts (+ optionally full PDFs)
  - Deduplicate and persist to disk
  - Emit clean Paper objects downstream

Senior note: we use the abstract + title as our text corpus for now.
Full PDF parsing is in scripts/fetch_pdfs.py — keep ingestion fast.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterator

import arxiv  # arxiv python client

from config import cfg, RAW_DIR
from src.models import Paper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

class ArxivFetcher:
    """
    Pulls papers from ArXiv by category.

    Usage:
        fetcher = ArxivFetcher()
        papers = fetcher.fetch(max_results=500)
    """

    QUERIES = [
        "large language models",
        "retrieval augmented generation",
        "transformer attention mechanism",
        "instruction tuning fine-tuning",
        "chain of thought reasoning",
        "RLHF reinforcement learning human feedback",
        "vector database embedding search",
        "LLM evaluation benchmark",
    ]

    def __init__(self) -> None:
        self.client = arxiv.Client(
            page_size=50,
            delay_seconds=1.5,   # respect ArXiv rate limits
            num_retries=3,
        )
        self._seen_ids: set[str] = set()
        self._load_existing()

    def _load_existing(self) -> None:
        """Skip papers we've already fetched."""
        for f in RAW_DIR.glob("*.json"):
            self._seen_ids.add(f.stem)
        logger.info(f"Found {len(self._seen_ids)} existing papers in cache.")

    def fetch(self, max_results: int = 500) -> list[Paper]:
        """Fetch up to max_results papers, skipping duplicates."""
        papers: list[Paper] = []
        per_query = max(max_results // len(self.QUERIES), 20)

        for query in self.QUERIES:
            if len(papers) >= max_results:
                break
            logger.info(f"Fetching: '{query}' (want {per_query} papers)")
            try:
                batch = list(self._fetch_query(query, per_query))
                papers.extend(batch)
                logger.info(f"  → got {len(batch)} new papers (total: {len(papers)})")
            except Exception as e:
                logger.warning(f"  → query failed: {e}")
            time.sleep(1)

        logger.info(f"Ingestion complete. {len(papers)} new papers fetched.")
        return papers[:max_results]

    def _fetch_query(self, query: str, max_results: int) -> Iterator[Paper]:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for result in self.client.results(search):
            arxiv_id = result.entry_id.split("/")[-1]
            if arxiv_id in self._seen_ids:
                continue

            paper = Paper(
                arxiv_id=arxiv_id,
                title=result.title.strip().replace("\n", " "),
                abstract=result.summary.strip().replace("\n", " "),
                authors=[str(a) for a in result.authors[:10]],
                categories=result.categories,
                published=str(result.published.date()),
                pdf_url=result.pdf_url or "",
            )

            # Persist immediately — crash-safe incremental ingestion
            self._save(paper)
            self._seen_ids.add(arxiv_id)
            yield paper

    def _save(self, paper: Paper) -> None:
        path = RAW_DIR / f"{paper.arxiv_id}.json"
        path.write_text(paper.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Loader (reads already-fetched papers from disk)
# ---------------------------------------------------------------------------

def load_all_papers() -> list[Paper]:
    """Load every paper we've fetched so far."""
    papers = []
    for f in sorted(RAW_DIR.glob("*.json")):
        try:
            papers.append(Paper.model_validate_json(f.read_text()))
        except Exception as e:
            logger.warning(f"Skipping corrupt file {f.name}: {e}")
    logger.info(f"Loaded {len(papers)} papers from disk.")
    return papers
