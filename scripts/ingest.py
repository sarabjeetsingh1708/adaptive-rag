"""
scripts/ingest.py — Day 1 entrypoint.

Run this to:
  1. Fetch ArXiv papers
  2. Chunk them
  3. Embed them
  4. Store in Qdrant

Usage:
    python scripts/ingest.py --papers 200

Make sure Qdrant is running first:
    bash scripts/start_qdrant.sh
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import cfg
from src.ingestion.fetcher import ArxivFetcher, load_all_papers
from src.ingestion.chunker import DocumentChunker
from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(max_papers: int = 200, skip_fetch: bool = False) -> None:
    logger.info("=" * 60)
    logger.info("ADAPTIVE RAG — INGESTION PIPELINE")
    logger.info("=" * 60)

    # --- Step 1: Fetch papers ---
    if not skip_fetch:
        logger.info(f"\n[1/4] Fetching up to {max_papers} ArXiv papers...")
        fetcher = ArxivFetcher()
        fetcher.fetch(max_results=max_papers)
    else:
        logger.info("[1/4] Skipping fetch (--skip-fetch)")

    # --- Step 2: Load from disk ---
    logger.info("\n[2/4] Loading papers from disk...")
    papers = load_all_papers()
    if not papers:
        logger.error("No papers found. Run without --skip-fetch first.")
        sys.exit(1)

    # --- Step 3: Chunk ---
    logger.info(f"\n[3/4] Chunking {len(papers)} papers...")
    chunker = DocumentChunker()
    chunks = chunker.chunk_papers(papers)

    # --- Step 4: Embed + Index ---
    logger.info(f"\n[4/4] Embedding and indexing {len(chunks)} chunks...")
    embedder = Embedder()
    store = VectorStore()

    # Embed in one shot (efficient batch processing)
    vectors = embedder.embed_chunks(chunks)

    # Index into Qdrant
    store.index(chunks, vectors)

    logger.info("\n✓ Ingestion complete!")
    logger.info(f"  Papers  : {len(papers)}")
    logger.info(f"  Chunks  : {len(chunks)}")
    logger.info(f"  Vectors : {store.count()}")
    logger.info("\nNext: python scripts/run_eval.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", type=int, default=200)
    parser.add_argument("--skip-fetch", action="store_true")
    args = parser.parse_args()
    main(max_papers=args.papers, skip_fetch=args.skip_fetch)
