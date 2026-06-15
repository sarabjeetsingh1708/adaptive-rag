"""
scripts/run_eval.py — Day 4-5 entrypoint.

Runs the golden eval set against the naive RAG pipeline
and saves baseline scores. This is your "before" photo.

Usage:
    python scripts/run_eval.py --tag naive

Output:
    evals/eval_naive.csv
    evals/summary_naive.json
"""
import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import NaiveRetriever
from src.generation.generator import GeminiGenerator
from src.pipeline import NaiveRAGPipeline
from src.evaluation.evaluator import RAGEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(tag: str = "naive") -> None:
    logger.info("=" * 60)
    logger.info(f"ADAPTIVE RAG — EVALUATION ({tag.upper()})")
    logger.info("=" * 60)

    # Build pipeline
    embedder = Embedder()
    store = VectorStore()

    count = store.count()
    if count == 0:
        logger.error("Vector store is empty. Run scripts/ingest.py first.")
        sys.exit(1)
    logger.info(f"Vector store has {count} vectors.")

    retriever = NaiveRetriever(embedder, store)
    generator = GeminiGenerator()
    pipeline = NaiveRAGPipeline(retriever, generator)

    # Run eval
    evaluator = RAGEvaluator()
    results = evaluator.evaluate(pipeline, delay_s=1.5)  # respect Gemini rate limit
    summary = evaluator.report(results, tag=tag)

    logger.info("\nBaseline scores saved. Improve these in Week 2.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="naive", help="Label for this eval run")
    args = parser.parse_args()
    main(tag=args.tag)
