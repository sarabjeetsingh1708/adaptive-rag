"""
scripts/run_eval_adaptive.py — Week 2 eval entrypoint.

Runs the golden eval set through AdaptiveRAGPipeline and saves
scores to evals/summary_adaptive.json.

Then run compare_evals.py to see the improvement.

Usage:
    python scripts/run_eval_adaptive.py
"""
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.factory import build_adaptive_pipeline
from src.evaluation.evaluator import RAGEvaluator
from src.retrieval.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 60)
    logger.info("ADAPTIVE RAG — WEEK 2 EVALUATION")
    logger.info("=" * 60)

    # Sanity check
    store = VectorStore()
    count = store.count()
    if count == 0:
        logger.error("Vector store is empty. Run scripts/ingest.py first.")
        sys.exit(1)
    logger.info(f"Vector store: {count} vectors indexed.")

    pipeline = build_adaptive_pipeline()
    evaluator = RAGEvaluator()

    logger.info(
        f"Running {len(evaluator.eval_set)} questions through adaptive pipeline..."
        "\nThis will make ~5 Gemini calls per question (retrieve, grade, generate, "
        "hallucination check). Expect ~3 mins total."
    )

    results = evaluator.evaluate(pipeline, delay_s=2.0)  # longer delay: more API calls
    evaluator.report(results, tag="adaptive")

    logger.info("\nDone. Now run: python scripts/compare_evals.py")


if __name__ == "__main__":
    main()
