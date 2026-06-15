"""
scripts/compare_chunking.py

Compares fixed-size chunking (Week 1 baseline) vs semantic chunking (Week 4).
Re-indexes with each strategy and runs the eval set to get metric comparison.

This is an interview conversation — you made a deliberate engineering decision,
measured it, and chose the better approach. That's senior thinking.

Usage:
    python scripts/compare_chunking.py

Output:
    evals/summary_fixed_chunking.json
    evals/summary_semantic_chunking.json
    Comparison table printed to console
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.fetcher import load_all_papers
from src.ingestion.chunker import DocumentChunker
from src.ingestion.semantic_chunker import SemanticChunker
from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import NaiveRetriever
from src.generation.generator import GeminiGenerator
from src.pipeline import NaiveRAGPipeline
from src.evaluation.evaluator import RAGEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_experiment(chunker_name: str, chunker) -> dict:
    logger.info(f"\n{'='*50}")
    logger.info(f"Experiment: {chunker_name}")
    logger.info(f"{'='*50}")

    papers = load_all_papers()
    if not papers:
        logger.error("No papers found. Run scripts/ingest.py first.")
        sys.exit(1)

    embedder = Embedder()
    store = VectorStore()

    # Wipe and re-index with this chunking strategy
    logger.info(f"Re-indexing {len(papers)} papers with {chunker_name}...")
    store.delete_collection()

    chunks = chunker.chunk_papers(papers)
    vectors = embedder.embed_chunks(chunks)
    store.index(chunks, vectors)

    logger.info(f"Indexed {len(chunks)} chunks ({len(chunks)/len(papers):.1f} avg/paper)")

    # Eval
    retriever = NaiveRetriever(embedder, store)
    generator = GeminiGenerator()
    pipeline = NaiveRAGPipeline(retriever, generator)

    evaluator = RAGEvaluator()
    results = evaluator.evaluate(pipeline, delay_s=1.5)
    tag = chunker_name.lower().replace(" ", "_")
    summary = evaluator.report(results, tag=tag)
    summary["chunks_total"] = len(chunks)
    summary["avg_chunks_per_paper"] = round(len(chunks) / len(papers), 1)
    return summary


def main():
    embedder = Embedder()

    results = {}

    # Fixed chunking
    fixed = DocumentChunker()
    results["Fixed (Week 1)"] = run_experiment("fixed_chunking", fixed)

    # Semantic chunking
    semantic = SemanticChunker(embedder, breakpoint_threshold=0.78)
    results["Semantic (Week 4)"] = run_experiment("semantic_chunking", semantic)

    # Side-by-side comparison
    print("\n" + "="*65)
    print("  CHUNKING STRATEGY COMPARISON")
    print("="*65)
    print(f"  {'Metric':<25} {'Fixed':>12} {'Semantic':>12} {'Delta':>10}")
    print("-"*65)

    metrics = [
        ("faithfulness",       "Faithfulness"),
        ("answer_relevancy",   "Answer Relevancy"),
        ("context_recall",     "Context Recall"),
        ("context_precision",  "Context Precision"),
        ("overall_score",      "Overall Score"),
        ("avg_chunks_per_paper", "Avg chunks/paper"),
    ]

    for key, label in metrics:
        fixed_v = results["Fixed (Week 1)"].get(key, 0)
        sem_v   = results["Semantic (Week 4)"].get(key, 0)
        delta   = sem_v - fixed_v
        arrow   = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "─")
        print(f"  {label:<25} {fixed_v:>12.3f} {sem_v:>12.3f} {arrow} {abs(delta):>8.3f}")

    print("="*65)
    winner = "Semantic" if results["Semantic (Week 4)"]["overall_score"] > results["Fixed (Week 1)"]["overall_score"] else "Fixed"
    print(f"\n  Winner: {winner} chunking")
    print()


if __name__ == "__main__":
    main()
