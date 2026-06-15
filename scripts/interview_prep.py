"""
scripts/interview_prep.py

Generates interview questions about YOUR specific implementation,
then lets you answer them interactively using Claude/Gemini.

Covers:
  - System design (whiteboard the full pipeline)
  - Technical depth (why each decision was made)
  - Tradeoffs (what you'd change with more time/resources)
  - ML fundamentals (RAGAS, embedding models, chunking)
  - Production thinking (cost, latency, monitoring)

Usage:
    python scripts/interview_prep.py              # full drill
    python scripts/interview_prep.py --category design
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

QUESTIONS = {
    "design": [
        {
            "q": "Walk me through your RAG pipeline end-to-end, as if whiteboarding it for a senior engineer.",
            "hint": "Cover: query → embed → retrieve → grade → [rewrite/fallback] → generate → hallucination check → return. Mention the path logging.",
            "strong_answer_signals": [
                "mentions retrieval_path field on QueryResult",
                "explains why grading happens BEFORE generation",
                "knows the exact threshold (MIN_PASSING=2)",
                "can draw the full flow from memory",
            ]
        },
        {
            "q": "Why did you choose Qdrant over Pinecone or Weaviate?",
            "hint": "Self-hostable, free, HNSW with tunable ef parameter, no index size limits on free tier, Docker-native.",
            "strong_answer_signals": [
                "mentions hnsw_ef=128 tuning",
                "compares cost at scale",
                "mentions Cosine vs Dot Product distance choice",
            ]
        },
        {
            "q": "How would you scale this to 10 million documents?",
            "hint": "Qdrant sharding, async batch embedding, pre-filtering by category, approximate search tradeoffs, caching layer.",
            "strong_answer_signals": [
                "mentions distributed Qdrant",
                "mentions embedding cache",
                "discusses approximate vs exact search tradeoff",
                "mentions pre-filtering to reduce search space",
            ]
        },
        {
            "q": "Your system has a fallback to web search. How do you ensure web results don't degrade answer quality?",
            "hint": "Domain allowlist (arxiv.org, openreview.net etc), same hallucination detector runs on web-sourced answers, web chunks score 0.5 neutral (not boosted).",
            "strong_answer_signals": [
                "mentions domain allowlist",
                "explains that web chunks get the same hallucination check",
                "knows the neutral score (0.5) assigned to web results",
            ]
        },
    ],
    "tradeoffs": [
        {
            "q": "Your adaptive pipeline is ~52% slower than naive RAG. How would you optimise it?",
            "hint": "Async grading in parallel with generation, cache grader results for repeated queries, batch grading, lighter grader model.",
            "strong_answer_signals": [
                "mentions parallel/async execution",
                "mentions caching grader results",
                "quantifies the overhead per layer",
                "proposes a lighter model for grading (Flash vs Pro)",
            ]
        },
        {
            "q": "When would you choose fine-tuning over RAG?",
            "hint": "RAG for dynamic/frequently updated knowledge, FT for style/format/domain adaptation, FT+RAG for both. RAG wins on auditability.",
            "strong_answer_signals": [
                "mentions knowledge staleness",
                "mentions auditability of RAG",
                "mentions combined FT+RAG approach",
                "discusses cost of maintaining a fine-tuned model",
            ]
        },
        {
            "q": "Your chunking strategy changed from fixed-size to semantic. What did you measure to justify the switch?",
            "hint": "Re-ran RAGAS eval with both strategies, compared faithfulness/recall, showed semantic chunking improved context recall by X%.",
            "strong_answer_signals": [
                "ran scripts/compare_chunking.py",
                "cites specific metric improvement",
                "explains what semantic chunking detects (cosine similarity drops)",
                "mentions the breakpoint threshold as a hyperparameter",
            ]
        },
        {
            "q": "How do you handle the case where Gemini's free tier rate limits you in production?",
            "hint": "Exponential backoff in generator, queue with Redis, multiple API keys rotation, fallback to smaller model.",
            "strong_answer_signals": [
                "mentions retry with backoff",
                "mentions request queuing",
                "discusses model fallback (Flash → lite)",
            ]
        },
    ],
    "ml": [
        {
            "q": "Explain RAGAS faithfulness score. How does it differ from accuracy?",
            "hint": "Faithfulness = fraction of claims in the answer supported by retrieved context. Accuracy requires knowing ground truth. Faithfulness is unsupervised.",
            "strong_answer_signals": [
                "explains claim decomposition",
                "distinguishes from accuracy (no ground truth needed)",
                "mentions NLI framing",
                "explains why it's better than BLEU/ROUGE for RAG",
            ]
        },
        {
            "q": "Why did you use all-MiniLM-L6-v2 and not OpenAI text-embedding-3-small?",
            "hint": "Free, runs on CPU, 5x faster batch throughput, 384 vs 1536 dims (cheaper to store/search), quality sufficient for this corpus size.",
            "strong_answer_signals": [
                "compares dimensionality (384 vs 1536)",
                "discusses batch throughput",
                "mentions MTEB benchmark",
                "explains when you'd upgrade (larger corpus, higher precision requirement)",
            ]
        },
        {
            "q": "What is the LLM-as-judge pattern and what are its failure modes?",
            "hint": "Using an LLM to evaluate another LLM's output. Failure modes: self-preference bias, position bias, verbosity bias, inconsistency.",
            "strong_answer_signals": [
                "names at least 2 bias types",
                "mentions using temperature=0 for determinism",
                "mentions using a different model as judge than generator",
                "mentions calibration against human labels",
            ]
        },
        {
            "q": "How does cosine similarity work in your vector search and why not dot product?",
            "hint": "Cosine = normalised dot product, invariant to vector magnitude. You normalise embeddings at encode time, making cosine == dot product but safer.",
            "strong_answer_signals": [
                "knows you set normalize_embeddings=True in embedder",
                "explains magnitude invariance",
                "mentions Qdrant Distance.COSINE setting",
            ]
        },
    ],
    "production": [
        {
            "q": "How do you monitor this system in production?",
            "hint": "SQLite query log → /stats endpoint → eval dashboard. Key metrics: avg hallucination score, retrieval path distribution, p95 latency, daily query volume.",
            "strong_answer_signals": [
                "mentions the SQLite query_logger",
                "mentions retrieval path distribution as a health signal",
                "proposes latency SLOs",
                "mentions eval re-runs on schedule",
            ]
        },
        {
            "q": "A user complains the system gave wrong information. How do you debug it?",
            "hint": "Check query_log for retrieval_path + hallucination_score. Re-run query manually. Check retrieved chunks. Check grader output. Check if answer was flagged for regeneration.",
            "strong_answer_signals": [
                "goes to query_log first",
                "checks hallucination_score",
                "knows how to re-run individual components",
                "proposes adding verbose logging mode",
            ]
        },
        {
            "q": "What's your cost per query in production?",
            "hint": "Gemini Flash free tier = 0 up to limits. Estimate: ~5 Gemini calls/query (grade chunks + rewrite + generate + hal check), each ~500 tokens = ~2500 tokens/query.",
            "strong_answer_signals": [
                "can estimate token count per query",
                "knows which calls are the most expensive",
                "discusses caching to reduce cost",
                "mentions free tier limits",
            ]
        },
    ],
}


def print_question(q_obj: dict, num: int, total: int) -> None:
    print(f"\n{'─'*60}")
    print(f"  Question {num}/{total}")
    print(f"{'─'*60}")
    print(f"\n  {q_obj['q']}\n")


def print_hint(q_obj: dict) -> None:
    print(f"\n  💡 Hint: {q_obj['hint']}\n")
    print("  Strong answer signals:")
    for s in q_obj["strong_answer_signals"]:
        print(f"    • {s}")


def drill(category: str = "all") -> None:
    all_q = []
    if category == "all":
        for cat_qs in QUESTIONS.values():
            all_q.extend(cat_qs)
    else:
        all_q = QUESTIONS.get(category, [])
        if not all_q:
            print(f"Unknown category: {category}. Choose from: {list(QUESTIONS.keys())}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  ADAPTIVE RAG — INTERVIEW DRILL")
    print(f"  {len(all_q)} questions | category: {category}")
    print(f"{'='*60}")
    print("\n  For each question:")
    print("  - Answer out loud (or type your answer)")
    print("  - Press ENTER to see hints + strong answer signals")
    print("  - Press ENTER again to move on\n")

    for i, q_obj in enumerate(all_q, 1):
        print_question(q_obj, i, len(all_q))
        input("  [Press ENTER when ready to answer, then ENTER again for hints]")
        your_answer = input("  Your answer (or press ENTER to skip): ").strip()
        print_hint(q_obj)
        input("  [Press ENTER to continue]")

    print(f"\n{'='*60}")
    print("  Drill complete. You're ready.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        default="all",
        choices=["all", "design", "tradeoffs", "ml", "production"],
        help="Question category to drill",
    )
    args = parser.parse_args()
    drill(args.category)
