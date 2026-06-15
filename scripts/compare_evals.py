"""
scripts/compare_evals.py

Loads saved eval summaries and prints a side-by-side comparison table.
This is what you screenshot for your README, LinkedIn post, and interviews.

Usage:
    python scripts/compare_evals.py

Expects:
    evals/summary_naive.json
    evals/summary_adaptive.json    (after Week 2)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EVALS_DIR


METRICS = [
    ("faithfulness",       "Faithfulness      "),
    ("answer_relevancy",   "Answer Relevancy  "),
    ("context_recall",     "Context Recall    "),
    ("context_precision",  "Context Precision "),
    ("overall_score",      "Overall Score     "),
    ("avg_latency_ms",     "Avg Latency (ms)  "),
]


def load_summary(tag: str) -> dict | None:
    path = EVALS_DIR / f"summary_{tag}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def delta_str(naive_val: float, adaptive_val: float, is_latency: bool = False) -> str:
    diff = adaptive_val - naive_val
    if is_latency:
        diff = -diff   # lower is better for latency
    if diff > 0.005:
        return f"  ▲ +{abs(diff):.3f}"
    elif diff < -0.005:
        return f"  ▼ -{abs(diff):.3f}"
    return "  ─ same"


def main() -> None:
    naive = load_summary("naive")
    adaptive = load_summary("adaptive")

    if naive is None:
        print("No baseline found. Run: python scripts/run_eval.py --tag naive")
        sys.exit(1)

    print("\n" + "=" * 72)
    print("  EVALUATION COMPARISON: NAIVE RAG vs ADAPTIVE RAG")
    print("=" * 72)
    print(f"  {'Metric':<22} {'Naive':>10}  {'Adaptive':>10}  {'Change':>16}")
    print("-" * 72)

    for key, label in METRICS:
        naive_val = naive.get(key, 0.0)
        adaptive_val = adaptive.get(key, 0.0) if adaptive else None
        is_latency = "latency" in key

        naive_str = f"{naive_val:.1f}" if is_latency else f"{naive_val:.3f}"

        if adaptive_val is not None:
            adaptive_str = f"{adaptive_val:.1f}" if is_latency else f"{adaptive_val:.3f}"
            change = delta_str(naive_val, adaptive_val, is_latency)
        else:
            adaptive_str = "pending"
            change = "  (run Week 2 evals)"

        print(f"  {label:<22} {naive_str:>10}  {adaptive_str:>10}  {change}")

    print("=" * 72)

    if adaptive:
        overall_improvement = (
            (adaptive["overall_score"] - naive["overall_score"])
            / max(naive["overall_score"], 0.001)
        ) * 100
        print(f"\n  Overall improvement: {overall_improvement:+.1f}%")

    print()


if __name__ == "__main__":
    main()
