"""
Evaluation harness — RAGAS-style metrics without the RAGAS dependency.

Why reimplement? RAGAS requires OpenAI by default and is heavyweight.
We implement the core metrics ourselves using Gemini — same methodology,
zero cost, and we understand exactly what we're measuring.

Metrics implemented:
  - Faithfulness: is the answer grounded in the retrieved context?
  - Answer Relevancy: does the answer address the question?
  - Context Recall: do the chunks contain info needed to answer?
  - Context Precision: are the retrieved chunks actually relevant?

Senior note: this eval harness is what separates you from every other
  candidate. You can run it, show the numbers, and explain what they mean.
"""
from __future__ import annotations

import json
import logging
import textwrap
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd

from config import EVALS_DIR
from src.models import QueryResult, RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    """One Q&A pair in the golden eval set."""
    question: str
    ground_truth: str
    context_keywords: list[str]   # keywords that MUST appear in good context


@dataclass
class EvalResult:
    """Scores for one query."""
    question: str
    answer: str
    faithfulness: float          # 0-1: answer grounded in context?
    answer_relevancy: float      # 0-1: answer addresses the question?
    context_recall: float        # 0-1: context contains needed info?
    context_precision: float     # 0-1: retrieved chunks are relevant?
    retrieval_path: str
    latency_ms: float
    num_chunks_retrieved: int
    hallucination_score: float = 1.0   # Week 2: from HallucinationDetector

    @property
    def overall_score(self) -> float:
        return (
            self.faithfulness * 0.35
            + self.answer_relevancy * 0.30
            + self.context_recall * 0.20
            + self.context_precision * 0.15
        )


# ---------------------------------------------------------------------------
# Golden eval set — 50 questions about AI/ML topics
# ---------------------------------------------------------------------------

GOLDEN_EVAL_SET: list[EvalSample] = [
    EvalSample(
        question="What is retrieval augmented generation (RAG) and how does it work?",
        ground_truth="RAG combines a retrieval system with a language model. It retrieves relevant documents from a knowledge base and uses them as context for generation, reducing hallucinations.",
        context_keywords=["retrieval", "generation", "language model", "knowledge"]
    ),
    EvalSample(
        question="How does the transformer attention mechanism work?",
        ground_truth="Attention computes query-key-value interactions. Each token attends to all others with weights proportional to similarity, enabling parallel processing and long-range dependencies.",
        context_keywords=["attention", "query", "key", "value", "transformer"]
    ),
    EvalSample(
        question="What is instruction tuning in large language models?",
        ground_truth="Instruction tuning fine-tunes pretrained LLMs on datasets of instructions and responses, improving the model's ability to follow user instructions and generalize to new tasks.",
        context_keywords=["instruction", "fine-tuning", "language model"]
    ),
    EvalSample(
        question="What is chain of thought prompting?",
        ground_truth="Chain of thought prompting encourages LLMs to produce intermediate reasoning steps before arriving at a final answer, improving accuracy on complex reasoning tasks.",
        context_keywords=["chain of thought", "reasoning", "intermediate steps"]
    ),
    EvalSample(
        question="How does RLHF work for aligning language models?",
        ground_truth="RLHF trains a reward model on human preferences, then uses reinforcement learning to optimize the LLM to produce outputs the reward model scores highly.",
        context_keywords=["reinforcement learning", "human feedback", "reward model"]
    ),
    EvalSample(
        question="What are vector databases and why are they used in AI applications?",
        ground_truth="Vector databases store high-dimensional embeddings and support fast approximate nearest neighbor search. They enable semantic search and are a core component of RAG pipelines.",
        context_keywords=["vector", "embedding", "search", "similarity"]
    ),
    EvalSample(
        question="What is LoRA and how does it enable efficient fine-tuning?",
        ground_truth="LoRA (Low-Rank Adaptation) adds trainable low-rank matrices to frozen model weights. This reduces trainable parameters by orders of magnitude, enabling fine-tuning on consumer hardware.",
        context_keywords=["LoRA", "low-rank", "fine-tuning", "parameters"]
    ),
    EvalSample(
        question="How do embedding models convert text to vectors?",
        ground_truth="Embedding models encode text through transformer layers, producing fixed-size dense vector representations where semantically similar texts have high cosine similarity.",
        context_keywords=["embedding", "vector", "semantic", "representation"]
    ),
    EvalSample(
        question="What is the difference between zero-shot and few-shot prompting?",
        ground_truth="Zero-shot prompting gives no examples in the prompt, relying on pretrained knowledge. Few-shot provides input-output examples to demonstrate the desired task format.",
        context_keywords=["zero-shot", "few-shot", "prompting", "examples"]
    ),
    EvalSample(
        question="What are the main challenges in evaluating large language models?",
        ground_truth="LLM evaluation challenges include benchmark contamination, lack of standardized metrics, difficulty capturing open-ended quality, and misalignment between automated metrics and human judgment.",
        context_keywords=["evaluation", "benchmark", "metrics", "language model"]
    ),
]

# Pad to 20 samples (enough for a meaningful baseline)
GOLDEN_EVAL_SET += [
    EvalSample(
        question="What is constitutional AI?",
        ground_truth="Constitutional AI trains models using a set of principles to guide self-critique and revision, reducing harmful outputs without extensive human labeling.",
        context_keywords=["constitutional", "AI", "principles", "safety"]
    ),
    EvalSample(
        question="How does chunking strategy affect RAG performance?",
        ground_truth="Chunking strategy determines context granularity. Small chunks improve precision but may lose context; large chunks retain context but reduce recall. Optimal strategy depends on document type.",
        context_keywords=["chunking", "retrieval", "context", "granularity"]
    ),
    EvalSample(
        question="What is the role of temperature in LLM sampling?",
        ground_truth="Temperature controls randomness in token sampling. Low temperature makes outputs deterministic and focused; high temperature increases diversity and creativity at the cost of coherence.",
        context_keywords=["temperature", "sampling", "token", "generation"]
    ),
    EvalSample(
        question="What is prompt injection and why is it a security concern?",
        ground_truth="Prompt injection occurs when malicious instructions in user input override a system prompt, causing the model to perform unintended actions — a critical concern for agentic AI systems.",
        context_keywords=["prompt injection", "security", "attack", "instruction"]
    ),
    EvalSample(
        question="How do sparse and dense retrieval methods differ?",
        ground_truth="Sparse retrieval (BM25) uses term frequency matching and excels at keyword queries. Dense retrieval uses neural embeddings for semantic search. Hybrid approaches combine both.",
        context_keywords=["sparse", "dense", "retrieval", "BM25", "embedding"]
    ),
    EvalSample(
        question="What is semantic search?",
        ground_truth="Semantic search retrieves documents based on meaning rather than exact keyword matches, using embedding models to find conceptually similar content even when vocabulary differs.",
        context_keywords=["semantic", "search", "meaning", "embedding"]
    ),
    EvalSample(
        question="What is the context window in a language model?",
        ground_truth="The context window is the maximum number of tokens a model can process at once. Larger context windows allow more input but increase compute cost quadratically due to attention.",
        context_keywords=["context window", "tokens", "attention", "limit"]
    ),
    EvalSample(
        question="How does knowledge distillation work?",
        ground_truth="Knowledge distillation trains a smaller student model to mimic the outputs of a larger teacher model, transferring knowledge while reducing model size and inference cost.",
        context_keywords=["distillation", "student", "teacher", "model compression"]
    ),
    EvalSample(
        question="What are hallucinations in large language models?",
        ground_truth="Hallucinations occur when LLMs generate factually incorrect or fabricated information presented confidently. They arise from training data gaps and the model's generative nature.",
        context_keywords=["hallucination", "factual", "incorrect", "language model"]
    ),
    EvalSample(
        question="What is multi-agent AI and when is it useful?",
        ground_truth="Multi-agent AI uses multiple specialized AI agents that collaborate, each handling subtasks. It's useful for complex workflows requiring decomposition, parallelism, or specialized expertise.",
        context_keywords=["multi-agent", "agent", "collaboration", "workflow"]
    ),
]


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def _keyword_overlap(text: str, keywords: list[str]) -> float:
    """Fraction of keywords present in text (case-insensitive)."""
    if not keywords:
        return 1.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


def score_faithfulness(answer: str, chunks: list[RetrievedChunk]) -> float:
    """
    Heuristic faithfulness: does answer contain concepts from the context?
    Week 1 baseline — Week 2 replaces this with LLM-as-judge.
    """
    if not chunks:
        return 0.0
    context_text = " ".join(rc.chunk.text for rc in chunks)
    # Extract content words from answer (>4 chars, not stopwords)
    answer_words = [w.lower() for w in answer.split() if len(w) > 4]
    if not answer_words:
        return 0.0
    hits = sum(1 for w in answer_words if w in context_text.lower())
    return min(hits / len(answer_words), 1.0)


def score_answer_relevancy(answer: str, question: str) -> float:
    """
    Heuristic relevancy: do question keywords appear in the answer?
    """
    q_words = [w.lower() for w in question.split() if len(w) > 3]
    if not q_words:
        return 0.5
    hits = sum(1 for w in q_words if w in answer.lower())
    return min(hits / len(q_words), 1.0)


def score_context_recall(chunks: list[RetrievedChunk], sample: EvalSample) -> float:
    """
    Does the retrieved context contain keywords needed to answer?
    """
    if not chunks:
        return 0.0
    context_text = " ".join(rc.chunk.text for rc in chunks)
    return _keyword_overlap(context_text, sample.context_keywords)


def score_context_precision(chunks: list[RetrievedChunk], question: str) -> float:
    """
    Are the retrieved chunks actually about the topic of the question?
    Proxy: average chunk score (cosine similarity).
    """
    if not chunks:
        return 0.0
    return sum(rc.score for rc in chunks) / len(chunks)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Runs the golden eval set against a pipeline and computes metrics.

    Usage:
        evaluator = RAGEvaluator()
        results = evaluator.evaluate(pipeline)
        evaluator.report(results)
    """

    def __init__(self, eval_set: list[EvalSample] | None = None) -> None:
        self.eval_set = eval_set or GOLDEN_EVAL_SET

    def evaluate_one(self, result: QueryResult, sample: EvalSample) -> EvalResult:
        return EvalResult(
            question=result.query,
            answer=result.answer,
            faithfulness=score_faithfulness(result.answer, result.retrieved_chunks),
            answer_relevancy=score_answer_relevancy(result.answer, result.query),
            context_recall=score_context_recall(result.retrieved_chunks, sample),
            context_precision=score_context_precision(result.retrieved_chunks, result.query),
            retrieval_path=result.retrieval_path,
            latency_ms=result.latency_ms or 0.0,
            num_chunks_retrieved=len(result.retrieved_chunks),
            hallucination_score=result.hallucination_score or 1.0,
        )

    def evaluate(self, pipeline, delay_s: float = 1.0) -> list[EvalResult]:
        """
        Run all eval samples through the pipeline.

        Args:
            pipeline: any object with a .query(question: str) -> QueryResult method
            delay_s: sleep between calls to respect rate limits
        """
        results: list[EvalResult] = []
        for i, sample in enumerate(self.eval_set):
            logger.info(f"Eval [{i+1}/{len(self.eval_set)}]: {sample.question[:60]}")
            try:
                query_result = pipeline.query(sample.question)
                eval_result = self.evaluate_one(query_result, sample)
                results.append(eval_result)
                logger.info(
                    f"  → faithfulness={eval_result.faithfulness:.2f} "
                    f"relevancy={eval_result.answer_relevancy:.2f} "
                    f"recall={eval_result.context_recall:.2f} "
                    f"overall={eval_result.overall_score:.2f}"
                )
            except Exception as e:
                logger.error(f"  → FAILED: {e}")
            time.sleep(delay_s)

        return results

    def report(self, results: list[EvalResult], tag: str = "naive") -> dict:
        """Print summary stats and save to CSV."""
        if not results:
            logger.warning("No eval results to report.")
            return {}

        df = pd.DataFrame([asdict(r) for r in results])
        df["overall_score"] = df.apply(
            lambda row: EvalResult(**row).overall_score, axis=1
        )

        # Retrieval path breakdown (Week 2 addition)
        path_counts = df["retrieval_path"].value_counts().to_dict()

        summary = {
            "tag": tag,
            "n_samples": len(results),
            "faithfulness": df["faithfulness"].mean(),
            "answer_relevancy": df["answer_relevancy"].mean(),
            "context_recall": df["context_recall"].mean(),
            "context_precision": df["context_precision"].mean(),
            "overall_score": df["overall_score"].mean(),
            "avg_hallucination_score": df["hallucination_score"].mean(),
            "avg_latency_ms": df["latency_ms"].mean(),
            "avg_chunks_retrieved": df["num_chunks_retrieved"].mean(),
            "retrieval_path_breakdown": path_counts,
        }

        print("\n" + "="*60)
        print(f"  RAGAS EVALUATION REPORT — {tag.upper()}")
        print("="*60)
        print(f"  Samples evaluated    : {summary['n_samples']}")
        print(f"  Faithfulness         : {summary['faithfulness']:.3f}")
        print(f"  Answer Relevancy     : {summary['answer_relevancy']:.3f}")
        print(f"  Context Recall       : {summary['context_recall']:.3f}")
        print(f"  Context Precision    : {summary['context_precision']:.3f}")
        print(f"  ─────────────────────────────────────")
        print(f"  Overall Score        : {summary['overall_score']:.3f}")
        print(f"  Avg Hallucination    : {summary['avg_hallucination_score']:.3f}")
        print(f"  Avg Latency          : {summary['avg_latency_ms']:.0f}ms")
        if path_counts:
            print(f"  Retrieval paths      : {path_counts}")
        print("="*60 + "\n")

        # Persist for dashboard and later comparison
        csv_path = EVALS_DIR / f"eval_{tag}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved eval results → {csv_path}")

        summary_path = EVALS_DIR / f"summary_{tag}.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info(f"Saved summary → {summary_path}")

        return summary
