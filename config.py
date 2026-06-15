"""
Central configuration for Adaptive RAG.
All tunable parameters live here — never scatter magic numbers in code.
"""
from pathlib import Path
from dataclasses import dataclass, field


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVALS_DIR = BASE_DIR / "evals"

# Ensure dirs exist
for d in [RAW_DIR, PROCESSED_DIR, EVALS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class IngestConfig:
    # ArXiv categories to pull from
    arxiv_categories: list[str] = field(default_factory=lambda: [
        "cs.AI", "cs.LG", "cs.CL", "stat.ML"
    ])
    max_papers: int = 500          # target corpus size
    max_per_query: int = 100       # arxiv API page size limit

    # Chunking strategy
    chunk_size: int = 512          # tokens
    chunk_overlap: int = 64        # token overlap between chunks
    min_chunk_length: int = 100    # discard chunks shorter than this (chars)


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"   # fast, good quality, free
    batch_size: int = 64
    device: str = "cpu"


@dataclass
class VectorDBConfig:
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "arxiv_papers"
    vector_size: int = 384          # matches all-MiniLM-L6-v2
    distance: str = "Cosine"


@dataclass
class RetrievalConfig:
    top_k: int = 5                  # chunks to retrieve per query
    score_threshold: float = 0.3    # minimum similarity score


@dataclass
class GenerationConfig:
    model: str = "gemini-2.0-flash"  # free tier, fast
    temperature: float = 0.1         # low temp for factual RAG
    max_output_tokens: int = 1024


@dataclass
class AppConfig:
    ingest: IngestConfig = field(default_factory=IngestConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


# Singleton — import this everywhere
cfg = AppConfig()
