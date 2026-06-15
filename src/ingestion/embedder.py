"""
Embedding pipeline.

Model: all-MiniLM-L6-v2
  - 384 dimensions
  - Runs on CPU, no GPU needed
  - 5x faster than OpenAI embeddings for batch workloads
  - Good enough for a strong portfolio project baseline

Senior note: swapping the embedding model is a 1-line config change.
  That's intentional — embeddings are a hyperparameter, not infrastructure.
"""
from __future__ import annotations

import logging
from typing import Generator

import numpy as np
from sentence_transformers import SentenceTransformer

from config import cfg
from src.models import Chunk

logger = logging.getLogger(__name__)


def _batched(items: list, size: int) -> Generator[list, None, None]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class Embedder:
    """
    Encodes text chunks into dense vectors.

    Usage:
        embedder = Embedder()
        vectors = embedder.embed_chunks(chunks)  # np.ndarray shape (N, 384)
    """

    def __init__(self) -> None:
        logger.info(f"Loading embedding model: {cfg.embedding.model_name}")
        self.model = SentenceTransformer(
            cfg.embedding.model_name,
            device=cfg.embedding.device,
        )
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model ready. Dimension: {self.dimension}")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings → (N, D) float32 array."""
        all_vectors = []
        total_batches = (len(texts) + cfg.embedding.batch_size - 1) // cfg.embedding.batch_size

        for i, batch in enumerate(_batched(texts, cfg.embedding.batch_size)):
            vectors = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,   # required for cosine similarity
            )
            all_vectors.append(vectors)
            if (i + 1) % 5 == 0:
                logger.info(f"  Embedded batch {i+1}/{total_batches}")

        return np.vstack(all_vectors).astype(np.float32)

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """Embed Chunk objects using their text field."""
        texts = [c.text for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        vectors = self.embed_texts(texts)
        logger.info(f"Done. Shape: {vectors.shape}")
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string → (1, D) float32 array."""
        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vector.astype(np.float32)
