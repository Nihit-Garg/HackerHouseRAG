"""
embeddings.py — Sentence-transformer embedding wrapper.

Model: BAAI/bge-small-en-v1.5
  - 33M parameters, 384-dim embeddings
  - Normalised outputs → cosine similarity via inner product (FAISS IndexFlatIP)
  - BGE models require a query prefix for retrieval tasks (see BGE docs)
"""
from __future__ import annotations

import logging
from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# BGE-specific query prefix — required for asymmetric retrieval quality
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingModel:
    """Thin wrapper around SentenceTransformer with BGE-aware query encoding."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._is_bge = "bge" in model_name.lower()

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of document/passage texts.

        Args:
            texts: Raw passage strings (no prefix for BGE passage encoding).
            batch_size: Inference batch size.
            show_progress: Show tqdm progress bar.

        Returns:
            Float32 array of shape (len(texts), dimension), L2-normalised.
        """
        logger.info("Embedding %d documents…", len(texts))
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # required for cosine via inner product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string with BGE query prefix if applicable.

        Args:
            query: The user's search query.

        Returns:
            Float32 array of shape (dimension,), L2-normalised.
        """
        text = (_BGE_QUERY_PREFIX + query) if self._is_bge else query
        embedding = self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0].astype(np.float32)


# ── Module-level singleton (lazy init) ───────────────────────────────────────

_model_instance: EmbeddingModel | None = None


def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> EmbeddingModel:
    """Return the cached EmbeddingModel singleton."""
    global _model_instance
    if _model_instance is None or _model_instance._model_name != model_name:
        _model_instance = EmbeddingModel(model_name)
    return _model_instance
