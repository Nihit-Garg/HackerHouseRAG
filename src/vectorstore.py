"""
vectorstore.py — FAISS index build/load/query + BM25 index.

FAISS index type: IndexFlatIP (exact cosine via inner product on L2-normalised vecs).
BM25 index: rank_bm25 BM25Okapi.

Persisted files:
  data/index/faiss.index    → raw FAISS index
  data/index/faiss_meta.json → ordered list of chunk dicts (parallel to FAISS rows)
  data/index/bm25.pkl       → pickled BM25 object + tokenised corpus
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from config import (
    BM25_INDEX_PATH,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    INDEX_DIR,
)
from embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


# ── FAISS ─────────────────────────────────────────────────────────────────────

def build_faiss_index(
    embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
    save: bool = True,
) -> faiss.IndexFlatIP:
    """
    Build a flat inner-product FAISS index from pre-computed embeddings.

    Args:
        embeddings: Float32 array of shape (N, dim), L2-normalised.
        chunks: Parallel list of chunk dicts (same order as embeddings).
        save: If True, persist index and metadata to disk.

    Returns:
        Built FAISS IndexFlatIP.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)  # type: ignore[arg-type]
    logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)

    if save:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
        logger.info("FAISS index saved → %s", FAISS_INDEX_PATH)

    return index


def load_faiss_index() -> tuple[faiss.IndexFlatIP, list[dict[str, Any]]]:
    """
    Load FAISS index and metadata from disk.

    Returns:
        (index, chunks) tuple.

    Raises:
        FileNotFoundError: If index files do not exist.
    """
    if not FAISS_INDEX_PATH.exists() or not FAISS_META_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. Run scripts/build_index.py first."
        )
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(FAISS_META_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("FAISS index loaded: %d vectors", index.ntotal)
    return index, chunks


def query_faiss(
    index: faiss.IndexFlatIP,
    chunks: list[dict[str, Any]],
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Run a nearest-neighbour search on the FAISS index.

    Args:
        index: Loaded FAISS index.
        chunks: Parallel chunk metadata list.
        query_embedding: 1-D float32 array of shape (dim,).
        top_k: Number of results to return.

    Returns:
        List of dicts: {chunk, score (inner product / cosine)}
    """
    query_vec = query_embedding.reshape(1, -1).astype(np.float32)
    scores, indices = index.search(query_vec, top_k)  # type: ignore[arg-type]

    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:  # FAISS returns -1 for padding
            continue
        results.append({"chunk": chunks[idx], "score": float(score)})
    return results


# ── BM25 ──────────────────────────────────────────────────────────────────────

def _tokenise_for_bm25(text: str) -> list[str]:
    """Simple whitespace tokeniser for BM25 (lower-cased)."""
    return text.lower().split()


def build_bm25_index(
    chunks: list[dict[str, Any]],
    save: bool = True,
) -> BM25Okapi:
    """
    Build a BM25Okapi index over the chunk corpus.

    Args:
        chunks: List of chunk dicts (must have 'text' key).
        save: If True, persist to disk.

    Returns:
        Fitted BM25Okapi instance.
    """
    corpus = [_tokenise_for_bm25(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)

    if save:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({"bm25": bm25, "corpus": corpus}, f)
        logger.info("BM25 index saved → %s", BM25_INDEX_PATH)

    return bm25


def load_bm25_index() -> BM25Okapi:
    """
    Load BM25 index from disk.

    Raises:
        FileNotFoundError: If BM25 pickle not found.
    """
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_INDEX_PATH}. Run scripts/build_index.py first."
        )
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    logger.info("BM25 index loaded.")
    return data["bm25"]


def query_bm25(
    bm25: BM25Okapi,
    chunks: list[dict[str, Any]],
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Run a BM25 search.

    Args:
        bm25: Fitted BM25Okapi.
        chunks: Parallel chunk list (same order as BM25 corpus).
        query: Raw query string.
        top_k: Number of results to return.

    Returns:
        List of dicts: {chunk, score}
    """
    tokenised_query = _tokenise_for_bm25(query)
    scores = bm25.get_scores(tokenised_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {"chunk": chunks[int(i)], "score": float(scores[i])}
        for i in top_indices
        if scores[i] > 0.0
    ]


# ── Convenience: build both from chunks ──────────────────────────────────────

def build_all_indexes(
    chunks: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
) -> tuple[faiss.IndexFlatIP, BM25Okapi]:
    """
    Embed chunks, build FAISS + BM25 indexes, and save both to disk.

    Returns:
        (faiss_index, bm25_index)
    """
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.embed_documents(texts, show_progress=True)
    faiss_index = build_faiss_index(embeddings, chunks, save=True)
    bm25_index = build_bm25_index(chunks, save=True)
    return faiss_index, bm25_index
