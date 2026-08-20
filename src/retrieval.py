"""
retrieval.py — Hybrid dense (FAISS) + sparse (BM25) retrieval with RRF fusion.

Retrieval latency is logged internally (start/end timestamps) so Day-2
benchmarking can ingest this module without modification.

Return schema per result:
    {text, metadata, dense_score, bm25_score, rrf_score}
"""
from __future__ import annotations

import logging
import time
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from config import RRF_K_CONSTANT, TOP_K
from embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _rrf_score(rank: int, k: int = RRF_K_CONSTANT) -> float:
    """Standard RRF formula: 1 / (k + rank).  rank is 1-indexed."""
    return 1.0 / (k + rank)


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    top_k: int = TOP_K,
    rrf_k: int = RRF_K_CONSTANT,
) -> list[dict[str, Any]]:
    """
    Fuse dense and BM25 result lists via Reciprocal Rank Fusion.

    Args:
        dense_results: Output of query_faiss — [{chunk, score}, …]
        bm25_results:  Output of query_bm25  — [{chunk, score}, …]
        top_k:         Number of fused results to return.
        rrf_k:         RRF constant (default 60 per the original paper).

    Returns:
        List of merged result dicts with all scores populated.
    """
    # chunk_id → accumulated data
    fused: dict[str, dict[str, Any]] = {}

    for rank, item in enumerate(dense_results, start=1):
        cid = item["chunk"]["metadata"]["chunk_id"]
        if cid not in fused:
            fused[cid] = {
                "chunk": item["chunk"],
                "dense_score": 0.0,
                "bm25_score": 0.0,
                "rrf_score": 0.0,
            }
        fused[cid]["dense_score"] = item["score"]
        fused[cid]["rrf_score"] += _rrf_score(rank, rrf_k)

    for rank, item in enumerate(bm25_results, start=1):
        cid = item["chunk"]["metadata"]["chunk_id"]
        if cid not in fused:
            fused[cid] = {
                "chunk": item["chunk"],
                "dense_score": 0.0,
                "bm25_score": 0.0,
                "rrf_score": 0.0,
            }
        fused[cid]["bm25_score"] = item["score"]
        fused[cid]["rrf_score"] += _rrf_score(rank, rrf_k)

    # Sort by RRF score descending, take top_k
    sorted_results = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]

    # Flatten to final schema: {text, metadata, dense_score, bm25_score, rrf_score}
    return [
        {
            "text": r["chunk"]["text"],
            "metadata": r["chunk"]["metadata"],
            "dense_score": r["dense_score"],
            "bm25_score": r["bm25_score"],
            "rrf_score": r["rrf_score"],
        }
        for r in sorted_results
    ]


# ── Retriever class ───────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Hybrid retriever combining FAISS dense search and BM25 sparse search.

    Usage:
        retriever = HybridRetriever(faiss_index, bm25_index, chunks, embedding_model)
        results = retriever.retrieve("what is machine learning?", top_k=5)
    """

    def __init__(
        self,
        faiss_index: faiss.IndexFlatIP,
        bm25_index: BM25Okapi,
        chunks: list[dict[str, Any]],
        embedding_model: EmbeddingModel,
    ) -> None:
        self._faiss = faiss_index
        self._bm25 = bm25_index
        self._chunks = chunks
        self._emb_model = embedding_model

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        rrf_k: int = RRF_K_CONSTANT,
    ) -> dict[str, Any]:
        """
        Run hybrid retrieval for a query.

        Args:
            query: User query string.
            top_k: Number of results to return.
            rrf_k: RRF constant.

        Returns:
            Dict with keys:
              results: list of {text, metadata, dense_score, bm25_score, rrf_score}
              latency: {start_ts, end_ts, elapsed_ms}
        """
        from vectorstore import query_bm25, query_faiss  # local import to avoid circular

        start_ts = time.time()

        # Embed query and run both retrievers in sequence (parallel in Day-2 async)
        query_vec = self._emb_model.embed_query(query)
        dense_results = query_faiss(self._faiss, self._chunks, query_vec, top_k=top_k * 2)
        bm25_results = query_bm25(self._bm25, self._chunks, query, top_k=top_k * 2)

        fused = reciprocal_rank_fusion(dense_results, bm25_results, top_k=top_k, rrf_k=rrf_k)

        end_ts = time.time()
        elapsed_ms = (end_ts - start_ts) * 1000

        logger.debug(
            "Retrieval complete: %d results in %.1f ms (dense=%d, bm25=%d)",
            len(fused), elapsed_ms, len(dense_results), len(bm25_results),
        )

        return {
            "results": fused,
            "latency": {
                "start_ts": start_ts,
                "end_ts": end_ts,
                "elapsed_ms": round(elapsed_ms, 2),
            },
        }
